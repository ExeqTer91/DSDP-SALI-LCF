"""
Pythia Temporal Emergence Experiment
=====================================
Tests whether H1 (temporal coherence) and topological separation emerge
during neural network training, using EleutherAI's Pythia checkpoints.

Pythia provides public checkpoints at regular intervals from random init
to fully trained. We extract hidden states from the same corpus at each
checkpoint and measure emergence metrics across the training trajectory.

Requirements:
    pip install torch transformers numpy scipy scikit-learn

Hardware:
    - Pythia-70m: runs on CPU or any GPU (< 1GB VRAM)
    - Pythia-160m: any GPU (< 2GB VRAM)
    - Pythia-410m: GPU with 4GB+ VRAM
    
    Replit can handle 70m on CPU. RunPod recommended for 160m+.

Usage:
    python pythia_temporal_emergence.py

Output:
    results/pythia_emergence_results.json
    results/pythia_emergence_plot_data.json
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Model: start small. 70m runs anywhere.
    "model_name": "EleutherAI/pythia-70m",
    
    # Checkpoints to sample across training trajectory
    # Pythia saves at: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    # 1000, 2000, ..., 143000
    # We sample ~15 points spanning the full range
    "checkpoints": [0, 1, 2, 8, 32, 128, 512, 1000, 2000, 4000,
                    8000, 16000, 32000, 64000, 143000],
    
    # Layer to extract (middle layer captures intermediate representations)
    # Pythia-70m has 6 layers; we use layer 3 (0-indexed)
    "extract_layer": 3,
    
    # Number of sentences for embedding extraction
    "n_sentences": 2000,
    
    # H1 parameters (matching paper)
    "h1_window": 32,
    "h1_n_shuffles": 500,
    "h1_r1_threshold": 0.2,
    "h1_p_threshold": 0.05,
    
    # Topological parameters
    "n_pca_dims": 10,
    "n_tau_steps": 20,
    
    # sqrt(phi) lattice step for alignment
    "phi": (1 + np.sqrt(5)) / 2,
    
    # Output directory
    "output_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "pythia"),
    
    # Random seed
    "seed": 42,
}

CONFIG["lattice_step"] = np.log(np.sqrt(CONFIG["phi"]))  # ~0.2406

# ============================================================
# CORPUS GENERATION
# ============================================================

def get_corpus(n_sentences):
    """
    Generate a simple ordered corpus for consistent extraction.
    Uses numbered sentences to ensure reproducible token sequences
    across all checkpoints.
    """
    # Simple deterministic corpus - same tokens every time
    corpus = []
    topics = [
        "mathematics", "physics", "biology", "chemistry", "history",
        "philosophy", "economics", "psychology", "linguistics", "astronomy",
        "geology", "medicine", "engineering", "literature", "music",
        "architecture", "ecology", "neuroscience", "genetics", "statistics"
    ]
    templates = [
        "The study of {topic} reveals fundamental principles about the natural world.",
        "Researchers in {topic} have discovered important patterns and relationships.",
        "Understanding {topic} requires careful observation and systematic analysis.",
        "The history of {topic} shows how knowledge accumulates over centuries.",
        "Modern advances in {topic} continue to transform our understanding.",
    ]
    
    i = 0
    while len(corpus) < n_sentences:
        topic = topics[i % len(topics)]
        template = templates[i % len(templates)]
        corpus.append(template.format(topic=topic))
        i += 1
    
    return corpus[:n_sentences]


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

def extract_hidden_states(model_name, checkpoint_step, corpus, layer_idx, device):
    """
    Extract hidden states from a specific Pythia checkpoint.
    Returns: np.array of shape (n_sentences, hidden_dim)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    revision = f"step{checkpoint_step}"
    print(f"  Loading {model_name} @ {revision}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        revision=revision,
        output_hidden_states=True
    ).to(device)
    model.eval()
    
    all_embeddings = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i+batch_size]
            inputs = tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=64
            ).to(device)
            
            outputs = model(**inputs)
            # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_dim)
            # Take the specified layer, mean-pool over sequence length
            hidden = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
            
            # Mean pool (ignoring padding)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            
            all_embeddings.append(pooled.cpu().numpy())
    
    # Cleanup
    del model
    if device != "cpu":
        import torch
        torch.cuda.empty_cache()
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"    Extracted: {embeddings.shape}")
    return embeddings


# ============================================================
# METRICS (matching paper exactly)
# ============================================================

def compute_h1(embeddings, window=32, n_shuffles=1000, seed=42):
    """
    H1: Temporal coherence via lag-1 autocorrelation.
    Conjunctive criterion: p < 0.05 AND |r1| > 0.2
    """
    from scipy.stats import pearsonr
    
    n = len(embeddings)
    if n < window * 3:
        return {"r1": 0.0, "p": 1.0, "pass": False, "reason": "insufficient_data"}
    
    # Compute structural signal: rolling entropy proxy (pairwise distances)
    signal = []
    for i in range(0, n - window, window // 2):
        chunk = embeddings[i:i+window]
        # Use mean pairwise distance as structural signal
        dists = np.sqrt(np.sum((chunk[:, None] - chunk[None, :]) ** 2, axis=-1))
        signal.append(np.mean(dists[np.triu_indices(len(chunk), k=1)]))
    
    signal = np.array(signal)
    if len(signal) < 10:
        return {"r1": 0.0, "p": 1.0, "pass": False, "reason": "insufficient_windows"}
    
    # Lag-1 autocorrelation
    r1 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
    
    # Permutation test
    rng = np.random.RandomState(seed)
    null_r1s = []
    for _ in range(n_shuffles):
        shuffled = signal.copy()
        rng.shuffle(shuffled)
        null_r1s.append(np.corrcoef(shuffled[:-1], shuffled[1:])[0, 1])
    
    null_r1s = np.array(null_r1s)
    p = np.mean(np.abs(null_r1s) >= np.abs(r1))
    
    passes = (p < 0.05) and (abs(r1) > 0.2)
    
    return {
        "r1": float(r1),
        "p": float(p),
        "pass": bool(passes),
        "n_windows": len(signal),
    }


def compute_topo_sep(embeddings, n_pca_dims=10, n_tau_steps=20, seed=42):
    """
    Topological separation: Euler characteristic transitions in 
    structured vs random regime. Uses subsampling for efficiency.
    """
    from sklearn.decomposition import PCA
    
    rng = np.random.RandomState(seed)
    
    n_subsample = min(200, len(embeddings))
    idx = rng.choice(len(embeddings), n_subsample, replace=False)
    sub = embeddings[idx]
    
    pca = PCA(n_components=min(n_pca_dims, sub.shape[1]))
    projected = pca.fit_transform(sub)
    
    chi_real = _euler_curve(projected, n_tau_steps)
    
    shuffled = projected.copy()
    for d in range(shuffled.shape[1]):
        rng.shuffle(shuffled[:, d])
    chi_random = _euler_curve(shuffled, n_tau_steps)
    
    transitions_real = np.sum(np.diff(np.sign(chi_real)) != 0)
    transitions_random = np.sum(np.diff(np.sign(chi_random)) != 0)
    
    topo_sep = (transitions_real - transitions_random) / max(n_tau_steps, 1)
    
    return {
        "topo_sep": float(topo_sep),
        "transitions_real": int(transitions_real),
        "transitions_random": int(transitions_random),
    }


def _euler_curve(data, n_steps):
    """Compute Euler characteristic curve chi(tau) = C0(tau) - C1(tau)."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    
    dists = pdist(data)
    tau_min, tau_max = np.percentile(dists, [5, 95])
    taus = np.linspace(tau_min, tau_max, n_steps)
    dist_sq = squareform(dists)
    n = data.shape[0]
    
    chi = []
    for tau in taus:
        adj_mat = csr_matrix(dist_sq <= tau)
        c0, _ = connected_components(adj_mat, directed=False)
        n_edges = int(np.sum(dist_sq <= tau) - n) // 2
        c1 = max(0, n_edges - (n - c0))
        chi.append(c0 - c1)
    
    return np.array(chi)


def compute_alignment(embeddings, lattice_step, tau=0.03):
    """
    Lattice alignment: fraction of points within tau of nearest
    sqrt(phi)-lattice node after PCA projection.
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=min(3, embeddings.shape[1]))
    projected = pca.fit_transform(embeddings)
    
    # Distance to nearest lattice node
    residuals = np.mod(projected, lattice_step)
    residuals = np.minimum(residuals, lattice_step - residuals)
    min_residuals = np.min(residuals, axis=1)
    
    alignment = np.mean(min_residuals < tau * lattice_step)
    return float(alignment)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Checkpoints: {CONFIG['checkpoints']}")
    print(f"Layer: {CONFIG['extract_layer']}")
    print(f"Sentences: {CONFIG['n_sentences']}")
    print()
    
    # Setup
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    np.random.seed(CONFIG["seed"])
    corpus = get_corpus(CONFIG["n_sentences"])
    
    results = {
        "config": {k: v for k, v in CONFIG.items() if not isinstance(v, np.floating)},
        "checkpoints": {},
        "predictions": {
            "H1": "H1 should transition from FAIL to PASS as training progresses (temporal coherence emerges)",
            "topo_sep": "topo_sep should increase monotonically (topological complexity increases with constraint tightening)",
            "alignment": "alignment should increase (learned representations increasingly exhibit lattice-compatible geometry)",
        }
    }
    
    for step in CONFIG["checkpoints"]:
        print(f"\n{'='*60}")
        print(f"Checkpoint: step {step}")
        print(f"{'='*60}")
        
        t0 = time.time()
        
        try:
            # Extract embeddings
            embeddings = extract_hidden_states(
                CONFIG["model_name"],
                step,
                corpus,
                CONFIG["extract_layer"],
                device
            )
            
            # Compute metrics
            print("  Computing H1...")
            h1 = compute_h1(
                embeddings,
                window=CONFIG["h1_window"],
                n_shuffles=CONFIG["h1_n_shuffles"],
                seed=CONFIG["seed"]
            )
            
            print("  Computing topo_sep...")
            topo = compute_topo_sep(
                embeddings,
                n_pca_dims=CONFIG["n_pca_dims"],
                n_tau_steps=CONFIG["n_tau_steps"],
                seed=CONFIG["seed"]
            )
            
            print("  Computing alignment...")
            alignment = compute_alignment(
                embeddings,
                CONFIG["lattice_step"]
            )
            
            elapsed = time.time() - t0
            
            checkpoint_result = {
                "step": step,
                "h1": h1,
                "topo_sep": topo,
                "alignment": alignment,
                "embedding_shape": list(embeddings.shape),
                "elapsed_seconds": round(elapsed, 1),
            }
            
            results["checkpoints"][str(step)] = checkpoint_result
            
            print(f"\n  Results @ step {step}:")
            print(f"    H1: r1={h1['r1']:.3f}, p={h1['p']:.3f}, pass={h1['pass']}")
            print(f"    topo_sep: {topo['topo_sep']:.4f}")
            print(f"    alignment: {alignment:.4f}")
            print(f"    time: {elapsed:.1f}s")
            
        except Exception as e:
            print(f"  ERROR at step {step}: {e}")
            results["checkpoints"][str(step)] = {"step": step, "error": str(e)}
        
        # Save incrementally
        with open(os.path.join(CONFIG["output_dir"], "pythia_emergence_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    steps = []
    h1_r1s = []
    h1_passes = []
    topo_seps = []
    alignments = []
    
    for step_key, res in results["checkpoints"].items():
        if "error" in res:
            continue
        steps.append(res["step"])
        h1_r1s.append(res["h1"]["r1"])
        h1_passes.append(res["h1"]["pass"])
        topo_seps.append(res["topo_sep"]["topo_sep"])
        alignments.append(res["alignment"])
    
    print(f"\n{'Step':>8} {'H1_r1':>8} {'H1_pass':>8} {'topo_sep':>10} {'alignment':>10}")
    print("-" * 50)
    for i in range(len(steps)):
        print(f"{steps[i]:>8} {h1_r1s[i]:>8.3f} {str(h1_passes[i]):>8} {topo_seps[i]:>10.4f} {alignments[i]:>10.4f}")
    
    # Check predictions
    print(f"\n\nPREDICTION CHECKS:")
    
    # H1: should transition FAIL -> PASS
    first_pass = None
    for i, p in enumerate(h1_passes):
        if p:
            first_pass = steps[i]
            break
    if first_pass is not None:
        print(f"  H1 FAIL->PASS transition: step {first_pass} ✓")
    else:
        print(f"  H1 FAIL->PASS transition: not observed ✗")
    
    # topo_sep: should trend upward
    if len(topo_seps) > 2:
        early = np.mean(topo_seps[:3])
        late = np.mean(topo_seps[-3:])
        trend = "increasing ✓" if late > early else "not increasing ✗"
        print(f"  topo_sep trend: early={early:.4f} → late={late:.4f} ({trend})")
    
    # alignment: should trend upward
    if len(alignments) > 2:
        early = np.mean(alignments[:3])
        late = np.mean(alignments[-3:])
        trend = "increasing ✓" if late > early else "not increasing ✗"
        print(f"  alignment trend: early={early:.4f} → late={late:.4f} ({trend})")
    
    # Save plot data
    plot_data = {
        "steps": steps,
        "h1_r1": h1_r1s,
        "h1_pass": h1_passes,
        "topo_sep": topo_seps,
        "alignment": alignments,
    }
    with open(os.path.join(CONFIG["output_dir"], "pythia_emergence_plot_data.json"), "w") as f:
        json.dump(plot_data, f, indent=2)
    
    print(f"\nResults saved to {CONFIG['output_dir']}/")
    print("Done.")


if __name__ == "__main__":
    run_experiment()
