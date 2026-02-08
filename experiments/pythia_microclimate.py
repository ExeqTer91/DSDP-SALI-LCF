"""
Pythia Micro-Climate Experiment
================================
Tests emergence metrics at TOKEN-LEVEL granularity instead of sentence-level.

KEY INSIGHT: Autoregressive models don't organize information per-sentence.
They create LOCAL structure within context windows ("micro-climates").
Mean-pooling per sentence destroys this signal. 

Instead, we extract the hidden state at EACH TOKEN POSITION from a single
long forward pass, creating a natural temporal sequence of representations.
The "time" axis is token position — each successive token accumulates more
context, tightening the local constraint.

Three extraction strategies:
1. LAST-TOKEN: hidden state at position t (accumulates context 0..t)
2. ALL-TOKENS: raw hidden states at each position (no pooling)
3. WINDOWED: mean-pool over sliding windows of K tokens (intermediate scale)

We test at multiple checkpoints: step 0 (random) vs step 143000 (trained).
If micro-climate hypothesis is correct:
- Trained model shows H1 PASS (temporal coherence in token flow)
- Random model shows H1 FAIL (no coherent token flow)
- topo_sep positive for trained, near-zero for random

Requirements:
    pip install torch transformers numpy scipy scikit-learn

Hardware:
    Pythia-70m runs on CPU. GPU recommended for speed.
    
Usage:
    python pythia_microclimate.py
"""

import json
import os
import time
import numpy as np
from pathlib import Path


CONFIG = {
    "model_name": "EleutherAI/pythia-70m",
    "checkpoints": [0, 143000],
    "layers": [1, 3, 5],
    "seq_length": 1024,
    "window_sizes": [8, 16, 32],
    "h1_window": 32,
    "h1_n_shuffles": 1000,
    "h1_r1_threshold": 0.2,
    "h1_p_threshold": 0.05,
    "n_pca_dims": 10,
    "n_tau_steps": 50,
    "phi": (1 + np.sqrt(5)) / 2,
    "output_dir": "outputs/pythia_microclimate",
    "seed": 42,
}

CONFIG["lattice_step"] = np.log(np.sqrt(CONFIG["phi"]))


def get_long_text():
    """
    Generate a long, coherent text for a single forward pass.
    Uses a deterministic, topically structured sequence.
    """
    paragraphs = [
        "Mathematics provides the foundation for understanding patterns in nature. "
        "Numbers and their relationships reveal deep structures that govern physical reality. "
        "From the geometry of crystals to the dynamics of planetary orbits, mathematical "
        "principles constrain the space of possible configurations.",
        
        "Physics extends these mathematical structures into empirical territory. "
        "The conservation laws that govern energy and momentum are constraints that "
        "shape every physical process. Temperature, pressure, and density interact "
        "through equations of state that define accessible configurations.",
        
        "Biology demonstrates how constraints generate complexity. Genetic codes "
        "constrain protein folding into specific geometric configurations. Natural "
        "selection acts as an environmental constraint that shapes phenotypic space. "
        "Convergent evolution shows that similar constraints produce similar solutions.",
        
        "Chemistry reveals how atomic constraints determine molecular geometry. "
        "Electron orbital configurations constrain bonding patterns and reaction "
        "pathways. The periodic table itself is a map of constraint boundaries "
        "that govern elemental properties and chemical behavior.",
        
        "Neuroscience studies how neural constraints shape cognition and behavior. "
        "Synaptic connections constrain information flow through neural circuits. "
        "Attention mechanisms act as dynamic constraints that filter and organize "
        "sensory input into coherent representations of the external world.",
        
        "Economics examines how resource constraints shape market dynamics and "
        "institutional behavior. Supply and demand interact through price mechanisms "
        "that constrain exchange. Budget constraints force optimization decisions "
        "that reveal preferences and priorities in resource allocation.",
        
        "Ecology studies how environmental constraints shape species distributions "
        "and community structure. Temperature gradients create biogeographic zones. "
        "Nutrient availability constrains primary productivity. Competition and "
        "predation constrain population dynamics within carrying capacity limits.",
        
        "Linguistics reveals how grammatical constraints shape language structure. "
        "Phonological rules constrain sound combinations. Syntactic structures "
        "constrain word ordering. Semantic constraints ensure meaningful composition. "
        "These layers of constraint produce the infinite creativity of natural language.",
        
        "Geology demonstrates how physical constraints shape planetary surfaces. "
        "Tectonic forces constrain continental configurations over geological time. "
        "Erosion and deposition are constrained by gravity, fluid dynamics, and "
        "material properties. The landscape is a record of accumulated constraints.",
        
        "Astronomy extends constraint principles to cosmic scales. Gravitational "
        "constraints determine orbital mechanics and stellar evolution. Nuclear "
        "physics constrains the life cycles of stars. Cosmological parameters "
        "constrain the large-scale structure of the observable universe.",
    ]
    
    full_text = " ".join(paragraphs * 5)
    return full_text


def extract_token_hidden_states(model_name, checkpoint_step, text, layers, seq_length, device):
    """
    Extract hidden states at EVERY token position from a single forward pass.
    
    Returns: dict of {layer_idx: np.array of shape (seq_length, hidden_dim)}
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
    
    tokens = tokenizer(text, return_tensors="pt", truncation=True, 
                       max_length=seq_length)
    actual_length = tokens["input_ids"].shape[1]
    print(f"    Token sequence length: {actual_length}")
    
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    with torch.no_grad():
        outputs = model(**tokens)
    
    result = {}
    for layer_idx in layers:
        hs = outputs.hidden_states[layer_idx][0].cpu().numpy()
        result[layer_idx] = hs
        print(f"    Layer {layer_idx}: {hs.shape}")
    
    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    
    return result, actual_length


def compute_h1(embeddings, window=32, n_shuffles=1000, seed=42):
    """H1: temporal coherence via lag-1 autocorrelation on token sequence."""
    n = len(embeddings)
    if n < window * 3:
        return {"r1": 0.0, "p": 1.0, "pass": False, "reason": "insufficient_data"}
    
    signal = []
    for i in range(0, n - window, window // 2):
        chunk = embeddings[i:i+window]
        dists = np.sqrt(np.sum((chunk[:, None] - chunk[None, :]) ** 2, axis=-1))
        signal.append(np.mean(dists[np.triu_indices(len(chunk), k=1)]))
    
    signal = np.array(signal)
    if len(signal) < 10:
        return {"r1": 0.0, "p": 1.0, "pass": False, "reason": "insufficient_windows"}
    
    r1 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
    
    rng = np.random.RandomState(seed)
    null_r1s = []
    for _ in range(n_shuffles):
        shuffled = signal.copy()
        rng.shuffle(shuffled)
        null_r1s.append(np.corrcoef(shuffled[:-1], shuffled[1:])[0, 1])
    
    p = np.mean(np.abs(np.array(null_r1s)) >= np.abs(r1))
    passes = (p < 0.05) and (abs(r1) > 0.2)
    
    return {"r1": float(r1), "p": float(p), "pass": bool(passes), "n_windows": len(signal)}


def compute_topo_sep(embeddings, n_pca_dims=10, n_tau_steps=50, seed=42):
    """Topological separation between structured and shuffled data."""
    from sklearn.decomposition import PCA
    
    rng = np.random.RandomState(seed)
    n_components = min(n_pca_dims, embeddings.shape[1], embeddings.shape[0] - 1)
    
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(embeddings)
    
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
    """Euler characteristic curve."""
    from scipy.spatial.distance import pdist
    
    if len(data) > 500:
        idx = np.linspace(0, len(data)-1, 500, dtype=int)
        data = data[idx]
    
    dists = pdist(data)
    tau_min, tau_max = np.percentile(dists, [5, 95])
    taus = np.linspace(tau_min, tau_max, n_steps)
    
    chi = []
    for tau in taus:
        adj = dists <= tau
        n = data.shape[0]
        parent = list(range(n))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[idx]:
                    union(i, j)
                idx += 1
        
        c0 = len(set(find(i) for i in range(n)))
        n_edges = np.sum(adj)
        c1 = max(0, n_edges - (n - c0))
        chi.append(c0 - c1)
    
    return np.array(chi)


def compute_alignment(embeddings, lattice_step, tau=0.03):
    """Lattice alignment after PCA projection."""
    from sklearn.decomposition import PCA
    
    n_components = min(3, embeddings.shape[1], embeddings.shape[0] - 1)
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(embeddings)
    
    residuals = np.mod(projected, lattice_step)
    residuals = np.minimum(residuals, lattice_step - residuals)
    min_residuals = np.min(residuals, axis=1)
    
    return float(np.mean(min_residuals < tau * lattice_step))


def compute_windowed_embeddings(token_embeddings, window_size):
    """Mean-pool token embeddings over sliding windows."""
    n = len(token_embeddings)
    stride = window_size // 2
    windows = []
    for i in range(0, n - window_size + 1, stride):
        windows.append(np.mean(token_embeddings[i:i+window_size], axis=0))
    return np.array(windows)


def run_experiment():
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Checkpoints: {CONFIG['checkpoints']}")
    print(f"Layers: {CONFIG['layers']}")
    print(f"Seq length: {CONFIG['seq_length']}")
    print()
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    np.random.seed(CONFIG["seed"])
    text = get_long_text()
    
    results = {
        "experiment": "pythia_microclimate",
        "hypothesis": "Autoregressive models create LOCAL structure (micro-climates) "
                      "detectable at token-level granularity but destroyed by sentence-level pooling.",
        "predictions": {
            "trained_vs_random": "Trained model shows H1 PASS and positive topo_sep at token level; "
                                "random model shows H1 FAIL and near-zero topo_sep",
            "layer_gradient": "Deeper layers show stronger emergence metrics (more accumulated constraints)",
            "window_effect": "Intermediate window sizes (8-16 tokens) may capture micro-climate structure "
                           "that single-token and large-window extraction miss",
        },
        "config": {k: v for k, v in CONFIG.items() if not isinstance(v, np.floating)},
        "checkpoints": {},
    }
    
    for step in CONFIG["checkpoints"]:
        print(f"\n{'='*70}")
        print(f"CHECKPOINT: step {step} ({'RANDOM INIT' if step == 0 else 'FULLY TRAINED'})")
        print(f"{'='*70}")
        
        hidden_states, actual_length = extract_token_hidden_states(
            CONFIG["model_name"], step, text, CONFIG["layers"], 
            CONFIG["seq_length"], device
        )
        
        step_results = {"step": step, "seq_length": actual_length, "analyses": {}}
        
        for layer_idx in CONFIG["layers"]:
            token_embs = hidden_states[layer_idx]
            print(f"\n  --- Layer {layer_idx} ---")
            
            layer_results = {}
            
            print(f"  [RAW TOKENS] n={len(token_embs)}")
            h1_raw = compute_h1(token_embs, window=CONFIG["h1_window"], 
                               n_shuffles=CONFIG["h1_n_shuffles"], seed=CONFIG["seed"])
            topo_raw = compute_topo_sep(token_embs, seed=CONFIG["seed"])
            align_raw = compute_alignment(token_embs, CONFIG["lattice_step"])
            
            layer_results["raw_tokens"] = {
                "h1": h1_raw, "topo_sep": topo_raw, "alignment": align_raw
            }
            print(f"    H1: r1={h1_raw['r1']:.3f}, p={h1_raw['p']:.3f}, pass={h1_raw['pass']}")
            print(f"    topo_sep: {topo_raw['topo_sep']:.4f}")
            print(f"    alignment: {align_raw:.4f}")
            
            layer_results["windowed"] = {}
            for ws in CONFIG["window_sizes"]:
                windowed = compute_windowed_embeddings(token_embs, ws)
                print(f"  [WINDOW={ws}] n={len(windowed)}")
                
                h1_w = compute_h1(windowed, window=CONFIG["h1_window"],
                                 n_shuffles=CONFIG["h1_n_shuffles"], seed=CONFIG["seed"])
                topo_w = compute_topo_sep(windowed, seed=CONFIG["seed"])
                align_w = compute_alignment(windowed, CONFIG["lattice_step"])
                
                layer_results["windowed"][str(ws)] = {
                    "h1": h1_w, "topo_sep": topo_w, "alignment": align_w
                }
                print(f"    H1: r1={h1_w['r1']:.3f}, p={h1_w['p']:.3f}, pass={h1_w['pass']}")
                print(f"    topo_sep: {topo_w['topo_sep']:.4f}")
                print(f"    alignment: {align_w:.4f}")
            
            step_results["analyses"][str(layer_idx)] = layer_results
        
        results["checkpoints"][str(step)] = step_results
        
        with open(os.path.join(CONFIG["output_dir"], "pythia_microclimate_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("SUMMARY: PREDICTION CHECKS")
    print(f"{'='*70}")
    
    for layer_idx in CONFIG["layers"]:
        print(f"\n  Layer {layer_idx}:")
        
        for strategy in ["raw_tokens"] + [f"windowed_{ws}" for ws in CONFIG["window_sizes"]]:
            if strategy == "raw_tokens":
                key = "raw_tokens"
                label = "Raw tokens"
            else:
                ws = strategy.split("_")[1]
                key = f"windowed"
                label = f"Window={ws}"
            
            for step in CONFIG["checkpoints"]:
                cp = results["checkpoints"][str(step)]
                analysis = cp["analyses"][str(layer_idx)]
                
                if strategy == "raw_tokens":
                    data = analysis["raw_tokens"]
                else:
                    ws = strategy.split("_")[1]
                    data = analysis["windowed"][ws]
                
                step_label = "RANDOM" if step == 0 else "TRAINED"
                h1_status = "PASS" if data["h1"]["pass"] else "FAIL"
                print(f"    {label:>15} | {step_label:>8} | H1={h1_status} (r1={data['h1']['r1']:+.3f}, p={data['h1']['p']:.3f}) | topo={data['topo_sep']['topo_sep']:+.4f} | align={data['alignment']:.4f}")
    
    print(f"\n  CRITICAL TEST: Does training create token-level temporal coherence?")
    for layer_idx in CONFIG["layers"]:
        random_h1 = results["checkpoints"]["0"]["analyses"][str(layer_idx)]["raw_tokens"]["h1"]["pass"]
        trained_h1 = results["checkpoints"]["143000"]["analyses"][str(layer_idx)]["raw_tokens"]["h1"]["pass"]
        
        if trained_h1 and not random_h1:
            print(f"    Layer {layer_idx}: PREDICTION CONFIRMED (random=FAIL, trained=PASS)")
        elif trained_h1 and random_h1:
            print(f"    Layer {layer_idx}: PARTIAL (both PASS - structure present even at init)")
        elif not trained_h1 and not random_h1:
            print(f"    Layer {layer_idx}: NULL (neither passes)")
        else:
            print(f"    Layer {layer_idx}: UNEXPECTED (random=PASS, trained=FAIL)")
    
    print(f"\nResults saved to {CONFIG['output_dir']}/pythia_microclimate_results.json")
    print("Done.")


if __name__ == "__main__":
    run_experiment()
