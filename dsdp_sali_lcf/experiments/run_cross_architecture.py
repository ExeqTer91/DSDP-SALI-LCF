"""Cross-Architecture Embedding Validation.

Tests whether DSDP pipeline findings generalize across different neural
architectures by comparing two sentence-transformer models:

  1. all-MiniLM-L6-v2 (384d, 6-layer MiniLM distilled from BERT)
  2. BAAI/bge-small-en-v1.5 (384d, 6-layer BGE trained with RetroMAE)

Same text corpus, same pipeline parameters, different architectures.
If spatial structure is positive on both models while controls remain
near-zero, this constitutes cross-architecture validation.

Design:
  Fixed tau=0.03, seed=42, n=5000 sentences, topically ordered.
  For each model: REAL (ordered embeddings) vs RANDOM (iid Gaussian 384d).
"""
import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dsdp_sali_lcf.experiments.run_real_embeddings import (
    ENGLISH_PARAGRAPHS,
    generate_expanded_texts_topical,
    tokenize_batch,
    run_dsdp_on_embeddings,
    run_topology_analysis,
)

import logging
logging.basicConfig(level=logging.WARNING)

SEED = 42
TAU = 0.03
N_TARGET = 5000

MODELS = {
    "MiniLM-L6-v2": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "onnx_file": "onnx/model.onnx",
        "tokenizer_file": "tokenizer.json",
        "dim": 384,
        "arch": "6-layer MiniLM (BERT distillation)",
    },
    "BGE-small-v1.5": {
        "repo_id": "BAAI/bge-small-en-v1.5",
        "onnx_file": "onnx/model.onnx",
        "tokenizer_file": "tokenizer.json",
        "dim": 384,
        "arch": "6-layer BGE (RetroMAE pre-training)",
    },
}


def extract_embeddings(texts: List[str], model_cfg: Dict, batch_size: int = 32) -> np.ndarray:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    repo = model_cfg["repo_id"]
    print(f"  Loading {repo}...", flush=True)
    model_path = hf_hub_download(
        repo_id=repo,
        filename=model_cfg["onnx_file"],
        cache_dir="/tmp/hf_cache"
    )
    tok_path = hf_hub_download(
        repo_id=repo,
        filename=model_cfg["tokenizer_file"],
        cache_dir="/tmp/hf_cache"
    )
    session = ort.InferenceSession(model_path)
    tokenizer = Tokenizer.from_file(tok_path)

    input_names = [inp.name for inp in session.get_inputs()]

    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(n_batches):
        batch_texts = texts[i * batch_size:(i + 1) * batch_size]
        inputs = tokenize_batch(tokenizer, batch_texts)
        feed = {k: v for k, v in inputs.items() if k in input_names}
        outputs = session.run(None, feed)
        hidden_states = outputs[0]
        mask = inputs["attention_mask"]
        mask_expanded = mask[:, :, np.newaxis].astype(np.float32)
        sum_hidden = (hidden_states * mask_expanded).sum(axis=1)
        count = mask_expanded.sum(axis=1)
        mean_pooled = sum_hidden / np.maximum(count, 1e-9)
        all_embeddings.append(mean_pooled)
        if (i + 1) % 10 == 0 or i == n_batches - 1:
            print(f"    Batch {i+1}/{n_batches}")

    return np.vstack(all_embeddings)


def main():
    print("=" * 78)
    print("CROSS-ARCHITECTURE EMBEDDING VALIDATION")
    print("=" * 78)
    t0 = time.time()

    print(f"\n[1] Generating text corpus (n={N_TARGET}, topically ordered)...")
    texts = generate_expanded_texts_topical(ENGLISH_PARAGRAPHS, N_TARGET, seed=SEED)
    print(f"  {len(texts)} sentences prepared")

    rng = np.random.RandomState(SEED)
    random_384 = rng.randn(N_TARGET, 384).astype(np.float32)

    model_results = {}
    for model_name, model_cfg in MODELS.items():
        print(f"\n[MODEL] {model_name} ({model_cfg['arch']})")
        print("-" * 60)

        emb = extract_embeddings(texts, model_cfg, batch_size=32)
        print(f"  Embeddings: {emb.shape}")

        print(f"  Running DSDP on {model_name}_REAL...")
        real_result = run_dsdp_on_embeddings(emb, f"{model_name}_REAL", seed=SEED)

        print(f"  Running DSDP on RANDOM_384...")
        rand_result = run_dsdp_on_embeddings(random_384, "RANDOM_384", seed=SEED)

        print(f"  Running topology...")
        real_topo = run_topology_analysis(emb, f"{model_name}_REAL", seed=SEED)
        rand_topo = run_topology_analysis(random_384, "RANDOM_384", seed=SEED)

        model_results[model_name] = {
            "config": model_cfg,
            "real": real_result,
            "random": rand_result,
            "topo_real": real_topo,
            "topo_random": rand_topo,
            "alignment_ratio": real_result["alignment"] / max(rand_result["alignment"], 1e-12),
            "topo_separation": real_topo["topo_score"] - rand_topo["topo_score"],
        }

        print(f"  Result: align_real={real_result['alignment']:.4f}, "
              f"align_rand={rand_result['alignment']:.4f}, "
              f"ratio={model_results[model_name]['alignment_ratio']:.1f}x, "
              f"topo_sep={model_results[model_name]['topo_separation']:.1f}")

    elapsed = time.time() - t0

    print("\n" + "=" * 78)
    print("CROSS-ARCHITECTURE COMPARISON")
    print("=" * 78)

    print(f"\n{'Model':<20} {'Arch':<35} {'Real Align':>10} {'Rand Align':>10} "
          f"{'Ratio':>6} {'Topo Sep':>8}")
    print("-" * 95)
    for model_name, res in model_results.items():
        print(f"{model_name:<20} {res['config']['arch']:<35} "
              f"{res['real']['alignment']:>10.4f} {res['random']['alignment']:>10.4f} "
              f"{res['alignment_ratio']:>6.1f}x {res['topo_separation']:>8.1f}")

    all_positive = all(r["alignment_ratio"] > 1.0 for r in model_results.values())
    all_strong = all(r["alignment_ratio"] > 3.0 for r in model_results.values())

    print(f"\n  All models alignment > random: {all_positive}")
    print(f"  All models alignment > 3x random: {all_strong}")

    if all_strong:
        print("\n  >> CROSS-ARCHITECTURE VALIDATION: CONFIRMED")
        print("  >> Spatial structure is model-independent.")
        print("  >> Lattice geometry detected in both MiniLM and BGE architectures.")
    elif all_positive:
        print("\n  >> CROSS-ARCHITECTURE VALIDATION: PARTIAL")
        print("  >> Both models show positive alignment, but separation varies.")
    else:
        print("\n  >> CROSS-ARCHITECTURE VALIDATION: INCONCLUSIVE")

    print(f"\n  Total time: {elapsed:.1f}s")

    output_dir = Path("dsdp_sali_lcf/outputs/cross_architecture")
    output_dir.mkdir(parents=True, exist_ok=True)

    full_output = {
        "metadata": {
            "n_sentences": N_TARGET,
            "seed": SEED,
            "tau": TAU,
            "elapsed_seconds": elapsed,
        },
        "models": {},
    }
    for model_name, res in model_results.items():
        full_output["models"][model_name] = {
            "architecture": res["config"]["arch"],
            "repo_id": res["config"]["repo_id"],
            "dim": res["config"]["dim"],
            "real_alignment": res["real"]["alignment"],
            "random_alignment": res["random"]["alignment"],
            "alignment_ratio": res["alignment_ratio"],
            "real_fsi": res["real"]["fsi"],
            "real_waves": res["real"]["longest_wave"],
            "real_h1_p": res["real"]["h1_p_value"],
            "real_h1_pass": res["real"]["h1_pass"],
            "topo_separation": res["topo_separation"],
        }

    full_output["verdict"] = {
        "all_positive": all_positive,
        "all_strong": all_strong,
        "status": "CONFIRMED" if all_strong else ("PARTIAL" if all_positive else "INCONCLUSIVE"),
    }

    out_path = output_dir / "cross_architecture_results.json"
    with open(out_path, "w") as f:
        json.dump(full_output, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
