"""Ablation tests: seed, coupler, anchor, temporal order."""
import numpy as np
from typing import Dict, Any, List
from .lattice import score_with_seed, score_per_magistral, dual_grid_score, compute_log_radii
from .geometry import build_geometry, compute_radii, approximate_farthest_pair
import logging

logger = logging.getLogger(__name__)


def seed_ablation(y: np.ndarray, labels: np.ndarray, n_magistrales: int = 6,
                  tau: float = 0.03, decoy_seeds: List[float] = None) -> Dict[str, Any]:
    """Compare sqrt(phi) alignment vs decoy seeds."""
    from .lattice import SQRT_PHI
    
    if decoy_seeds is None:
        decoy_seeds = [1.70, 2.10, 2.60]
    
    real_score = score_with_seed(y, labels, SQRT_PHI, n_magistrales, tau)
    
    decoy_results = []
    for ds in decoy_seeds:
        ds_score = score_with_seed(y, labels, ds, n_magistrales, tau)
        decoy_results.append({"seed": ds, "score": ds_score})
    
    max_decoy = max(d["score"] for d in decoy_results) if decoy_results else 0.0
    advantage = real_score - max_decoy
    
    result = {
        "real_seed": float(SQRT_PHI),
        "real_score": real_score,
        "decoy_results": decoy_results,
        "max_decoy_score": max_decoy,
        "advantage": advantage,
        "seed_is_special": advantage > 0,
    }
    
    logger.info(f"Seed ablation: real={real_score:.4f}, max_decoy={max_decoy:.4f}, advantage={advantage:.4f}")
    return result


def coupler_ablation(y: np.ndarray, labels: np.ndarray, n_magistrales: int = 6,
                     tau: float = 0.03) -> Dict[str, Any]:
    """Compare score with and without coupler."""
    score_with = score_per_magistral(y, labels, n_magistrales, tau,
                                     coupler_threshold=np.log(2.25))
    score_without = score_per_magistral(y, labels, n_magistrales, tau,
                                        coupler_threshold=1e15)
    
    diff = score_with["global_alignment"] - score_without["global_alignment"]
    
    return {
        "with_coupler": score_with["global_alignment"],
        "without_coupler": score_without["global_alignment"],
        "difference": diff,
        "coupler_matters": abs(diff) > 0.01,
    }


def anchor_ablation(E: np.ndarray, y_original: np.ndarray, labels_original: np.ndarray,
                    n_magistrales: int = 6, tau: float = 0.03,
                    n_random: int = 5, seed: int = 42) -> Dict[str, Any]:
    """Compare original anchors vs random anchor pairs."""
    original_score = score_per_magistral(y_original, labels_original, n_magistrales, tau)
    original_global = original_score["global_alignment"]
    
    rng = np.random.RandomState(seed)
    n = E.shape[0]
    random_scores = []
    
    for i in range(n_random):
        try:
            pair = tuple(rng.choice(n, 2, replace=False))
            geo = build_geometry(E, n_magistrales, seed=seed + i * 100, anchor_pair=pair)
            r = geo["r"]
            y_rand = compute_log_radii(r)
            sc = score_per_magistral(y_rand, geo["labels"], n_magistrales, tau)
            random_scores.append(sc["global_alignment"])
        except Exception as e:
            logger.warning(f"Anchor ablation trial {i} failed: {e}")
    
    mean_random = float(np.mean(random_scores)) if random_scores else 0.0
    stability = original_global - mean_random
    
    return {
        "original_score": original_global,
        "random_scores": random_scores,
        "mean_random": mean_random,
        "stability_advantage": stability,
        "anchor_stable": stability > 0,
    }


def temporal_ablation(y: np.ndarray, t: np.ndarray, labels: np.ndarray,
                      n_magistrales: int = 6, tau: float = 0.03,
                      seed: int = 42) -> Dict[str, Any]:
    """Shuffle temporal order and check if wave persistence changes."""
    from .temporal import compute_temporal_scores, wave_persistence
    
    original_temporal = compute_temporal_scores(y, t, labels, n_magistrales, tau)
    original_waves = wave_persistence(original_temporal["scores"])
    
    rng = np.random.RandomState(seed)
    t_shuffled = t.copy()
    rng.shuffle(t_shuffled)
    
    shuffled_temporal = compute_temporal_scores(y, t_shuffled, labels, n_magistrales, tau)
    shuffled_waves = wave_persistence(shuffled_temporal["scores"])
    
    return {
        "original_longest_wave": original_waves["longest_wave"],
        "shuffled_longest_wave": shuffled_waves["longest_wave"],
        "original_n_waves": original_waves["n_waves"],
        "shuffled_n_waves": shuffled_waves["n_waves"],
        "temporal_order_matters": original_waves["longest_wave"] > shuffled_waves["longest_wave"],
    }
