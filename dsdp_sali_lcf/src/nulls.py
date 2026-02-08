"""Null model generators and statistical testing."""
import numpy as np
from typing import Dict, Any, List, Callable
import logging

logger = logging.getLogger(__name__)


def uniform_null(y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Generate uniform null matching range of y."""
    rng = np.random.RandomState(seed)
    return rng.uniform(y.min(), y.max(), size=len(y))


def gaussian_cov_null(E: np.ndarray, seed: int = 0) -> np.ndarray:
    """Generate covariance-preserving Gaussian null embedding."""
    rng = np.random.RandomState(seed)
    mean = E.mean(axis=0)
    n, d = E.shape
    
    if d > 200:
        std = E.std(axis=0)
        return rng.randn(n, d) * std + mean
    else:
        try:
            cov = np.cov(E, rowvar=False)
            cov += np.eye(d) * 1e-8
            return rng.multivariate_normal(mean, cov, size=n)
        except Exception:
            std = E.std(axis=0)
            return rng.randn(n, d) * std + mean


def gaussian_cov_null_y(y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Fast Gaussian null that matches mean and std of y directly."""
    rng = np.random.RandomState(seed)
    return rng.normal(loc=y.mean(), scale=y.std(), size=len(y))


def spacing_null(y: np.ndarray, labels: np.ndarray, n_magistrales: int = 6,
                 seed: int = 0) -> np.ndarray:
    """Spacing-preserving null: permute delta-y within each magistrale."""
    rng = np.random.RandomState(seed)
    y_null = np.zeros_like(y)
    
    for m in range(n_magistrales):
        mask = labels == m
        y_m = y[mask]
        if len(y_m) < 3:
            y_null[mask] = y_m
            continue
        
        sorted_y = np.sort(y_m)
        deltas = np.diff(sorted_y)
        rng.shuffle(deltas)
        
        reconstructed = np.zeros(len(y_m))
        reconstructed[0] = sorted_y[0]
        for i in range(len(deltas)):
            reconstructed[i + 1] = reconstructed[i] + deltas[i]
        
        rng.shuffle(reconstructed)
        y_null[mask] = reconstructed
    
    return y_null


def run_null_suite(y: np.ndarray, labels: np.ndarray, E: np.ndarray,
                   score_fn: Callable, n_surrogates: int = 50,
                   n_magistrales: int = 6, seed: int = 42) -> Dict[str, Any]:
    """Run all three null models and compute p-values.
    
    score_fn: callable that takes (y, labels) and returns a scalar score.
    Higher score = more structure detected.
    """
    observed = score_fn(y, labels)
    
    null_types = ["uniform", "gaussian_cov", "spacing"]
    results = {}
    
    for null_type in null_types:
        null_scores = []
        for b in range(n_surrogates):
            s = seed + b * 1000
            try:
                if null_type == "uniform":
                    y_null = uniform_null(y, seed=s)
                elif null_type == "gaussian_cov":
                    E_null = gaussian_cov_null(E, seed=s)
                    from .geometry import compute_radii
                    hub = E.mean(axis=0)
                    r_null = compute_radii(E_null, hub)
                    from .lattice import compute_log_radii
                    y_null = compute_log_radii(r_null)
                else:
                    y_null = spacing_null(y, labels, n_magistrales, seed=s)
                
                ns = score_fn(y_null, labels)
                null_scores.append(ns)
            except Exception as e:
                logger.warning(f"Null {null_type} surrogate {b} failed: {e}")
                continue
        
        if null_scores:
            null_arr = np.array(null_scores)
            p_value = float(np.mean(null_arr >= observed))
            z_score = float((observed - null_arr.mean()) / (null_arr.std() + 1e-12))
        else:
            p_value = 1.0
            z_score = 0.0
        
        results[null_type] = {
            "p_value": p_value,
            "z_score": z_score,
            "observed": float(observed),
            "null_mean": float(np.mean(null_scores)) if null_scores else 0.0,
            "null_std": float(np.std(null_scores)) if null_scores else 0.0,
            "n_valid_surrogates": len(null_scores),
        }
    
    conservative_p = max(r["p_value"] for r in results.values())
    min_z = min(r["z_score"] for r in results.values())
    
    results["conservative_p"] = conservative_p
    results["min_z"] = min_z
    results["observed"] = float(observed)
    
    logger.info(f"Null suite: observed={observed:.4f}, conservative_p={conservative_p:.4f}")
    
    return results


def holm_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """Holm-Bonferroni correction for multiple testing."""
    m = len(p_values)
    if m == 0:
        return {"adjusted": [], "any_significant": False, "alpha": alpha}
    
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * m
    rejected = [False] * m
    
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (m - rank)
        if p <= threshold:
            rejected[orig_idx] = True
            adjusted[orig_idx] = p * (m - rank)
        else:
            adjusted[orig_idx] = min(p * (m - rank), 1.0)
            for j in range(rank, m):
                orig_j = indexed[j][0]
                adjusted[orig_j] = min(indexed[j][1] * (m - j), 1.0)
                rejected[orig_j] = False
            break
    
    return {
        "adjusted": adjusted,
        "rejected": rejected,
        "any_significant": any(rejected),
        "alpha": alpha,
    }
