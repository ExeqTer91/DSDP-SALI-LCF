"""Perturbation curve analysis with tipping point detection."""
import numpy as np
from typing import Dict, Any, List, Callable
import logging

logger = logging.getLogger(__name__)


def perturb_embeddings(E: np.ndarray, alpha: float, seed: int = 42) -> np.ndarray:
    """Perturb embeddings by mixing with noise at level alpha.
    
    alpha=0: no perturbation (original)
    alpha=1: full noise (destroys structure)
    """
    rng = np.random.RandomState(seed)
    n, d = E.shape
    
    noise = rng.randn(n, d)
    noise *= np.std(E)
    
    E_perturbed = (1 - alpha) * E + alpha * noise
    return E_perturbed


def perturbation_curve(E: np.ndarray, score_fn: Callable,
                       alphas: List[float] = None,
                       seed: int = 42) -> Dict[str, Any]:
    """Compute score vs perturbation level alpha."""
    if alphas is None:
        alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    scores = []
    for alpha in alphas:
        try:
            if alpha == 0.0:
                E_p = E.copy()
            else:
                E_p = perturb_embeddings(E, alpha, seed)
            sc = score_fn(E_p)
            scores.append(float(sc))
        except Exception as e:
            logger.warning(f"Perturbation at alpha={alpha} failed: {e}")
            scores.append(0.0)
    
    tipping = find_tipping_point(alphas, scores)
    
    return {
        "alphas": alphas,
        "scores": scores,
        "tipping_alpha": tipping["tipping_alpha"],
        "bic_delta": tipping["bic_delta"],
    }


def find_tipping_point(alphas: List[float], scores: List[float]) -> Dict[str, Any]:
    """Find tipping point via piecewise-linear BIC change."""
    if len(alphas) < 4:
        return {"tipping_alpha": None, "bic_delta": 0.0}
    
    x = np.array(alphas)
    y = np.array(scores)
    n = len(x)
    
    best_bic_delta = 0.0
    best_split = 2
    
    def fit_linear(xx, yy):
        if len(xx) < 2:
            return float("inf")
        A = np.column_stack([xx, np.ones(len(xx))])
        try:
            _, resid, _, _ = np.linalg.lstsq(A, yy, rcond=None)
            return float(resid[0]) if len(resid) > 0 else float(np.sum((yy - A @ np.linalg.lstsq(A, yy, rcond=None)[0]) ** 2))
        except Exception:
            return float(np.sum((yy - yy.mean()) ** 2))
    
    full_resid = fit_linear(x, y)
    
    for split in range(2, n - 1):
        r1 = fit_linear(x[:split], y[:split])
        r2 = fit_linear(x[split:], y[split:])
        combined = r1 + r2
        
        bic_single = n * np.log(full_resid / n + 1e-15) + 2 * np.log(n)
        bic_two = n * np.log(combined / n + 1e-15) + 4 * np.log(n)
        delta = bic_single - bic_two
        
        if delta > best_bic_delta:
            best_bic_delta = delta
            best_split = split
    
    return {
        "tipping_alpha": float(x[best_split]) if best_bic_delta > 0 else None,
        "bic_delta": float(best_bic_delta),
    }
