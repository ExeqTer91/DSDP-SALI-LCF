"""Lattice scoring in log-space with dual-grid and coupler."""
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

PHI = 1.6180339887498949
SQRT_PHI = np.sqrt(PHI)


def compute_log_radii(r: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert radii to log-space."""
    return np.log(r + eps)


def lattice_residuals(y: np.ndarray, step: float, tau: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
    """Compute residuals from a log-space lattice grid.
    
    Returns (residuals, nearest_band_index).
    """
    if step < 1e-15:
        return np.full_like(y, np.inf), np.zeros(len(y), dtype=int)
    
    band_index = np.round(y / step)
    nearest = band_index * step
    residuals = np.abs(y - nearest)
    return residuals, band_index.astype(int)


def alignment_rate(residuals: np.ndarray, tau: float = 0.03) -> float:
    """Fraction of points within tolerance tau of a lattice band."""
    return float(np.mean(residuals < tau))


def dual_grid_score(y: np.ndarray, tau: float = 0.03, coupler_threshold: float = None,
                    switching_penalty: float = 0.1) -> Dict[str, Any]:
    """Score alignment to dual grid (sqrt(phi) and phi) with coupler switching."""
    step_a = np.log(SQRT_PHI)
    step_b = np.log(PHI)
    
    if coupler_threshold is None:
        coupler_threshold = np.log(2.25)
    
    res_a, bands_a = lattice_residuals(y, step_a, tau)
    res_b, bands_b = lattice_residuals(y, step_b, tau)
    
    if len(y) == 0:
        return {
            "alignment_rate_a": 0.0, "alignment_rate_b": 0.0,
            "combined_alignment": 0.0, "median_resid": 0.0,
            "switch_rate": 0.0, "grid_assignment": np.array([]),
            "residuals": np.array([]), "bands": np.array([]),
        }
    
    grid_assign = np.where(np.abs(y) < coupler_threshold, 0, 1)  # 0=A, 1=B
    
    residuals = np.where(grid_assign == 0, res_a, res_b)
    bands = np.where(grid_assign == 0, bands_a, bands_b)
    
    if len(y) > 1:
        switches = np.sum(np.diff(grid_assign) != 0)
        switch_rate = float(switches) / (len(y) - 1)
    else:
        switch_rate = 0.0
    
    combined = alignment_rate(residuals, tau) - switching_penalty * switch_rate
    
    return {
        "alignment_rate_a": alignment_rate(res_a, tau),
        "alignment_rate_b": alignment_rate(res_b, tau),
        "combined_alignment": float(combined),
        "median_resid": float(np.median(residuals)),
        "switch_rate": switch_rate,
        "grid_assignment": grid_assign,
        "residuals": residuals,
        "bands": bands,
    }


def score_per_magistral(y: np.ndarray, labels: np.ndarray, n_magistrales: int = 6,
                        tau: float = 0.03, coupler_threshold: float = None,
                        switching_penalty: float = 0.1) -> Dict[str, Any]:
    """Score lattice alignment per magistrale and globally."""
    results_per_mag = []
    all_residuals = []
    
    for m in range(n_magistrales):
        mask = labels == m
        y_m = y[mask]
        if len(y_m) < 5:
            results_per_mag.append({
                "magistrale": m, "n_points": len(y_m),
                "alignment_rate": 0.0, "median_resid": float("inf"),
                "switch_rate": 0.0, "status": "too_few_points",
            })
            continue
        
        score = dual_grid_score(y_m, tau, coupler_threshold, switching_penalty)
        results_per_mag.append({
            "magistrale": m, "n_points": int(mask.sum()),
            "alignment_rate": score["combined_alignment"],
            "median_resid": score["median_resid"],
            "switch_rate": score["switch_rate"],
            "status": "ok",
        })
        all_residuals.extend(score["residuals"].tolist())
    
    valid = [r for r in results_per_mag if r["status"] == "ok"]
    global_alignment = float(np.mean([r["alignment_rate"] for r in valid])) if valid else 0.0
    global_median_resid = float(np.median(all_residuals)) if all_residuals else float("inf")
    
    return {
        "per_magistral": results_per_mag,
        "global_alignment": global_alignment,
        "global_median_resid": global_median_resid,
    }


def score_with_seed(y: np.ndarray, labels: np.ndarray, seed_value: float,
                    n_magistrales: int = 6, tau: float = 0.03) -> float:
    """Score alignment using a specific seed value (for ablation)."""
    step = np.log(seed_value) if seed_value > 0 else 0.01
    total_aligned = 0
    total_points = 0
    
    for m in range(n_magistrales):
        mask = labels == m
        y_m = y[mask]
        if len(y_m) < 5:
            continue
        res, _ = lattice_residuals(y_m, step, tau)
        total_aligned += int(np.sum(res < tau))
        total_points += len(y_m)
    
    return total_aligned / max(total_points, 1)


def predict_next_band(y: np.ndarray, labels: np.ndarray, n_magistrales: int = 6,
                      tau: float = 0.03) -> Dict[str, Any]:
    """Predict next band center per magistrale (E6 prediction).
    
    Split data into known bands and predict the next one.
    """
    step_a = np.log(SQRT_PHI)
    predictions = []
    
    for m in range(n_magistrales):
        mask = labels == m
        y_m = y[mask]
        if len(y_m) < 10:
            predictions.append({
                "magistrale": m, "status": "too_few_points",
                "predicted_center": None, "actual_center": None, "error": None,
            })
            continue
        
        _, bands = lattice_residuals(y_m, step_a, tau)
        unique_bands = np.unique(bands)
        
        if len(unique_bands) < 3:
            predictions.append({
                "magistrale": m, "status": "too_few_bands",
                "predicted_center": None, "actual_center": None, "error": None,
            })
            continue
        
        sorted_bands = np.sort(unique_bands)
        train_bands = sorted_bands[:-1]
        test_band = sorted_bands[-1]
        
        if len(train_bands) >= 2:
            diffs = np.diff(train_bands.astype(float))
            median_diff = np.median(diffs)
            predicted_band = train_bands[-1] + median_diff
        else:
            predicted_band = train_bands[-1] + 1
        
        predicted_center = predicted_band * step_a
        actual_center = test_band * step_a
        error = abs(predicted_center - actual_center)
        
        predictions.append({
            "magistrale": m, "status": "ok",
            "predicted_center": float(predicted_center),
            "actual_center": float(actual_center),
            "error": float(error),
            "predicted_band_idx": float(predicted_band),
            "actual_band_idx": float(test_band),
        })
    
    valid = [p for p in predictions if p["status"] == "ok" and p["error"] is not None]
    median_error = float(np.median([p["error"] for p in valid])) if valid else float("inf")
    
    return {
        "predictions": predictions,
        "median_error": median_error,
        "n_valid": len(valid),
    }
