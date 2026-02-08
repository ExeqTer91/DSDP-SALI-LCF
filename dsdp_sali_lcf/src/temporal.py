"""Temporal analysis: S(t) score, wave persistence, FSI."""
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def compute_temporal_scores(y: np.ndarray, t: np.ndarray, labels: np.ndarray,
                           n_magistrales: int = 6, tau: float = 0.03,
                           n_bins: int = 50) -> Dict[str, Any]:
    """Compute S(t) - global lattice score per temporal bin."""
    from .lattice import lattice_residuals, SQRT_PHI
    
    step_a = np.log(SQRT_PHI)
    
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    y_sorted = y[sort_idx]
    
    bin_edges = np.linspace(t_sorted[0], t_sorted[-1] + 1e-12, n_bins + 1)
    scores = []
    bin_centers = []
    
    for i in range(n_bins):
        mask = (t_sorted >= bin_edges[i]) & (t_sorted < bin_edges[i + 1])
        y_bin = y_sorted[mask]
        if len(y_bin) < 5:
            scores.append(0.0)
        else:
            res, _ = lattice_residuals(y_bin, step_a, tau)
            scores.append(float(np.mean(res < tau)))
        bin_centers.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
    
    return {
        "bin_centers": bin_centers,
        "scores": scores,
        "n_bins": n_bins,
    }


def wave_persistence(scores: List[float], quantile_threshold: float = 0.75) -> Dict[str, Any]:
    """Compute wave persistence from S(t) scores.
    
    A 'wave' is a contiguous run of scores above the quantile threshold.
    """
    if not scores or len(scores) < 3:
        return {"n_waves": 0, "longest_wave": 0, "mean_wave": 0.0, "wave_lengths": []}
    
    threshold = float(np.quantile(scores, quantile_threshold))
    above = np.array(scores) >= threshold
    
    wave_lengths = []
    current_len = 0
    for a in above:
        if a:
            current_len += 1
        else:
            if current_len > 0:
                wave_lengths.append(current_len)
            current_len = 0
    if current_len > 0:
        wave_lengths.append(current_len)
    
    return {
        "n_waves": len(wave_lengths),
        "longest_wave": max(wave_lengths) if wave_lengths else 0,
        "mean_wave": float(np.mean(wave_lengths)) if wave_lengths else 0.0,
        "wave_lengths": wave_lengths,
        "threshold": threshold,
    }


def field_strength_index(signal_rate: float, surprise_z: float,
                         clustering_entropy: float,
                         weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Compute Field Strength Index (FSI) as weighted combination."""
    if weights is None:
        weights = {"signal_rate": 0.4, "surprise_z": 0.3, "clustering_entropy": 0.3}
    
    entropy_component = max(0.0, 1.0 - clustering_entropy / np.log(6 + 1e-12))
    
    surprise_norm = min(1.0, max(0.0, surprise_z / 5.0))
    signal_norm = min(1.0, max(0.0, signal_rate))
    
    fsi = (weights["signal_rate"] * signal_norm +
           weights["surprise_z"] * surprise_norm +
           weights["clustering_entropy"] * entropy_component)
    
    return {
        "fsi": float(fsi),
        "signal_rate_component": float(signal_norm * weights["signal_rate"]),
        "surprise_component": float(surprise_norm * weights["surprise_z"]),
        "entropy_component": float(entropy_component * weights["clustering_entropy"]),
        "raw_signal_rate": signal_rate,
        "raw_surprise_z": surprise_z,
        "raw_clustering_entropy": clustering_entropy,
    }


def compute_clustering_entropy(labels: np.ndarray, n_magistrales: int = 6) -> float:
    """Compute entropy of magistrale label distribution."""
    counts = np.bincount(labels, minlength=n_magistrales).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))
