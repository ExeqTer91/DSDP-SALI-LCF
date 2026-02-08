"""Entropy metrics for coherence analysis."""
import numpy as np
from typing import List


def compute_token_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def compute_alignment_entropy(residuals: np.ndarray, tau: float = 0.03,
                               n_bins: int = 20) -> float:
    hist, _ = np.histogram(residuals, bins=n_bins, density=True)
    hist = hist + 1e-12
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log(hist)))


def compute_per_bin_entropy(y: np.ndarray, t: np.ndarray, labels: np.ndarray,
                             n_bins: int = 50, tau: float = 0.03) -> List[float]:
    from dsdp_sali_lcf.src.lattice import lattice_residuals, SQRT_PHI

    step_a = np.log(SQRT_PHI)
    sort_idx = np.argsort(t)
    y_sorted = y[sort_idx]
    t_sorted = t[sort_idx]
    labels_sorted = labels[sort_idx]

    bin_edges = np.linspace(t_sorted[0], t_sorted[-1] + 1e-12, n_bins + 1)
    entropies = []

    for i in range(n_bins):
        mask = (t_sorted >= bin_edges[i]) & (t_sorted < bin_edges[i + 1])
        y_bin = y_sorted[mask]
        lab_bin = labels_sorted[mask]

        if len(y_bin) < 5:
            entropies.append(float(np.log(n_bins)))
            continue

        res, _ = lattice_residuals(y_bin, step_a, tau)
        ent = compute_alignment_entropy(res, tau)
        entropies.append(ent)

    return entropies


def compute_magistrale_entropy(labels: np.ndarray, n_magistrales: int = 6) -> float:
    counts = np.bincount(labels, minlength=n_magistrales).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))
