"""H1: Coherence Threshold Test (Residual Entropy variant).

Tests whether residual entropy (alignment disorder) drops BEFORE wave persistence rises.
If confirmed: internal consolidation precedes resonance (coherence threshold mechanism).

Note: This uses per-bin residual entropy as proxy for internal entropy.
      For session-level token probabilities, replace compute_per_bin_entropy
      with actual token_probs from session logs.

Method: lagged cross-correlation between per-bin entropy and per-bin wave signal.
Negative lag-1 correlation = entropy drop predicts next-step wave rise.
Statistical significance via permutation null.
"""
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def lagged_cross_correlation(x: np.ndarray, y_signal: np.ndarray,
                              max_lag: int = 5) -> Dict[str, float]:
    x = np.array(x, dtype=float)
    y_signal = np.array(y_signal, dtype=float)

    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    y_signal = (y_signal - np.mean(y_signal)) / (np.std(y_signal) + 1e-12)

    n = len(x)
    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = np.corrcoef(x[:n - lag], y_signal[lag:])[0, 1]
        else:
            corr = np.corrcoef(x[-lag:], y_signal[:n + lag])[0, 1]
        if np.isnan(corr):
            corr = 0.0
        correlations[lag] = float(corr)
    return correlations


def permutation_null_lag1(entropies: np.ndarray, wave_signals: np.ndarray,
                           n_perm: int = 500, seed: int = 42) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    observed = np.corrcoef(entropies[:-1], wave_signals[1:])[0, 1]
    if np.isnan(observed):
        observed = 0.0

    null_corrs = []
    for _ in range(n_perm):
        perm_ent = rng.permutation(entropies)
        c = np.corrcoef(perm_ent[:-1], wave_signals[1:])[0, 1]
        if np.isnan(c):
            c = 0.0
        null_corrs.append(c)

    null_corrs = np.array(null_corrs)
    p_value = float(np.mean(null_corrs <= observed))
    z_score = float((observed - np.mean(null_corrs)) / (np.std(null_corrs) + 1e-12))

    return {
        "observed": float(observed),
        "null_mean": float(np.mean(null_corrs)),
        "null_std": float(np.std(null_corrs)),
        "p_value": p_value,
        "z_score": z_score,
    }


def test_coherence_threshold(entropies: List[float],
                              wave_signals: List[float],
                              alignment_scores: List[float],
                              n_perm: int = 500,
                              seed: int = 42) -> Dict[str, Any]:
    entropies = np.array(entropies)
    wave_signals = np.array(wave_signals)
    alignment_scores = np.array(alignment_scores)

    n = min(len(entropies), len(wave_signals))
    if n < 6:
        return {
            "status": "insufficient_data",
            "n_bins": n,
            "pass": False,
        }

    entropies = entropies[:n]
    wave_signals = wave_signals[:n]
    alignment_scores = alignment_scores[:n]

    lag1_corr = np.corrcoef(entropies[:-1], wave_signals[1:])[0, 1]
    if np.isnan(lag1_corr):
        lag1_corr = 0.0

    lag0_corr = np.corrcoef(entropies, wave_signals)[0, 1]
    if np.isnan(lag0_corr):
        lag0_corr = 0.0

    cross_corr = lagged_cross_correlation(entropies, wave_signals, max_lag=5)

    optimal_lag = min(cross_corr, key=lambda k: cross_corr[k])
    optimal_corr = cross_corr[optimal_lag]

    lag1_ent_align = np.corrcoef(entropies[:-1], alignment_scores[1:])[0, 1]
    if np.isnan(lag1_ent_align):
        lag1_ent_align = 0.0

    perm_result = permutation_null_lag1(entropies, wave_signals, n_perm, seed)

    sig_pass = perm_result["p_value"] < 0.05 and lag1_corr < 0
    threshold_pass = lag1_corr < -0.2
    strong_pass = sig_pass and threshold_pass
    conservative_pass = sig_pass and threshold_pass

    return {
        "status": "ok",
        "metric_type": "residual_entropy (proxy for token entropy)",
        "n_bins": n,
        "entropy_wave_lag1_correlation": float(lag1_corr),
        "entropy_wave_lag0_correlation": float(lag0_corr),
        "entropy_alignment_lag1_correlation": float(lag1_ent_align),
        "cross_correlation_profile": cross_corr,
        "optimal_lag": int(optimal_lag),
        "optimal_correlation": float(optimal_corr),
        "permutation_test": perm_result,
        "pass": conservative_pass,
        "strong_pass": strong_pass,
        "sig_pass": sig_pass,
        "threshold_pass": threshold_pass,
        "interpretation": (
            "CONFIRMED: Residual entropy drop precedes wave rise (p<0.05, |r|>0.2)"
            if conservative_pass else
            "WEAK: Lag-1 correlation detected but not significant under permutation null"
            if threshold_pass and not sig_pass else
            "MARGINAL: Significant permutation test but weak correlation magnitude"
            if sig_pass and not threshold_pass else
            "NOT CONFIRMED: No significant entropy-wave lag structure"
        ),
    }
