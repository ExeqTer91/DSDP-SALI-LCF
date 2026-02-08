"""Wave persistence metrics."""
import numpy as np
from typing import Dict, Any, List


def compute_wave_profile(scores: List[float],
                          quantile_threshold: float = 0.75) -> Dict[str, Any]:
    if not scores or len(scores) < 3:
        return {
            "n_waves": 0, "longest_wave": 0, "mean_wave": 0.0,
            "wave_lengths": [], "above_threshold": []
        }

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
        "above_threshold": above.tolist(),
        "threshold": threshold,
    }


def compute_per_bin_wave_signal(scores: List[float],
                                 quantile_threshold: float = 0.75) -> List[float]:
    if not scores or len(scores) < 3:
        return [0.0] * len(scores) if scores else []

    threshold = float(np.quantile(scores, quantile_threshold))
    return [float(s >= threshold) for s in scores]
