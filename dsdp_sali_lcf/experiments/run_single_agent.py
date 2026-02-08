"""Run a single pipeline pass and extract metrics for hypothesis testing."""
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dsdp_sali_lcf.src.geometry import build_geometry, compute_radii
from dsdp_sali_lcf.src.lattice import compute_log_radii, score_per_magistral, SQRT_PHI
from dsdp_sali_lcf.src.temporal import (
    compute_temporal_scores, wave_persistence,
    field_strength_index, compute_clustering_entropy
)


def run_pipeline_extract(E: np.ndarray, t: np.ndarray,
                          n_mag: int = 6, tau: float = 0.03,
                          eps: float = 1e-12, seed: int = 42,
                          wave_quantile: float = 0.75) -> Dict[str, Any]:
    geo = build_geometry(E, n_mag, seed)
    r = compute_radii(E, geo["hub"])
    y = compute_log_radii(r, eps)

    lattice = score_per_magistral(y, geo["labels"], n_mag, tau)

    temporal = compute_temporal_scores(y, t, geo["labels"], n_mag, tau)
    waves = wave_persistence(temporal["scores"], wave_quantile)
    entropy = compute_clustering_entropy(geo["labels"], n_mag)

    fsi_data = field_strength_index(lattice["global_alignment"], 0.0, entropy)

    return {
        "alignment": lattice["global_alignment"],
        "centers": geo["centers"],
        "labels": geo["labels"],
        "hub": geo["hub"],
        "y": y,
        "r": r,
        "temporal_scores": temporal["scores"],
        "temporal_bin_centers": temporal["bin_centers"],
        "longest_wave": waves["longest_wave"],
        "n_waves": waves["n_waves"],
        "mean_wave": waves["mean_wave"],
        "wave_lengths": waves["wave_lengths"],
        "fsi": fsi_data["fsi"],
        "entropy": entropy,
        "E": E,
        "t": t,
    }
