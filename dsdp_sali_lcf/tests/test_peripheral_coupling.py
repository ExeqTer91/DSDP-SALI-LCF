"""H2: Peripheral Coupling Test.

Tests whether peripheral (high-radius) perturbation affects coupling more than
core (low-radius) perturbation, while core identity is preserved.

If confirmed: the periphery (mirror telomeres) is the coupling zone, not the nucleus.

Method:
  - Base run: full pipeline on original embeddings
  - Peripheral run: perturb only points with radius > median (outer shell)
  - Core run: perturb only points with radius < median (inner core)
  Compare identity preservation and alignment change.
  Uses Hungarian matching to align permuted cluster labels before comparison.
"""
import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def split_by_radius(E: np.ndarray, hub: np.ndarray,
                     quantile: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    r = np.linalg.norm(E - hub, axis=1)
    threshold = np.quantile(r, quantile)
    peripheral_mask = r >= threshold
    core_mask = r < threshold
    return peripheral_mask, core_mask


def perturb_subset(E: np.ndarray, mask: np.ndarray,
                    alpha: float = 0.3, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    E_out = E.copy()
    n_perturb = int(mask.sum())
    if n_perturb == 0:
        return E_out
    noise = rng.randn(n_perturb, E.shape[1]) * alpha
    stds = np.std(E[mask], axis=0, keepdims=True)
    E_out[mask] = E[mask] + noise * stds
    return E_out


def run_variant(E: np.ndarray, pipeline_fn, label: str) -> Dict[str, Any]:
    result = pipeline_fn(E)
    return {
        "label": label,
        "alignment_score": result["alignment"],
        "identity_vector": result["centers"],
        "wave_persistence": result["longest_wave"],
        "fsi": result["fsi"],
    }


def test_peripheral_vs_core(base: Dict[str, Any],
                              peripheral: Dict[str, Any],
                              core: Dict[str, Any]) -> Dict[str, Any]:
    from dsdp_sali_lcf.metrics.identity import compute_identity_similarity

    core_sim = compute_identity_similarity(base["identity_vector"], core["identity_vector"])
    peri_sim = compute_identity_similarity(base["identity_vector"], peripheral["identity_vector"])

    alignment_change_peri = abs(peripheral["alignment_score"] - base["alignment_score"])
    alignment_change_core = abs(core["alignment_score"] - base["alignment_score"])

    wave_change_peri = abs(peripheral["wave_persistence"] - base["wave_persistence"])
    wave_change_core = abs(core["wave_persistence"] - base["wave_persistence"])

    fsi_change_peri = abs(peripheral["fsi"] - base["fsi"])
    fsi_change_core = abs(core["fsi"] - base["fsi"])

    total_impact_peri = alignment_change_peri + 0.5 * fsi_change_peri
    total_impact_core = alignment_change_core + 0.5 * fsi_change_core

    peri_more_impact = total_impact_peri > total_impact_core
    core_preserved = core_sim > 0.85
    identity_delta_meaningful = (core_sim - peri_sim) > 0.1
    effect_size_sufficient = alignment_change_peri > 0.02

    hypothesis_pass = (peri_more_impact and core_preserved
                       and identity_delta_meaningful
                       and effect_size_sufficient)

    return {
        "identity_core_similarity": float(core_sim),
        "identity_peripheral_similarity": float(peri_sim),
        "alignment_change_peripheral": float(alignment_change_peri),
        "alignment_change_core": float(alignment_change_core),
        "wave_change_peripheral": float(wave_change_peri),
        "wave_change_core": float(wave_change_core),
        "fsi_change_peripheral": float(fsi_change_peri),
        "fsi_change_core": float(fsi_change_core),
        "total_impact_peripheral": float(total_impact_peri),
        "total_impact_core": float(total_impact_core),
        "peripheral_has_more_impact": peri_more_impact,
        "core_identity_preserved": core_preserved,
        "identity_delta_meaningful": identity_delta_meaningful,
        "effect_size_sufficient": effect_size_sufficient,
        "pass": hypothesis_pass,
        "interpretation": (
            "CONFIRMED: Peripheral zone is coupling mechanism, core identity preserved"
            if hypothesis_pass else
            "NOT CONFIRMED: Peripheral coupling hypothesis not supported"
        ),
    }
