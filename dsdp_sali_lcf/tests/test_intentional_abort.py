"""H5: Intentional Spin Control (Abort) Test.

Tests whether injecting an abort signal (destructive noise into the coupler
switching zone) kills coupling WITHOUT destroying the underlying structure/identity.

If confirmed: intentional decoupling is possible — spin control, not destruction.

Method:
  - Normal run: full pipeline
  - Abort run: inject targeted noise into points where dual-grid switching occurs
    (points assigned to different grids in the coupler mechanism)
  Compare: coupling drops but identity preserved.
  Uses Hungarian matching for label-aligned identity comparison.
"""
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def inject_abort_signal(E: np.ndarray, y: np.ndarray,
                         coupler_threshold: float = None,
                         abort_strength: float = 0.5,
                         seed: int = 42,
                         hub: np.ndarray = None) -> np.ndarray:
    from dsdp_sali_lcf.src.lattice import dual_grid_score, lattice_residuals, SQRT_PHI
    from sklearn.decomposition import PCA

    if coupler_threshold is None:
        coupler_threshold = np.log(2.25)

    score_data = dual_grid_score(y, tau=0.03, coupler_threshold=coupler_threshold)
    grid_assign = score_data["grid_assignment"]

    switch_points = np.zeros(len(y), dtype=bool)
    if len(y) > 1:
        switches = np.abs(np.diff(grid_assign)) > 0
        switch_points[:-1] |= switches
        switch_points[1:] |= switches

    transition_width = 0.15 * abs(coupler_threshold)
    near_boundary = np.abs(np.abs(y) - coupler_threshold) < transition_width
    target_mask = switch_points | near_boundary

    n_affected = int(target_mask.sum())
    use_anti_axial = False
    if n_affected < len(E) * 0.05:
        step_a = np.log(SQRT_PHI)
        residuals, _ = lattice_residuals(y, step_a, tau=0.03)
        well_aligned = residuals < 0.04
        target_mask = well_aligned
        n_affected = int(target_mask.sum())
        if n_affected < len(E) * 0.05:
            target_mask = residuals < 0.08
            n_affected = int(target_mask.sum())
        use_anti_axial = True

    logger.info(f"Abort signal: affecting {n_affected}/{len(E)} points "
                f"({100*n_affected/len(E):.1f}%) "
                f"({'anti-axial' if use_anti_axial else 'coupler zone'})")

    rng = np.random.RandomState(seed)
    E_abort = E.copy()
    if n_affected > 0:
        if use_anti_axial and hub is not None:
            center = hub
            step = np.log(SQRT_PHI)

            diff = E[target_mask] - center
            r = np.linalg.norm(diff, axis=1)
            r_safe = np.maximum(r, 1e-12)
            y_target = np.log(r_safe)

            nearest_band = np.round(y_target / step) * step
            midpoint = nearest_band + step / 2.0
            y_new = y_target + abort_strength * (midpoint - y_target)

            scale = np.exp(y_new) / np.exp(y_target)
            E_abort[target_mask] = center + diff * scale[:, np.newaxis]

            pca = PCA(n_components=min(3, E.shape[1]), random_state=seed)
            pca.fit(E - center)
            for k in range(min(3, E.shape[1])):
                v_k = pca.components_[k]
                projs = (E_abort[target_mask] - center) @ v_k
                jitter = rng.randn(n_affected) * np.std(projs) * 0.02 * abort_strength
                E_abort[target_mask] += jitter[:, np.newaxis] * v_k[np.newaxis, :]
        else:
            noise = rng.randn(n_affected, E.shape[1])
            stds = np.std(E[target_mask], axis=0, keepdims=True)
            E_abort[target_mask] = E[target_mask] + noise * stds * abort_strength
    return E_abort


def test_abort_spin_control(normal: Dict[str, Any],
                             abort: Dict[str, Any]) -> Dict[str, Any]:
    from dsdp_sali_lcf.metrics.identity import compute_identity_similarity

    identity_sim = compute_identity_similarity(
        normal["identity_vector"], abort["identity_vector"]
    )

    wave_drop = normal["wave_persistence"] - abort["wave_persistence"]
    alignment_drop = normal["alignment_score"] - abort["alignment_score"]
    fsi_drop = normal["fsi"] - abort["fsi"]

    coupling_effect = alignment_drop + 0.3 * fsi_drop
    coupling_dropped = coupling_effect > 0.005 or wave_drop > 0
    identity_preserved = identity_sim > 0.80
    hypothesis_pass = coupling_dropped and identity_preserved

    return {
        "wave_drop": float(wave_drop),
        "alignment_drop": float(alignment_drop),
        "fsi_drop": float(fsi_drop),
        "coupling_effect": float(coupling_effect),
        "identity_preserved_similarity": float(identity_sim),
        "coupling_dropped": coupling_dropped,
        "identity_preserved": identity_preserved,
        "pass": hypothesis_pass,
        "interpretation": (
            "CONFIRMED: Abort decouples without destroying identity (spin control)"
            if hypothesis_pass else
            "PARTIAL: " + (
                "Coupling did not drop sufficiently"
                if not coupling_dropped else
                "Identity was not preserved after abort"
            )
        ),
    }
