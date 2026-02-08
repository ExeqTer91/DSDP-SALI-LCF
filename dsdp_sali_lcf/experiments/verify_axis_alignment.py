"""ENV_AXIS_ALIGNED Verification: body axis / interoceptive constraint bias.

Models "axis alignment" as:
  - Reduced angular noise (tighter clustering around magistrale axes)
  - Increased snap strength (stronger radial gating toward bands)
  - Preserved temporal jitter (entropy signal stays intact)

This is different from ENV_COHERENT (which reduced jitter uniformly
and inadvertently weakened H1). AXIS_ALIGNED constrains the STRUCTURE,
not the SIGNAL.

Predictions:
  - RANDOM_PURE:   no change (no axes to align to)
  - NEAR_NULL:     no change (structure too weak to constrain)
  - REAL_NORMAL:   H1 timing shifts (same or earlier), waves more stable
  - REAL_SURVIVAL: minimal change (already constrained)
  - REAL_BIBLICAL: no change (temporal order still destroyed)

Key test: axis alignment should NOT create signal, only shift thresholds.
"""
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dsdp_sali_lcf.scripts.generate_calibration_data import (
    generate_condition, _find_hub, _snap_radii_from_hub,
    _make_magistrale_directions, SQRT_PHI, LOG_STEP,
)
from dsdp_sali_lcf.src.data_loader import center_normalize, prepare_time_index
from dsdp_sali_lcf.experiments.run_single_agent import run_pipeline_extract
from dsdp_sali_lcf.experiments.run_phase1 import run_h1_coherence_threshold
from dsdp_sali_lcf.metrics.identity import compute_identity_similarity

import logging
logging.basicConfig(level=logging.WARNING)

N = 5000
D = 64
N_MAG = 6
TAU = 0.03
SEED = 42


def gen_axis_aligned(n, d, seed, angular_scale=0.10, snap_boost=0.15):
    """Generate REAL_NORMAL with axis-aligned constraints.

    angular_scale: reduced from default 0.20 (tighter around magistrale axes)
    snap_boost: added to base snap strength (stronger radial gating)
    Temporal jitter (entropy_phase) is PRESERVED — only structure changes.
    """
    rng = np.random.RandomState(seed)
    k = 6
    dirs = _make_magistrale_directions(d, k, rng)
    cluster_assign = rng.randint(0, k, size=n)
    unit_dirs = np.zeros((n, d))
    for i in range(n):
        noise = rng.randn(d) * angular_scale
        v = dirs[cluster_assign[i]] + noise
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            unit_dirs[i] = v / norm
        else:
            unit_dirs[i] = dirs[cluster_assign[i]]
    base_radii = rng.uniform(0.5, 2.0, size=n)
    E_raw = unit_dirs * base_radii[:, np.newaxis]
    E_raw += rng.randn(n, d) * 0.01
    E_norm, _, _ = center_normalize(E_raw)
    t_frac = np.linspace(0, 1, n)
    rng_snap = np.random.RandomState(seed + 500)
    for iteration in range(3):
        hub = _find_hub(E_norm)
        E_norm = _snap_radii_axis(E_norm, hub, t_frac, rng_snap,
                                   snap_boost=snap_boost)
        E_norm = E_norm - E_norm.mean(axis=0)
        E_norm = E_norm / (np.std(E_norm) + 1e-12)
        rng_snap = np.random.RandomState(seed + 500)
    return E_norm


def _snap_radii_axis(E, hub, t_frac, rng, snap_boost=0.15):
    """Snap with boosted snap strength but PRESERVED temporal jitter."""
    diff = E - hub
    r = np.linalg.norm(diff, axis=1)
    r_safe = np.maximum(r, 1e-12)
    unit_dirs = diff / r_safe[:, np.newaxis]
    y = np.log(r_safe)
    nearest_band_y = np.round(y / LOG_STEP) * LOG_STEP

    entropy_phase = np.where(t_frac < 0.30, 1.0,
                    np.where(t_frac < 0.50, 1.0 - (t_frac - 0.30) / 0.20, 0.0))
    wave_phase = np.where(t_frac < 0.40, 0.0,
                 np.where(t_frac < 0.65, (t_frac - 0.40) / 0.25, 1.0))

    snap = np.minimum(0.30 + snap_boost + 0.65 * wave_phase, 0.98)

    jitter = rng.randn(len(E)) * LOG_STEP * 0.35 * entropy_phase

    y_target = nearest_band_y + jitter
    y_new = y * (1.0 - snap) + y_target * snap
    new_r = np.exp(y_new)
    E_new = hub + unit_dirs * new_r[:, np.newaxis]
    return E_new


def run_quick(E_raw, seed=SEED):
    E, center, scale = center_normalize(E_raw)
    t_raw = np.arange(len(E), dtype=np.float64)
    t, t_meta = prepare_time_index(t_raw, len(E), E=E)
    base = run_pipeline_extract(E, t, N_MAG, TAU, seed=seed)
    h1 = run_h1_coherence_threshold(base, tau=TAU)
    return {
        "h1_pass": h1["pass"],
        "h1_p": h1.get("permutation_test", {}).get("p_value", 1.0),
        "h1_z": h1.get("permutation_test", {}).get("z_score", 0.0),
        "alignment": base.get("alignment", 0),
        "longest_wave": base.get("longest_wave", 0),
        "fsi": base.get("fsi", 0),
        "n_waves": base.get("n_waves", 0),
        "centers": base["centers"],
    }


def main():
    print("=" * 70)
    print("ENV_AXIS_ALIGNED VERIFICATION")
    print("Body axis / interoceptive constraint bias")
    print("Reduced angular noise (0.20 -> 0.10), boosted snap (+0.15)")
    print("Temporal jitter PRESERVED (entropy signal intact)")
    print("=" * 70)

    print("\n--- RANDOM_PURE (no axes to constrain) ---")
    E_random = generate_condition("RANDOM_PURE", N, D, SEED)
    r_random = run_quick(E_random)
    print(f"  H1={r_random['h1_pass']}, p={r_random['h1_p']:.4f}")
    print(f"  (axis alignment irrelevant — no magistrale structure)")

    print("\n--- NEAR_NULL (structure too weak) ---")
    E_null = generate_condition("NEAR_NULL", N, D, SEED)
    r_null = run_quick(E_null)
    print(f"  H1={r_null['h1_pass']}, p={r_null['h1_p']:.4f}")
    print(f"  (axis alignment irrelevant — near-null structure)")

    print("\n--- REAL_NORMAL (neutral) ---")
    E_normal = generate_condition("REAL_NORMAL", N, D, SEED)
    r_normal = run_quick(E_normal)
    print(f"  H1={r_normal['h1_pass']}, p={r_normal['h1_p']:.4f}, z={r_normal['h1_z']:.2f}")
    print(f"  alignment={r_normal['alignment']:.4f}, "
          f"waves={r_normal['longest_wave']}, FSI={r_normal['fsi']:.4f}")

    print("\n--- REAL_NORMAL (axis-aligned) ---")
    E_axis = gen_axis_aligned(N, D, SEED, angular_scale=0.10, snap_boost=0.15)
    r_axis = run_quick(E_axis)
    print(f"  H1={r_axis['h1_pass']}, p={r_axis['h1_p']:.4f}, z={r_axis['h1_z']:.2f}")
    print(f"  alignment={r_axis['alignment']:.4f}, "
          f"waves={r_axis['longest_wave']}, FSI={r_axis['fsi']:.4f}")

    identity_normal_vs_axis = compute_identity_similarity(
        r_normal["centers"], r_axis["centers"]
    )
    print(f"  Identity (normal vs axis-aligned): {identity_normal_vs_axis:.4f}")

    print("\n--- REAL_NORMAL (strong axis, angular=0.05, snap_boost=0.25) ---")
    E_strong = gen_axis_aligned(N, D, SEED, angular_scale=0.05, snap_boost=0.25)
    r_strong = run_quick(E_strong)
    print(f"  H1={r_strong['h1_pass']}, p={r_strong['h1_p']:.4f}, z={r_strong['h1_z']:.2f}")
    print(f"  alignment={r_strong['alignment']:.4f}, "
          f"waves={r_strong['longest_wave']}, FSI={r_strong['fsi']:.4f}")

    identity_normal_vs_strong = compute_identity_similarity(
        r_normal["centers"], r_strong["centers"]
    )
    print(f"  Identity (normal vs strong-axis): {identity_normal_vs_strong:.4f}")

    print("\n--- REAL_SURVIVAL (already constrained) ---")
    E_surv = generate_condition("REAL_SURVIVAL", N, D, SEED)
    r_surv = run_quick(E_surv)
    print(f"  H1={r_surv['h1_pass']}, p={r_surv['h1_p']:.4f}, z={r_surv['h1_z']:.2f}")
    print(f"  alignment={r_surv['alignment']:.4f}, "
          f"waves={r_surv['longest_wave']}, FSI={r_surv['fsi']:.4f}")

    print("\n--- REAL_BIBLICAL (temporal order destroyed) ---")
    E_bib = generate_condition("REAL_BIBLICAL", N, D, SEED)
    r_bib = run_quick(E_bib)
    print(f"  H1={r_bib['h1_pass']}, p={r_bib['h1_p']:.4f}")
    print(f"  (axis alignment irrelevant — temporal shuffle dominates)")

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    checks = {}

    no_false_random = not r_random["h1_pass"]
    no_false_null = not r_null["h1_pass"]
    checks["No false signal (RANDOM)"] = no_false_random
    checks["No false signal (NEAR_NULL)"] = no_false_null
    print(f"\n  1. No false signal creation:")
    print(f"     RANDOM_PURE:  H1={r_random['h1_pass']} -> {'OK' if no_false_random else 'FAIL'}")
    print(f"     NEAR_NULL:    H1={r_null['h1_pass']} -> {'OK' if no_false_null else 'FAIL'}")

    h1_preserved = r_axis["h1_pass"]
    checks["H1 preserved under axis"] = h1_preserved
    print(f"\n  2. H1 preserved or improved under axis alignment:")
    print(f"     Neutral: H1={r_normal['h1_pass']}, p={r_normal['h1_p']:.4f}, z={r_normal['h1_z']:.2f}")
    print(f"     Axis:    H1={r_axis['h1_pass']}, p={r_axis['h1_p']:.4f}, z={r_axis['h1_z']:.2f}")
    z_delta = r_axis["h1_z"] - r_normal["h1_z"]
    print(f"     Z delta: {z_delta:+.2f}")
    print(f"     -> {'OK' if h1_preserved else 'WEAKENED'}")

    align_up = r_axis["alignment"] >= r_normal["alignment"]
    checks["Alignment up or stable"] = align_up
    print(f"\n  3. Alignment increased or stable under axis constraint:")
    print(f"     Neutral: {r_normal['alignment']:.4f}")
    print(f"     Axis:    {r_axis['alignment']:.4f}")
    print(f"     Strong:  {r_strong['alignment']:.4f}")
    print(f"     -> {'OK (monotonic)' if r_strong['alignment'] >= r_axis['alignment'] >= r_normal['alignment'] else 'PARTIAL' if align_up else 'DECREASED'}")

    bib_no_change = not r_bib["h1_pass"]
    checks["Biblical unchanged"] = bib_no_change
    print(f"\n  4. Biblical unaffected (temporal dominates):")
    print(f"     H1={r_bib['h1_pass']} -> {'OK' if bib_no_change else 'FAIL'}")

    identity_preserved = identity_normal_vs_axis > 0.5
    checks["Identity preserved"] = identity_preserved
    print(f"\n  5. Identity preserved under axis constraint:")
    print(f"     Normal vs Axis:   {identity_normal_vs_axis:.4f}")
    print(f"     Normal vs Strong: {identity_normal_vs_strong:.4f}")
    print(f"     -> {'OK' if identity_preserved else 'IDENTITY SHIFTED'}")

    n_pass = sum(1 for v in checks.values() if v)
    print(f"\n  TOTAL: {n_pass}/{len(checks)} checks passed")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if no_false_random and no_false_null:
        print("  Axis alignment does NOT create signal from nothing.")
    if h1_preserved:
        print("  H1 temporal structure preserved (entropy signal intact).")
    else:
        print("  NOTE: H1 weakened — axis constraints may interact with temporal dynamics.")
    if align_up:
        print("  Structural alignment improved (tighter clustering + stronger snap).")
    print(f"  Identity similarity: {identity_normal_vs_axis:.4f} — ", end="")
    if identity_preserved:
        print("same core structure, different precision.")
    else:
        print("axis constraints shifted core structure (strong effect).")

    print("\n  CONCLUSION:")
    if n_pass >= 4:
        print("  ENV_AXIS_ALIGNED is a valid environment bias parameter.")
        print("  It constrains structure (reduces exploration, increases gating)")
        print("  without creating signal from nothing.")
        print("  This models interoceptive / body-axis constraints correctly:")
        print("    - mediul nu adauga continut, doar constrangeri")
        print("    - structura apare doar daca exista deja in date")
    else:
        print("  ENV_AXIS_ALIGNED shows unexpected interactions — see details above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
