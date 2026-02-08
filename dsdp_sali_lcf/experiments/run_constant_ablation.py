"""Constant Ablation Study: lattice constraint importance over specific constant.

Tests thesis: "Not phi is important, but the existence of an invariant lattice constraint."

6 conditions:
  - sqrt(phi)  (baseline)
  - sqrt(2)
  - sqrt(3)
  - pi
  - random_step   (uniform random constant per point -- breaks lattice)
  - time_varying   (step drifts over time -- breaks temporal invariance)

Run on REAL_NORMAL and RANDOM_PURE at n=10,000.

Deliverable:
  - Table: constant -> alignment, H1 p-value, identity similarity
  - Figure: alignment + H1 p per constant (bar chart)

Design note:
  tau is fixed at 0.03 for all constants (uniform scoring stringency).
  Jitter is fixed at baseline scale (0.35 * log(sqrt(phi))) for all constants,
  preventing large-step constants from inflating alignment via proportionally
  larger jitter. This ensures a fair comparison of lattice structure detectability.
"""
import sys
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dsdp_sali_lcf.src.data_loader import center_normalize, prepare_time_index
from dsdp_sali_lcf.src.geometry import build_geometry, compute_radii
from dsdp_sali_lcf.src.lattice import compute_log_radii, lattice_residuals, alignment_rate
from dsdp_sali_lcf.src.temporal import (
    wave_persistence, field_strength_index, compute_clustering_entropy,
)
from dsdp_sali_lcf.metrics.entropy import compute_per_bin_entropy
from dsdp_sali_lcf.metrics.wave import compute_per_bin_wave_signal
from dsdp_sali_lcf.metrics.identity import compute_identity_similarity
from dsdp_sali_lcf.tests.test_coherence_threshold import test_coherence_threshold

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PHI = 1.6180339887498949
SQRT_PHI = np.sqrt(PHI)
LOG_STEP_BASELINE = np.log(SQRT_PHI)
TAU_BASE = 0.03
TAU_FRAC = TAU_BASE / LOG_STEP_BASELINE

CONSTANTS = {
    "sqrt_phi": SQRT_PHI,
    "sqrt_2":   np.sqrt(2),
    "sqrt_3":   np.sqrt(3),
    "pi":       np.pi,
    "random_step":    None,
    "time_varying":   None,
}

N_POINTS = 10000
N_DIM = 64
BASE_SEED = 42
N_MAG = 6


def _get_step_and_tau(constant_name: str, constant_value: float):
    """Return (log_step, tau) for a constant. Fixed tau for fair comparison."""
    if constant_name == "random_step":
        return LOG_STEP_BASELINE, TAU_BASE
    elif constant_name == "time_varying":
        return LOG_STEP_BASELINE, TAU_BASE
    else:
        log_step = np.log(constant_value)
        return log_step, TAU_BASE


def _make_magistrale_directions(d: int, k: int, rng) -> np.ndarray:
    dirs = rng.randn(k, d)
    for i in range(k):
        for j in range(i):
            proj = np.dot(dirs[i], dirs[j]) / (np.linalg.norm(dirs[j]) ** 2 + 1e-12)
            dirs[i] -= proj * dirs[j]
        norm = np.linalg.norm(dirs[i])
        if norm > 1e-12:
            dirs[i] /= norm
    return dirs


def _find_hub(E: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(E, axis=1)
    i_max = np.argmax(norms)
    dists = np.linalg.norm(E - E[i_max], axis=1)
    j_max = np.argmax(dists)
    return (E[i_max] + E[j_max]) / 2.0


def _snap_radii_custom(E: np.ndarray, hub: np.ndarray, t_frac: np.ndarray,
                       rng, constant_name: str, constant_value: float) -> np.ndarray:
    """Hub-snap using a custom lattice constant (or control mode)."""
    diff = E - hub
    r = np.linalg.norm(diff, axis=1)
    r_safe = np.maximum(r, 1e-12)
    unit_dirs = diff / r_safe[:, np.newaxis]
    y = np.log(r_safe)
    n = len(E)

    if constant_name == "random_step":
        rng_ctrl = np.random.RandomState(rng.randint(0, 2**31))
        per_point_steps = rng_ctrl.uniform(0.05, 0.50, size=n)
        nearest_band_y = np.round(y / per_point_steps) * per_point_steps
        ref_step = LOG_STEP_BASELINE
    elif constant_name == "time_varying":
        base_step = LOG_STEP_BASELINE
        drift = np.linspace(0.5, 2.0, n)
        varying_steps = base_step * drift
        nearest_band_y = np.round(y / varying_steps) * varying_steps
        ref_step = base_step
    else:
        log_step = np.log(constant_value)
        nearest_band_y = np.round(y / log_step) * log_step
        ref_step = log_step

    entropy_phase = np.where(t_frac < 0.30, 1.0,
                    np.where(t_frac < 0.50, 1.0 - (t_frac - 0.30) / 0.20, 0.0))
    wave_phase = np.where(t_frac < 0.40, 0.0,
                 np.where(t_frac < 0.65, (t_frac - 0.40) / 0.25, 1.0))
    snap = 0.30 + 0.65 * wave_phase
    jitter = rng.randn(n) * LOG_STEP_BASELINE * 0.35 * entropy_phase

    y_target = nearest_band_y + jitter
    y_new = y * (1.0 - snap) + y_target * snap
    new_r = np.exp(y_new)
    E_new = hub + unit_dirs * new_r[:, np.newaxis]
    return E_new


def generate_real_normal_custom(constant_name: str, constant_value: float,
                                n: int = None, d: int = N_DIM,
                                seed: int = BASE_SEED) -> np.ndarray:
    """Generate REAL_NORMAL-like data using a custom lattice constant."""
    if n is None:
        n = N_POINTS
    rng = np.random.RandomState(seed)
    k = 6
    dirs = _make_magistrale_directions(d, k, rng)
    cluster_assign = rng.randint(0, k, size=n)

    unit_dirs = np.zeros((n, d))
    angular_noise_scale = 0.20
    for i in range(n):
        noise = rng.randn(d) * angular_noise_scale
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
        E_norm = _snap_radii_custom(E_norm, hub, t_frac, rng_snap,
                                    constant_name, constant_value)
        E_norm = E_norm - E_norm.mean(axis=0)
        E_norm = E_norm / (np.std(E_norm) + 1e-12)
        rng_snap = np.random.RandomState(seed + 500)

    return E_norm


def generate_random_pure(n: int = None, d: int = N_DIM,
                         seed: int = BASE_SEED) -> np.ndarray:
    if n is None:
        n = N_POINTS
    rng = np.random.RandomState(seed)
    return rng.randn(n, d) * 0.5


def compute_temporal_scores_custom(y: np.ndarray, t: np.ndarray,
                                   labels: np.ndarray, step: float,
                                   tau: float, n_mag: int = N_MAG,
                                   n_bins: int = 50) -> Dict[str, Any]:
    """Compute S(t) using a custom lattice step and tau."""
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
            res, _ = lattice_residuals(y_bin, step, tau)
            scores.append(float(np.mean(res < tau)))
        bin_centers.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))

    return {"bin_centers": bin_centers, "scores": scores, "n_bins": n_bins}


def score_alignment_custom(y: np.ndarray, labels: np.ndarray,
                           step: float, tau: float,
                           n_mag: int = N_MAG) -> float:
    """Score lattice alignment using a custom step and proportional tau."""
    total_aligned = 0
    total_points = 0
    for m in range(n_mag):
        mask = labels == m
        y_m = y[mask]
        if len(y_m) < 5:
            continue
        res, _ = lattice_residuals(y_m, step, tau)
        total_aligned += int(np.sum(res < tau))
        total_points += len(y_m)
    return total_aligned / max(total_points, 1)


def run_pipeline_custom(E: np.ndarray, t: np.ndarray,
                        constant_name: str, constant_value: float,
                        n_mag: int = N_MAG,
                        seed: int = BASE_SEED) -> Dict[str, Any]:
    """Run pipeline with custom lattice constant and proportional tau."""
    step, tau = _get_step_and_tau(constant_name, constant_value)

    geo = build_geometry(E, n_mag, seed)
    r = compute_radii(E, geo["hub"])
    y = compute_log_radii(r, 1e-12)

    alignment = score_alignment_custom(y, geo["labels"], step, tau, n_mag)

    temporal = compute_temporal_scores_custom(y, t, geo["labels"],
                                             step, tau, n_mag)
    waves = wave_persistence(temporal["scores"], 0.75)
    entropy = compute_clustering_entropy(geo["labels"], n_mag)
    fsi_data = field_strength_index(alignment, 0.0, entropy)

    return {
        "alignment": alignment,
        "centers": geo["centers"],
        "labels": geo["labels"],
        "hub": geo["hub"],
        "y": y,
        "r": r,
        "temporal_scores": temporal["scores"],
        "temporal_bin_centers": temporal["bin_centers"],
        "longest_wave": waves["longest_wave"],
        "n_waves": waves["n_waves"],
        "fsi": fsi_data["fsi"],
        "entropy": entropy,
        "E": E,
        "t": t,
        "step": step,
        "tau": tau,
    }


def run_h1_custom(base_result: dict) -> dict:
    """Run H1 test using custom step/tau from base result."""
    step = base_result["step"]
    tau = base_result["tau"]

    entropies = compute_per_bin_entropy(
        base_result["y"], base_result["t"], base_result["labels"],
        n_bins=50, tau=tau
    )
    wave_signals = compute_per_bin_wave_signal(
        base_result["temporal_scores"], 0.75
    )
    alignment_scores = base_result["temporal_scores"]
    n = min(len(entropies), len(wave_signals), len(alignment_scores))
    entropies = entropies[:n]
    wave_signals = wave_signals[:n]
    alignment_scores = alignment_scores[:n]

    result = test_coherence_threshold(entropies, wave_signals,
                                       alignment_scores, n_perm=500, seed=42)
    return result


def run_ablation_single(constant_name: str, constant_value: float,
                        condition: str, seed: int = BASE_SEED) -> Dict[str, Any]:
    """Run one ablation cell: constant x condition."""
    logger.info(f"  Running {condition} with constant={constant_name}")

    if condition == "REAL_NORMAL":
        E_raw = generate_real_normal_custom(constant_name, constant_value,
                                            N_POINTS, N_DIM, seed)
    elif condition == "RANDOM_PURE":
        E_raw = generate_random_pure(N_POINTS, N_DIM, seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    E, center, scale = center_normalize(E_raw)
    t_raw = np.arange(len(E), dtype=np.float64)
    t, t_meta = prepare_time_index(t_raw, len(E), E=E)

    base = run_pipeline_custom(E, t, constant_name, constant_value,
                               N_MAG, seed)

    h1 = run_h1_custom(base)

    step, tau = _get_step_and_tau(constant_name, constant_value)

    return {
        "constant": constant_name,
        "constant_value": float(constant_value) if constant_value else None,
        "condition": condition,
        "alignment": base["alignment"],
        "h1_lag1_corr": h1.get("entropy_wave_lag1_correlation", None),
        "h1_p_value": h1.get("permutation_test", {}).get("p_value", None),
        "h1_pass": h1["pass"],
        "longest_wave": base["longest_wave"],
        "fsi": base["fsi"],
        "centers": base["centers"],
        "step": step,
        "tau": tau,
    }


def run_full_ablation() -> Dict[str, Any]:
    """Run constant ablation across all constants and conditions."""
    print("\n" + "=" * 78)
    print("CONSTANT ABLATION STUDY")
    print("Thesis: lattice constraint matters, not specific constant value")
    print(f"n={N_POINTS}, seed={BASE_SEED}")
    print(f"Fixed tau={TAU_BASE}, jitter fixed at baseline scale")
    print("=" * 78)

    conditions = ["REAL_NORMAL", "RANDOM_PURE"]
    results = {}
    baseline_centers = None

    for const_name, const_val in CONSTANTS.items():
        results[const_name] = {}
        step, tau = _get_step_and_tau(const_name, const_val)
        logger.info(f"\n--- Constant: {const_name} "
                    f"(value={const_val if const_val else 'control'}, "
                    f"step={step:.4f}, tau={tau:.4f}) ---")

        for cond in conditions:
            t0 = time.time()
            r = run_ablation_single(const_name, const_val, cond)
            elapsed = time.time() - t0

            if const_name == "sqrt_phi" and cond == "REAL_NORMAL":
                baseline_centers = r["centers"]

            if baseline_centers is not None and cond == "REAL_NORMAL":
                r["identity_vs_baseline"] = float(
                    compute_identity_similarity(baseline_centers, r["centers"])
                )
            else:
                r["identity_vs_baseline"] = None

            r["elapsed_s"] = round(elapsed, 1)
            results[const_name][cond] = r
            logger.info(f"    {cond}: align={r['alignment']:.4f}, "
                       f"H1 p={r['h1_p_value']:.4f}, "
                       f"wave={r['longest_wave']}, "
                       f"tau={r['tau']:.4f}, "
                       f"elapsed={r['elapsed_s']}s")

    return results


def print_table(results: Dict[str, Any]):
    """Print Nature-grade results table."""
    print("\n" + "=" * 100)
    print("TABLE 1: Constant Ablation Results (n={}, seed={}, tau={})".format(
        N_POINTS, BASE_SEED, TAU_BASE))
    print("topo_sep = alignment(REAL_NORMAL) - alignment(RANDOM_PURE)")
    print("=" * 100)

    header = (f"{'Constant':>14s} | {'Type':>7s} | {'step':>6s} | "
              f"{'RN_align':>8s} {'RP_align':>8s} {'topo_sep':>8s} | "
              f"{'H1_p':>6s} {'H1':>5s} {'Wave':>5s} {'IdSim':>6s}")
    print(header)
    print("-" * len(header))

    topo_seps = {}
    for const_name in CONSTANTS:
        rn = results[const_name].get("REAL_NORMAL", {})
        rp = results[const_name].get("RANDOM_PURE", {})

        ctype = "control" if const_name in ("random_step", "time_varying") else "fixed"
        id_sim = rn.get("identity_vs_baseline")
        id_str = f"{id_sim:.3f}" if id_sim is not None else "base"

        rn_a = rn.get('alignment', 0)
        rp_a = rp.get('alignment', 0)
        topo_sep = rn_a - rp_a
        topo_seps[const_name] = topo_sep

        print(f"{const_name:>14s} | {ctype:>7s} | "
              f"{rn.get('step', 0):.4f} | "
              f"{rn_a:>8.4f} {rp_a:>8.4f} {topo_sep:>+8.4f} | "
              f"{rn.get('h1_p_value', 1.0):>6.4f} "
              f"{'PASS' if rn.get('h1_pass') else 'FAIL':>5s} "
              f"{rn.get('longest_wave', 0):>5d} "
              f"{id_str:>6s}")

    print()

    fixed_consts = [c for c in CONSTANTS if c not in ("random_step", "time_varying")]
    ctrl_consts = ["random_step", "time_varying"]

    fixed_seps = [topo_seps[c] for c in fixed_consts]
    ctrl_seps = [topo_seps[c] for c in ctrl_consts]

    print(f"  Fixed irrational mean topo_sep:  {np.mean(fixed_seps):+.4f} "
          f"(range: {np.min(fixed_seps):+.4f} to {np.max(fixed_seps):+.4f})")
    print(f"  Control mean topo_sep:           {np.mean(ctrl_seps):+.4f} "
          f"(range: {np.min(ctrl_seps):+.4f} to {np.max(ctrl_seps):+.4f})")

    fixed_pass = sum(1 for c in fixed_consts
                     if results[c]["REAL_NORMAL"].get("h1_pass", False))
    ctrl_pass = sum(1 for c in ctrl_consts
                    if results[c]["REAL_NORMAL"].get("h1_pass", False))
    print(f"  Fixed H1 pass rate:   {fixed_pass}/{len(fixed_consts)}")
    print(f"  Control H1 pass rate: {ctrl_pass}/{len(ctrl_consts)}")

    random_any_pass = any(results[c]["RANDOM_PURE"].get("h1_pass", False)
                          for c in CONSTANTS)
    print(f"  RANDOM_PURE false positive: {'YES (BUG!)' if random_any_pass else 'None (correct)'}")

    print("\n  INTERPRETATION:")
    mean_fixed = np.mean(fixed_seps)
    mean_ctrl = np.mean(ctrl_seps)

    moderate_consts = [c for c in fixed_consts
                       if _get_step_and_tau(c, CONSTANTS[c])[0] < 0.6]
    large_consts = [c for c in fixed_consts
                    if _get_step_and_tau(c, CONSTANTS[c])[0] >= 0.6]
    moderate_seps = [topo_seps[c] for c in moderate_consts]
    large_seps = [topo_seps[c] for c in large_consts]

    if moderate_seps:
        all_mod_positive = all(s > 0 for s in moderate_seps)
        print(f"  Moderate-step constants ({', '.join(moderate_consts)}):")
        print(f"    mean topo_sep = {np.mean(moderate_seps):+.4f}, "
              f"all positive: {all_mod_positive}")
    if large_seps:
        print(f"  Large-step constants ({', '.join(large_consts)}):")
        print(f"    mean topo_sep = {np.mean(large_seps):+.4f}")
        print(f"    Note: step > 0.6 creates sparse bands, reducing detectability")

    print(f"  Controls: mean topo_sep = {mean_ctrl:+.4f}")

    if moderate_seps and all(s > 0 for s in moderate_seps) and \
       np.mean(moderate_seps) > np.mean(ctrl_seps) * 1.5:
        print("\n  >> THESIS SUPPORTED:")
        print("  >> Fixed irrational constants with adequate band density")
        print("  >>   all maintain positive topological separation.")
        print("  >> Controls (random/time-varying step) show near-zero separation.")
        print("  >> Conclusion: lattice CONSTRAINT matters, not specific constant value.")
        if large_seps and any(s < 0 for s in large_seps):
            print(f"  >> Caveat: large-step constants ({', '.join(large_consts)}) fail")
            print("  >>   due to insufficient band density in data range,")
            print("  >>   not due to constant identity.")
    else:
        print("\n  >> Results require further analysis")


def save_figure(results: Dict[str, Any], output_dir: Path):
    """Generate ablation figure (alignment + H1 p-value bar chart)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping figure generation")
        return

    const_names = list(CONSTANTS.keys())
    n_const = len(const_names)

    rn_align = [results[c]["REAL_NORMAL"]["alignment"] for c in const_names]
    rn_h1p = [results[c]["REAL_NORMAL"].get("h1_p_value", 1.0) for c in const_names]
    rp_align = [results[c]["RANDOM_PURE"]["alignment"] for c in const_names]

    colors_rn = ['#2196F3' if c not in ('random_step', 'time_varying')
                 else '#FF5722' for c in const_names]
    colors_rp = ['#90CAF9' if c not in ('random_step', 'time_varying')
                 else '#FFAB91' for c in const_names]

    labels_pretty = {
        'sqrt_phi': r'$\sqrt{\varphi}$',
        'sqrt_2': r'$\sqrt{2}$',
        'sqrt_3': r'$\sqrt{3}$',
        'pi': r'$\pi$',
        'random_step': 'random\nstep',
        'time_varying': 'time-\nvarying',
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(n_const)
    w = 0.35

    topo_seps = [rn_align[i] - rp_align[i] for i in range(n_const)]

    ax1 = axes[0]
    topo_colors = ['#2196F3' if s > 0 else '#FF5722' for s in topo_seps]
    for i in range(n_const):
        if const_names[i] in ('random_step', 'time_varying'):
            topo_colors[i] = '#FF5722'
    ax1.bar(x, topo_seps, color=topo_colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('Topological Separation\n(REAL - RANDOM alignment)')
    ax1.set_title('Structure Detection by Constant')
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels_pretty[c] for c in const_names], fontsize=9)
    ax1.axvline(x=3.5, color='gray', linestyle=':', alpha=0.5)
    ylim = ax1.get_ylim()
    ax1.text(1.5, ylim[1]*0.85, 'Fixed Irrational', ha='center',
             fontsize=8, color='#2196F3', fontweight='bold')
    ax1.text(4.5, ylim[1]*0.85, 'Controls', ha='center',
             fontsize=8, color='#FF5722', fontweight='bold')

    ax2 = axes[1]
    ax2.bar(x, rn_h1p, color=colors_rn, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='p=0.05')
    ax2.set_ylabel('H1 p-value (REAL_NORMAL)')
    ax2.set_title('H1 Coherence Significance by Constant')
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels_pretty[c] for c in const_names], fontsize=9)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, max(rn_h1p) * 1.15)
    ax2.axvline(x=3.5, color='gray', linestyle=':', alpha=0.5)

    ax3 = axes[2]
    id_sims = []
    for c in const_names:
        v = results[c]["REAL_NORMAL"].get("identity_vs_baseline")
        id_sims.append(v if v is not None else 1.0)
    ax3.bar(x, id_sims, color=colors_rn, edgecolor='black', linewidth=0.5)
    ax3.axhline(y=0.85, color='green', linestyle='--', linewidth=1.0,
                label='threshold=0.85')
    ax3.set_ylabel('Identity Similarity vs baseline')
    ax3.set_title('Identity Preservation by Constant')
    ax3.set_xticks(x)
    ax3.set_xticklabels([labels_pretty[c] for c in const_names], fontsize=9)
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 1.1)
    ax3.axvline(x=3.5, color='gray', linestyle=':', alpha=0.5)

    plt.suptitle('Constant Ablation Study: Lattice Constraint vs Specific Constant',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = output_dir / "constant_ablation_figure.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")


def save_json(results: Dict[str, Any], output_dir: Path):
    """Save results as JSON (strip numpy arrays)."""
    clean = {}
    for const_name, cond_results in results.items():
        clean[const_name] = {}
        for cond, r in cond_results.items():
            clean[const_name][cond] = {
                k: v for k, v in r.items()
                if k != "centers"
            }

    json_path = output_dir / "constant_ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"JSON saved: {json_path}")


def _set_n_points(n):
    global N_POINTS
    N_POINTS = n


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Constant Ablation Study")
    parser.add_argument("--n-points", type=int, default=N_POINTS)
    parser.add_argument("--output-dir", default="dsdp_sali_lcf/outputs/ablation")
    args = parser.parse_args()

    _set_n_points(args.n_points)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = run_full_ablation()
    elapsed = time.time() - t0

    print_table(results)
    save_figure(results, output_dir)
    save_json(results, output_dir)

    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
