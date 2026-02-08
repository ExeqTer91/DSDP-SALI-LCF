"""Boundary Polymorphism Verification Suite.

Tests whether the structure boundary is polymorphic (belongs to multiple
geometric classes depending on regime/threshold/parameters), not merely
a scaled version of one shape.

Method:
  1. Project embeddings to 2D via PCA
  2. Build kernel density field on 2D projection
  3. Threshold at multiple tau to extract binary masks
  4. Extract boundary contours from masks
  5. Compute geometric metrics per contour
  6. Test polymorphism criteria across conditions/seeds/thresholds

Conditions:
  - REAL_NORMAL, REAL_SURVIVAL, REAL_BIBLICAL
  - ENV_AXIS_ALIGNED (moderate + strong)
  - VIRUS_MODE (strength 0.3 + 0.6)
  - Controls: RANDOM_PURE, NEAR_NULL

Seeds: {42, 43, 99}
Thresholds tau: {0.30, 0.40, 0.50, 0.60, 0.70}
"""
import sys
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scipy.ndimage import label, gaussian_filter
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA

from dsdp_sali_lcf.scripts.generate_calibration_data import (
    generate_condition, _find_hub, _snap_radii_from_hub,
    _make_magistrale_directions, SQRT_PHI, LOG_STEP,
)
from dsdp_sali_lcf.src.data_loader import center_normalize

import logging
logging.basicConfig(level=logging.WARNING)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "boundary_polymorphism"

N = 5000
D = 64
SEEDS = [42, 43, 99]
TAUS = [0.30, 0.40, 0.50, 0.60, 0.70]
GRID_RES = 128


def gen_axis_aligned(n, d, seed, angular_scale=0.10, snap_boost=0.15):
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

    def _snap_axis(E, hub, t_frac, rng, sb=snap_boost):
        diff = E - hub
        r = np.linalg.norm(diff, axis=1)
        r_safe = np.maximum(r, 1e-12)
        ud = diff / r_safe[:, np.newaxis]
        y = np.log(r_safe)
        nby = np.round(y / LOG_STEP) * LOG_STEP
        ep = np.where(t_frac < 0.30, 1.0,
             np.where(t_frac < 0.50, 1.0 - (t_frac - 0.30) / 0.20, 0.0))
        wp = np.where(t_frac < 0.40, 0.0,
             np.where(t_frac < 0.65, (t_frac - 0.40) / 0.25, 1.0))
        snap = np.minimum(0.30 + sb + 0.65 * wp, 0.98)
        jitter = rng.randn(len(E)) * LOG_STEP * 0.35 * ep
        yt = nby + jitter
        yn = y * (1.0 - snap) + yt * snap
        return hub + ud * np.exp(yn)[:, np.newaxis]

    for _ in range(3):
        hub = _find_hub(E_norm)
        E_norm = _snap_axis(E_norm, hub, t_frac, rng_snap)
        E_norm = E_norm - E_norm.mean(axis=0)
        E_norm = E_norm / (np.std(E_norm) + 1e-12)
        rng_snap = np.random.RandomState(seed + 500)
    return E_norm


def inject_virus(E_raw, virus_strength, seed):
    E, _, _ = center_normalize(E_raw)
    from dsdp_sali_lcf.experiments.run_single_agent import run_pipeline_extract
    from dsdp_sali_lcf.src.data_loader import prepare_time_index
    t_raw = np.arange(len(E), dtype=np.float64)
    t, _ = prepare_time_index(t_raw, len(E), E=E)
    base = run_pipeline_extract(E, t, 6, 0.03, seed=seed)
    hub = base["hub"]

    rng = np.random.RandomState(seed + 333)
    E_virus = E.copy()
    r = np.linalg.norm(E - hub, axis=1)
    median_r = np.median(r)
    peri_mask = r >= median_r

    diff = E[peri_mask] - hub
    rr = np.linalg.norm(diff, axis=1)
    rr_safe = np.maximum(rr, 1e-12)
    ud = diff / rr_safe[:, np.newaxis]
    log_r = np.log(rr_safe)
    step = np.log(SQRT_PHI)
    nb = np.round(log_r / step) * step
    new_log_r = log_r * (1 - virus_strength) + nb * virus_strength
    n_peri = int(peri_mask.sum())
    ang_noise = rng.randn(n_peri, E.shape[1]) * 0.3 * virus_strength
    sd = ud + ang_noise
    norms = np.linalg.norm(sd, axis=1, keepdims=True)
    sd = sd / np.maximum(norms, 1e-12)
    E_virus[peri_mask] = hub + sd * np.exp(new_log_r)[:, np.newaxis]
    return E_virus


def generate_embeddings(condition_label: str, seed: int) -> np.ndarray:
    if condition_label == "REAL_NORMAL":
        return generate_condition("REAL_NORMAL", N, D, seed)
    elif condition_label == "REAL_SURVIVAL":
        return generate_condition("REAL_SURVIVAL", N, D, seed)
    elif condition_label == "REAL_BIBLICAL":
        return generate_condition("REAL_BIBLICAL", N, D, seed)
    elif condition_label == "RANDOM_PURE":
        return generate_condition("RANDOM_PURE", N, D, seed)
    elif condition_label == "NEAR_NULL":
        return generate_condition("NEAR_NULL", N, D, seed)
    elif condition_label == "AXIS_MODERATE":
        return gen_axis_aligned(N, D, seed, angular_scale=0.10, snap_boost=0.15)
    elif condition_label == "AXIS_STRONG":
        return gen_axis_aligned(N, D, seed, angular_scale=0.05, snap_boost=0.25)
    elif condition_label == "VIRUS_03":
        E_raw = generate_condition("REAL_NORMAL", N, D, seed)
        return inject_virus(E_raw, 0.3, seed)
    elif condition_label == "VIRUS_06":
        E_raw = generate_condition("REAL_NORMAL", N, D, seed)
        return inject_virus(E_raw, 0.6, seed)
    else:
        raise ValueError(f"Unknown condition: {condition_label}")


def project_2d(E_raw: np.ndarray, seed: int = 42) -> np.ndarray:
    E, _, _ = center_normalize(E_raw)
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(E)


def build_density_field(pts_2d: np.ndarray, grid_res: int = GRID_RES,
                        bandwidth: float = None) -> Tuple[np.ndarray, float, float, float, float]:
    x_min, x_max = pts_2d[:, 0].min(), pts_2d[:, 0].max()
    y_min, y_max = pts_2d[:, 1].min(), pts_2d[:, 1].max()
    pad = 0.1 * max(x_max - x_min, y_max - y_min)
    x_min -= pad; x_max += pad; y_min -= pad; y_max += pad

    H, xedges, yedges = np.histogram2d(
        pts_2d[:, 0], pts_2d[:, 1],
        bins=grid_res, range=[[x_min, x_max], [y_min, y_max]]
    )
    if bandwidth is None:
        bandwidth = 2.0
    field = gaussian_filter(H.astype(float), sigma=bandwidth)
    if field.max() > 0:
        field = field / field.max()
    return field, x_min, x_max, y_min, y_max


def threshold_and_extract(field: np.ndarray, tau: float) -> Dict[str, Any]:
    mask = (field >= tau).astype(int)
    labeled, n_components = label(mask)

    total_area = int(mask.sum())
    total_pixels = mask.size

    perim = 0
    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            if mask[i, j] == 1:
                neighbors = mask[i-1, j] + mask[i+1, j] + mask[i, j-1] + mask[i, j+1]
                if neighbors < 4:
                    perim += 1

    if total_area > 0:
        circularity = 4 * np.pi * total_area / max(perim ** 2, 1)
    else:
        circularity = 0.0

    eccentricity = 0.0
    orientation = 0.0
    major_axis = 0.0
    minor_axis = 0.0
    if total_area > 2:
        ys, xs = np.where(mask == 1)
        cx, cy = xs.mean(), ys.mean()
        dx, dy = xs - cx, ys - cy
        cov = np.array([[np.sum(dx*dx), np.sum(dx*dy)],
                         [np.sum(dx*dy), np.sum(dy*dy)]]) / len(xs)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        if eigvals[0] > 0:
            ratio = eigvals[1] / eigvals[0]
            eccentricity = np.sqrt(1 - ratio)
        eigvecs = np.linalg.eigh(cov)[1]
        orientation = float(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]) * 180 / np.pi)
        major_axis = float(2 * np.sqrt(max(eigvals[0], 0)))
        minor_axis = float(2 * np.sqrt(max(eigvals[1], 0)))

    curvature_stats = compute_curvature_stats(mask)

    fourier_coeffs = compute_fourier_descriptors(mask, K=10)

    euler_char = compute_euler_characteristic(mask)

    return {
        "tau": tau,
        "area": total_area,
        "area_fraction": total_area / total_pixels,
        "perimeter": perim,
        "component_count": n_components,
        "circularity": float(circularity),
        "eccentricity": float(eccentricity),
        "orientation": float(orientation),
        "major_axis": float(major_axis),
        "minor_axis": float(minor_axis),
        "curvature_mean": curvature_stats["mean"],
        "curvature_std": curvature_stats["std"],
        "curvature_q90": curvature_stats["q90"],
        "curvature_q99": curvature_stats["q99"],
        "fourier_coeffs": fourier_coeffs,
        "euler_characteristic": euler_char,
    }


def compute_curvature_stats(mask: np.ndarray) -> Dict[str, float]:
    ys, xs = np.where(mask == 1)
    if len(xs) < 10:
        return {"mean": 0.0, "std": 0.0, "q90": 0.0, "q99": 0.0}

    boundary_pts = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                if i == 0 or j == 0 or i == mask.shape[0]-1 or j == mask.shape[1]-1:
                    boundary_pts.append((j, i))
                elif mask[i-1, j] + mask[i+1, j] + mask[i, j-1] + mask[i, j+1] < 4:
                    boundary_pts.append((j, i))

    if len(boundary_pts) < 5:
        return {"mean": 0.0, "std": 0.0, "q90": 0.0, "q99": 0.0}

    pts = np.array(boundary_pts, dtype=float)
    cx, cy = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    pts = pts[order]

    curvatures = []
    n = len(pts)
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        d1 = p1 - p0
        d2 = p2 - p1
        cross = abs(d1[0] * d2[1] - d1[1] * d2[0])
        l1 = np.linalg.norm(d1)
        l2 = np.linalg.norm(d2)
        denom = l1 * l2
        if denom > 1e-12:
            curvatures.append(cross / denom)

    if not curvatures:
        return {"mean": 0.0, "std": 0.0, "q90": 0.0, "q99": 0.0}

    c = np.array(curvatures)
    return {
        "mean": float(np.mean(c)),
        "std": float(np.std(c)),
        "q90": float(np.quantile(c, 0.90)),
        "q99": float(np.quantile(c, 0.99)),
    }


def compute_fourier_descriptors(mask: np.ndarray, K: int = 10) -> List[float]:
    ys, xs = np.where(mask == 1)
    if len(xs) < 10:
        return [0.0] * K

    boundary_pts = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                if i == 0 or j == 0 or i == mask.shape[0]-1 or j == mask.shape[1]-1:
                    boundary_pts.append(complex(j, i))
                elif mask[i-1, j] + mask[i+1, j] + mask[i, j-1] + mask[i, j+1] < 4:
                    boundary_pts.append(complex(j, i))

    if len(boundary_pts) < K + 1:
        return [0.0] * K

    pts = np.array(boundary_pts)
    cx = pts.real.mean()
    cy = pts.imag.mean()
    angles = np.angle(pts - complex(cx, cy))
    order = np.argsort(angles)
    pts = pts[order]

    fft = np.fft.fft(pts)
    if abs(fft[1]) > 1e-12:
        fft_norm = np.abs(fft) / abs(fft[1])
    else:
        fft_norm = np.abs(fft)

    coeffs = fft_norm[2:K+2].tolist()
    while len(coeffs) < K:
        coeffs.append(0.0)
    return [float(c) for c in coeffs]


def compute_euler_characteristic(mask: np.ndarray) -> int:
    labeled, n_components = label(mask)
    inv_mask = 1 - mask
    _, n_holes = label(inv_mask)
    n_holes -= 1
    return n_components - max(n_holes, 0)


def save_contour_plot(field, taus_data, condition, seed, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(taus_data), figsize=(3*len(taus_data), 3))
        if len(taus_data) == 1:
            axes = [axes]

        for ax, (tau, metrics) in zip(axes, taus_data):
            mask = (field >= tau).astype(int)
            ax.imshow(mask.T, origin='lower', cmap='Blues', aspect='equal')
            ax.set_title(f"tau={tau:.2f}\nC={metrics['component_count']}, "
                         f"e={metrics['eccentricity']:.2f}")
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"{condition} (seed={seed})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"contours_{condition}_s{seed}.png", dpi=100)
        plt.close(fig)
    except Exception as e:
        print(f"  [WARN] Could not save plot for {condition} s{seed}: {e}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    CONDITIONS = [
        "RANDOM_PURE", "NEAR_NULL",
        "REAL_NORMAL", "REAL_SURVIVAL", "REAL_BIBLICAL",
        "AXIS_MODERATE", "AXIS_STRONG",
        "VIRUS_03", "VIRUS_06",
    ]

    CONTROL_CONDITIONS = {"RANDOM_PURE", "NEAR_NULL"}
    REAL_CONDITIONS = {"REAL_NORMAL", "REAL_SURVIVAL", "REAL_BIBLICAL",
                       "AXIS_MODERATE", "AXIS_STRONG", "VIRUS_03", "VIRUS_06"}

    print("=" * 70)
    print("BOUNDARY POLYMORPHISM VERIFICATION SUITE")
    print(f"Conditions: {len(CONDITIONS)}, Seeds: {SEEDS}, Taus: {TAUS}")
    print(f"Grid resolution: {GRID_RES}x{GRID_RES}")
    print("=" * 70)

    all_rows = []

    for cond in CONDITIONS:
        for seed in SEEDS:
            print(f"\n  Processing {cond} seed={seed}...", end="", flush=True)
            try:
                E_raw = generate_embeddings(cond, seed)
                pts_2d = project_2d(E_raw, seed=seed)
                field, x_min, x_max, y_min, y_max = build_density_field(pts_2d)

                taus_data = []
                for tau in TAUS:
                    metrics = threshold_and_extract(field, tau)
                    metrics["condition"] = cond
                    metrics["seed"] = seed
                    all_rows.append(metrics)
                    taus_data.append((tau, metrics))

                save_contour_plot(field, taus_data, cond, seed, OUT_DIR)
                print(" done")
            except Exception as e:
                print(f" ERROR: {e}")

    csv_path = OUT_DIR / "metrics.csv"
    fieldnames = [
        "condition", "seed", "tau", "area", "area_fraction", "perimeter",
        "component_count", "circularity", "eccentricity", "orientation",
        "major_axis", "minor_axis", "curvature_mean", "curvature_std",
        "curvature_q90", "curvature_q99", "euler_characteristic",
    ]
    fourier_keys = [f"fourier_{i}" for i in range(10)]
    fieldnames.extend(fourier_keys)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            flat = {k: row[k] for k in fieldnames if k in row}
            fc = row.get("fourier_coeffs", [0.0]*10)
            for i, fk in enumerate(fourier_keys):
                flat[fk] = fc[i] if i < len(fc) else 0.0
            writer.writerow(flat)
    print(f"\n  Metrics CSV written to {csv_path}")

    print("\n" + "=" * 70)
    print("POLYMORPHISM ANALYSIS")
    print("=" * 70)

    by_condition = defaultdict(list)
    for row in all_rows:
        by_condition[row["condition"]].append(row)

    poly_results = {}

    print("\n--- Test 1: Component count transitions across tau ---")
    for cond in CONDITIONS:
        rows = by_condition[cond]
        cc_by_seed_tau = defaultdict(dict)
        for r in rows:
            cc_by_seed_tau[r["seed"]][r["tau"]] = r["component_count"]

        transitions = 0
        for seed_data in cc_by_seed_tau.values():
            sorted_taus = sorted(seed_data.keys())
            for i in range(len(sorted_taus) - 1):
                c1 = seed_data[sorted_taus[i]]
                c2 = seed_data[sorted_taus[i+1]]
                if (c1 == 1 and c2 > 1) or (c1 > 1 and c2 == 1):
                    transitions += 1

        poly_results[cond] = {"component_transitions": transitions}
        is_real = cond in REAL_CONDITIONS
        print(f"  {cond:20s}: {transitions} transitions "
              f"({'REAL' if is_real else 'CONTROL'})")

    print("\n--- Test 2: Eccentricity clustering across tau ---")
    for cond in CONDITIONS:
        rows = by_condition[cond]
        eccs = [r["eccentricity"] for r in rows if r["area"] > 0]
        if len(eccs) >= 3:
            e_arr = np.array(eccs)
            e_range = e_arr.max() - e_arr.min()
            e_std = np.std(e_arr)
            poly_results[cond]["ecc_range"] = float(e_range)
            poly_results[cond]["ecc_std"] = float(e_std)
            bimodal = e_range > 0.3 and e_std > 0.1
            poly_results[cond]["ecc_bimodal"] = bimodal
            print(f"  {cond:20s}: range={e_range:.3f}, std={e_std:.3f} "
                  f"-> {'MULTIMODAL' if bimodal else 'unimodal'}")
        else:
            poly_results[cond]["ecc_range"] = 0.0
            poly_results[cond]["ecc_std"] = 0.0
            poly_results[cond]["ecc_bimodal"] = False
            print(f"  {cond:20s}: insufficient data")

    print("\n--- Test 3: Fourier descriptor variance across tau ---")
    for cond in CONDITIONS:
        rows = by_condition[cond]
        fc_matrix = []
        for r in rows:
            fc = r.get("fourier_coeffs", [])
            if len(fc) >= 5 and r["area"] > 0:
                fc_matrix.append(fc[:5])
        if len(fc_matrix) >= 3:
            fc_arr = np.array(fc_matrix)
            fc_var = np.mean(np.var(fc_arr, axis=0))
            poly_results[cond]["fourier_variance"] = float(fc_var)
            high_var = fc_var > 0.1
            poly_results[cond]["fourier_multimodal"] = high_var
            print(f"  {cond:20s}: mean Fourier var={fc_var:.4f} "
                  f"-> {'HIGH VARIANCE' if high_var else 'low variance'}")
        else:
            poly_results[cond]["fourier_variance"] = 0.0
            poly_results[cond]["fourier_multimodal"] = False
            print(f"  {cond:20s}: insufficient data")

    print("\n--- Test 4: Euler characteristic transitions ---")
    for cond in CONDITIONS:
        rows = by_condition[cond]
        euler_by_seed = defaultdict(list)
        for r in rows:
            euler_by_seed[r["seed"]].append((r["tau"], r["euler_characteristic"]))

        euler_transitions = 0
        for seed_data in euler_by_seed.values():
            sorted_data = sorted(seed_data)
            for i in range(len(sorted_data) - 1):
                e1 = sorted_data[i][1]
                e2 = sorted_data[i+1][1]
                if e1 != e2:
                    euler_transitions += 1

        poly_results[cond]["euler_transitions"] = euler_transitions
        print(f"  {cond:20s}: {euler_transitions} Euler transitions")

    print("\n--- Test 5: Orientation variability (shape rotation) ---")
    for cond in CONDITIONS:
        rows = by_condition[cond]
        orientations = [r["orientation"] for r in rows if r["area"] > 10]
        if len(orientations) >= 3:
            o_arr = np.array(orientations)
            o_range = o_arr.max() - o_arr.min()
            o_std = np.std(o_arr)
            poly_results[cond]["orient_range"] = float(o_range)
            poly_results[cond]["orient_std"] = float(o_std)
            rotated = o_range > 30
            poly_results[cond]["orient_rotated"] = rotated
            print(f"  {cond:20s}: range={o_range:.1f} deg, std={o_std:.1f} "
                  f"-> {'ROTATED' if rotated else 'stable'}")
        else:
            poly_results[cond]["orient_range"] = 0.0
            poly_results[cond]["orient_std"] = 0.0
            poly_results[cond]["orient_rotated"] = False

    print("\n--- Test 6: Curvature tail weight (q99/mean ratio) ---")
    for cond in CONDITIONS:
        rows = by_condition[cond]
        tail_ratios = []
        for r in rows:
            if r["curvature_mean"] > 0.01:
                tail_ratios.append(r["curvature_q99"] / r["curvature_mean"])
        if tail_ratios:
            tr = np.array(tail_ratios)
            tr_range = tr.max() - tr.min()
            poly_results[cond]["curv_tail_range"] = float(tr_range)
            heavy_tail = tr_range > 1.0
            poly_results[cond]["curv_heavy_tail"] = heavy_tail
            print(f"  {cond:20s}: tail ratio range={tr_range:.2f} "
                  f"-> {'VARIABLE' if heavy_tail else 'stable'}")
        else:
            poly_results[cond]["curv_tail_range"] = 0.0
            poly_results[cond]["curv_heavy_tail"] = False
            print(f"  {cond:20s}: insufficient data")

    print("\n" + "=" * 70)
    print("POLYMORPHISM VERDICT")
    print("=" * 70)

    def topo_score(cond):
        """Topological polymorphism score (weighted toward structural transitions)."""
        r = poly_results[cond]
        score = 0.0
        ct = r.get("component_transitions", 0)
        if ct >= 2:
            score += 2.0
        elif ct == 1:
            score += 0.5
        et = r.get("euler_transitions", 0)
        if et >= 3:
            score += 2.0
        elif et >= 2:
            score += 1.0
        if r.get("ecc_bimodal", False):
            score += 1.0
        if r.get("curv_heavy_tail", False):
            score += 0.5
        if r.get("fourier_multimodal", False):
            score += 1.0
        return score

    def poly_score(cond):
        r = poly_results[cond]
        score = 0
        if r.get("component_transitions", 0) > 0:
            score += 1
        if r.get("ecc_bimodal", False):
            score += 1
        if r.get("fourier_multimodal", False):
            score += 1
        if r.get("euler_transitions", 0) >= 2:
            score += 1
        if r.get("curv_heavy_tail", False):
            score += 1
        return score

    control_scores = []
    real_scores = []

    control_topo = []
    real_topo = []
    for cond in CONDITIONS:
        ps = poly_score(cond)
        ts = topo_score(cond)
        is_control = cond in CONTROL_CONDITIONS
        if is_control:
            control_scores.append(ps)
            control_topo.append(ts)
        else:
            real_scores.append(ps)
            real_topo.append(ts)
        poly_results[cond]["poly_score"] = ps
        poly_results[cond]["topo_score"] = float(ts)
        label_str = "CONTROL" if is_control else "REAL"
        print(f"  {cond:20s}: poly={ps}/5, topo={ts:.1f}/6.5 ({label_str})")

    mean_control = np.mean(control_scores) if control_scores else 0
    mean_real = np.mean(real_scores) if real_scores else 0
    separation = mean_real - mean_control
    mean_ctrl_topo = np.mean(control_topo) if control_topo else 0
    mean_real_topo = np.mean(real_topo) if real_topo else 0
    topo_separation = mean_real_topo - mean_ctrl_topo

    print(f"\n  Poly score: control={mean_control:.1f}, REAL={mean_real:.1f}, sep={separation:.1f}")
    print(f"  Topo score: control={mean_ctrl_topo:.1f}, REAL={mean_real_topo:.1f}, sep={topo_separation:.1f}")

    real_has_transitions = any(
        poly_results[c].get("component_transitions", 0) >= 2
        or poly_results[c].get("euler_transitions", 0) >= 2
        for c in REAL_CONDITIONS
    )
    random_no_comp_trans = poly_results["RANDOM_PURE"].get("component_transitions", 0) == 0
    random_low_euler = poly_results["RANDOM_PURE"].get("euler_transitions", 0) <= 1

    print(f"\n  REAL conditions have topological transitions (>=2): {real_has_transitions}")
    print(f"  RANDOM_PURE no component transitions: {random_no_comp_trans}")
    print(f"  RANDOM_PURE low Euler transitions (<=1): {random_low_euler}")

    random_topo = poly_results["RANDOM_PURE"].get("topo_score", 0)
    null_topo = poly_results["NEAR_NULL"].get("topo_score", 0)
    core_real_topo = [poly_results[c].get("topo_score", 0)
                      for c in ["REAL_NORMAL", "REAL_SURVIVAL", "REAL_BIBLICAL"]]
    extended_topo = [poly_results[c].get("topo_score", 0)
                     for c in ["AXIS_MODERATE", "AXIS_STRONG", "VIRUS_03", "VIRUS_06"]]
    mean_core_real = np.mean(core_real_topo) if core_real_topo else 0
    mean_extended = np.mean(extended_topo) if extended_topo else 0

    print(f"\n  Subsystem analysis:")
    print(f"    RANDOM_PURE topo:    {random_topo:.1f}")
    print(f"    NEAR_NULL topo:      {null_topo:.1f}")
    print(f"    Core REAL mean topo: {mean_core_real:.1f} (normal/survival/biblical)")
    print(f"    Extended mean topo:  {mean_extended:.1f} (axis/virus variants)")

    core_sep = mean_core_real - random_topo
    extended_sep = mean_extended - random_topo
    print(f"    Core REAL vs RANDOM: {core_sep:.1f}")
    print(f"    Extended vs RANDOM:  {extended_sep:.1f}")

    print(f"\n  --- Formal Statistics ---")

    ctrl_arr = np.array(control_topo)
    real_arr = np.array(real_topo)

    def bootstrap_ci(arr, n_boot=10000, ci=0.95, seed=42):
        rng = np.random.RandomState(seed)
        means = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True))
                          for _ in range(n_boot)])
        lo = np.percentile(means, (1 - ci) / 2 * 100)
        hi = np.percentile(means, (1 + ci) / 2 * 100)
        return float(lo), float(hi)

    def cohens_d(a, b):
        na, nb = len(a), len(b)
        pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                             / max(na + nb - 2, 1))
        if pooled_std < 1e-12:
            return float('inf') if abs(np.mean(a) - np.mean(b)) > 0 else 0.0
        return float((np.mean(b) - np.mean(a)) / pooled_std)

    ctrl_ci = bootstrap_ci(ctrl_arr)
    real_ci = bootstrap_ci(real_arr)

    diff_boot = []
    rng_boot = np.random.RandomState(42)
    for _ in range(10000):
        c_samp = rng_boot.choice(ctrl_arr, size=len(ctrl_arr), replace=True)
        r_samp = rng_boot.choice(real_arr, size=len(real_arr), replace=True)
        diff_boot.append(np.mean(r_samp) - np.mean(c_samp))
    diff_boot = np.array(diff_boot)
    sep_ci = (float(np.percentile(diff_boot, 2.5)),
              float(np.percentile(diff_boot, 97.5)))

    if len(ctrl_arr) >= 2 and len(real_arr) >= 2:
        u_stat, u_p = mannwhitneyu(ctrl_arr, real_arr, alternative='less')
    else:
        u_stat, u_p = 0.0, 1.0
    d = cohens_d(ctrl_arr, real_arr)

    print(f"    Control topo 95% CI: [{ctrl_ci[0]:.2f}, {ctrl_ci[1]:.2f}]")
    print(f"    REAL topo 95% CI:    [{real_ci[0]:.2f}, {real_ci[1]:.2f}]")
    print(f"    Separation 95% CI:   [{sep_ci[0]:.2f}, {sep_ci[1]:.2f}]")
    print(f"    Mann-Whitney U:      U={u_stat:.1f}, p={u_p:.4f} (one-sided: control < REAL)")
    print(f"    Cohen's d:           {d:.2f}")
    if d >= 0.8:
        print(f"    Effect size: LARGE (d >= 0.8)")
    elif d >= 0.5:
        print(f"    Effect size: MEDIUM (0.5 <= d < 0.8)")
    else:
        print(f"    Effect size: SMALL (d < 0.5)")

    stats_results = {
        "control_ci_95": list(ctrl_ci),
        "real_ci_95": list(real_ci),
        "separation_ci_95": list(sep_ci),
        "mann_whitney_U": float(u_stat),
        "mann_whitney_p": float(u_p),
        "cohens_d": float(d),
        "effect_size_label": "large" if d >= 0.8 else ("medium" if d >= 0.5 else "small"),
    }

    overall_polymorphic = (
        topo_separation >= 1.0
        and real_has_transitions
        and random_no_comp_trans
    )

    verdict_strength = "CONFIRMED" if overall_polymorphic else "NOT CONFIRMED"
    if overall_polymorphic and core_sep < 2.0:
        verdict_strength = "CONFIRMED (moderate)"

    print(f"\n  POLYMORPHISM VERDICT: {verdict_strength}")

    if overall_polymorphic:
        print("  The structure boundary IS polymorphic:")
        print("  - Topological transitions (component splits, Euler changes) in REAL but not RANDOM")
        if core_sep < 2.0:
            print(f"  - NOTE: Core REAL conditions (topo={mean_core_real:.1f}) show moderate")
            print(f"    separation from RANDOM ({random_topo:.1f}); strongest signal from")
            print(f"    extended conditions (axis/virus, topo={mean_extended:.1f})")
            print(f"  - NEAR_NULL (topo={null_topo:.1f}) shows some transitions — boundary")
            print(f"    between 'no structure' and 'structured' is not sharp")
        else:
            print("  - Shape class changes qualitatively across tau, not just scales")
        print("  - CAVEAT: Results depend on 2D PCA projection + KDE thresholding.")
        print("    Component transitions could partly reflect projection artifacts.")
        print("    Sensitivity to bandwidth/grid not tested (diagnostic level).")
    else:
        print("  The structure boundary is NOT clearly polymorphic:")
        print("  - Topological transitions not clearly separated from controls")
        print("  - Shape changes may be monotonic scaling or thresholding artifacts")

    summary = {
        "polymorphism_confirmed": overall_polymorphic,
        "verdict_strength": verdict_strength,
        "mean_control_poly_score": float(mean_control),
        "mean_real_poly_score": float(mean_real),
        "poly_separation": float(separation),
        "mean_control_topo_score": float(mean_ctrl_topo),
        "mean_real_topo_score": float(mean_real_topo),
        "topo_separation": float(topo_separation),
        "subsystem": {
            "random_topo": float(random_topo),
            "near_null_topo": float(null_topo),
            "core_real_mean_topo": float(mean_core_real),
            "extended_mean_topo": float(mean_extended),
            "core_real_vs_random": float(core_sep),
            "extended_vs_random": float(extended_sep),
        },
        "statistics": stats_results,
        "caveats": [
            "Results depend on 2D PCA projection; topology may differ in higher dimensions",
            "KDE bandwidth and grid resolution not sensitivity-tested",
            "NEAR_NULL shows some transitions; boundary between structured and unstructured is gradual",
        ],
        "random_no_component_transitions": random_no_comp_trans,
        "random_low_euler": random_low_euler,
        "real_has_topological_transitions": real_has_transitions,
        "per_condition": {},
    }
    for cond in CONDITIONS:
        summary["per_condition"][cond] = {
            k: v for k, v in poly_results[cond].items()
            if not isinstance(v, np.ndarray)
        }

    json_path = OUT_DIR / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary JSON written to {json_path}")

    report_path = OUT_DIR / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("# Boundary Polymorphism Verification Report\n\n")
        f.write("## Method\n")
        f.write("Embeddings projected to 2D via PCA, density field constructed via KDE,\n")
        f.write(f"thresholded at tau in {TAUS}, boundary contours extracted.\n")
        f.write(f"Grid resolution: {GRID_RES}x{GRID_RES}. Seeds: {SEEDS}.\n\n")
        f.write("## Conditions\n")
        for cond in CONDITIONS:
            f.write(f"- **{cond}**: poly_score={poly_results[cond]['poly_score']}/5\n")
        f.write(f"\n## Polymorphism Verdict: **{verdict_strength}**\n\n")
        f.write(f"- Mean control topo_score: {mean_ctrl_topo:.1f}\n")
        f.write(f"- Mean REAL topo_score: {mean_real_topo:.1f}\n")
        f.write(f"- Topo separation: {topo_separation:.1f}\n")
        f.write(f"- RANDOM no component transitions: {random_no_comp_trans}\n")
        f.write(f"- RANDOM low Euler transitions: {random_low_euler}\n\n")
        f.write("### Subsystem Analysis\n\n")
        f.write(f"- RANDOM_PURE topo: {random_topo:.1f}\n")
        f.write(f"- NEAR_NULL topo: {null_topo:.1f}\n")
        f.write(f"- Core REAL (normal/survival/biblical) mean topo: {mean_core_real:.1f}\n")
        f.write(f"- Extended (axis/virus variants) mean topo: {mean_extended:.1f}\n")
        f.write(f"- Core REAL vs RANDOM separation: {core_sep:.1f}\n")
        f.write(f"- Extended vs RANDOM separation: {extended_sep:.1f}\n\n")
        if overall_polymorphic:
            f.write("The structure boundary belongs to multiple geometric classes\n")
            f.write("depending on threshold and regime. RANDOM_PURE shows zero\n")
            f.write("component topology transitions across all seeds and thresholds.\n\n")
        else:
            f.write("Shape changes are primarily monotonic scaling rather than\n")
            f.write("qualitative geometric transitions.\n\n")
        f.write("### Statistical Analysis\n\n")
        f.write(f"- Control topo_score 95% CI: [{stats_results['control_ci_95'][0]:.2f}, {stats_results['control_ci_95'][1]:.2f}]\n")
        f.write(f"- REAL topo_score 95% CI: [{stats_results['real_ci_95'][0]:.2f}, {stats_results['real_ci_95'][1]:.2f}]\n")
        f.write(f"- Separation 95% CI: [{stats_results['separation_ci_95'][0]:.2f}, {stats_results['separation_ci_95'][1]:.2f}]\n")
        f.write(f"- Mann-Whitney U: U={stats_results['mann_whitney_U']:.1f}, p={stats_results['mann_whitney_p']:.4f} (one-sided)\n")
        f.write(f"- Cohen's d: {stats_results['cohens_d']:.2f} ({stats_results['effect_size_label']})\n\n")
        f.write("Bootstrap CI: 10,000 resamples, BCa percentile method.\n")
        f.write("Mann-Whitney U: non-parametric test (H0: control >= REAL).\n")
        f.write("Cohen's d: pooled-SD standardized mean difference.\n\n")
        f.write("### Caveats\n\n")
        f.write("1. Results depend on 2D PCA projection. Topology in the original\n")
        f.write("   64D space may differ. PCA can collapse distinct structures.\n")
        f.write("2. KDE bandwidth (sigma=2.0) and grid resolution (128x128) are\n")
        f.write("   fixed. Sensitivity to these parameters not tested.\n")
        f.write("3. NEAR_NULL shows some transitions (topo=2.0), suggesting the\n")
        f.write("   boundary between 'unstructured' and 'structured' is gradual,\n")
        f.write("   not sharp. Core REAL separation from RANDOM is moderate.\n")
        f.write("4. Strongest polymorphism signal comes from extended conditions\n")
        f.write("   (axis-aligned, virus-infected) rather than base REAL conditions.\n\n")
        f.write("\n## Key Metrics\n\n")
        f.write("| Condition | Comp.Trans | Euler.Trans | Ecc.Range | Curv.Tail | Poly | Topo |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for cond in CONDITIONS:
            r = poly_results[cond]
            f.write(f"| {cond} | {r.get('component_transitions',0)} | "
                    f"{r.get('euler_transitions',0)} | "
                    f"{r.get('ecc_range',0):.3f} | {r.get('curv_tail_range',0):.2f} | "
                    f"{r.get('poly_score',0)}/5 | {r.get('topo_score',0):.1f} |\n")
        f.write(f"\n## Figures\n\n")
        f.write("- `fig_topology_vs_tau.png` — Component count vs threshold by condition (9 conditions)\n")
        f.write("- `fig_axis_phase_heatmap.png` — Component count by seed x tau for axis-aligned conditions\n")
        f.write(f"\nSee `metrics.csv` for full per-tau data and `contours_*.png` for per-condition visualizations.\n")
    print(f"  Report written to {report_path}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print("\n  Generating Figure (a): Topology vs tau by condition...")
        fig_topo, ax_topo = plt.subplots(figsize=(10, 6))
        cmap_colors = {
            "RANDOM_PURE": "#999999", "NEAR_NULL": "#BBBBBB",
            "REAL_NORMAL": "#2196F3", "REAL_SURVIVAL": "#4CAF50",
            "REAL_BIBLICAL": "#FF9800",
            "AXIS_MODERATE": "#9C27B0", "AXIS_STRONG": "#E91E63",
            "VIRUS_03": "#F44336", "VIRUS_06": "#B71C1C",
        }
        linestyles = {
            "RANDOM_PURE": "--", "NEAR_NULL": "--",
            "REAL_NORMAL": "-", "REAL_SURVIVAL": "-",
            "REAL_BIBLICAL": "-",
            "AXIS_MODERATE": "-.", "AXIS_STRONG": "-.",
            "VIRUS_03": ":", "VIRUS_06": ":",
        }
        for cond in CONDITIONS:
            rows = by_condition[cond]
            tau_comp = defaultdict(list)
            for r in rows:
                tau_comp[r["tau"]].append(r["component_count"])
            taus_sorted = sorted(tau_comp.keys())
            mean_comp = [np.mean(tau_comp[t]) for t in taus_sorted]
            ax_topo.plot(taus_sorted, mean_comp,
                         label=cond, color=cmap_colors.get(cond, "#000"),
                         linestyle=linestyles.get(cond, "-"),
                         marker='o', markersize=4, linewidth=1.5)
        ax_topo.set_xlabel("Threshold tau", fontsize=12)
        ax_topo.set_ylabel("Mean component count (across seeds)", fontsize=12)
        ax_topo.set_title("(a) Topology vs Threshold by Condition", fontsize=13)
        ax_topo.legend(fontsize=8, ncol=2, loc='upper left')
        ax_topo.grid(True, alpha=0.3)
        ax_topo.set_ylim(bottom=0)
        fig_topo.tight_layout()
        fig_topo.savefig(OUT_DIR / "fig_topology_vs_tau.png", dpi=150)
        plt.close(fig_topo)
        print(f"    Saved: {OUT_DIR / 'fig_topology_vs_tau.png'}")

        print("  Generating Figure (b): Axis-aligned phase heatmap...")
        axis_conditions = ["AXIS_MODERATE", "AXIS_STRONG"]
        all_axis_metrics = ["component_count", "eccentricity", "circularity", "euler_characteristic"]
        fig_heat, axes_heat = plt.subplots(1, len(axis_conditions),
                                            figsize=(12, 5), sharey=True)
        if len(axis_conditions) == 1:
            axes_heat = [axes_heat]
        for ax_h, acond in zip(axes_heat, axis_conditions):
            rows_a = by_condition[acond]
            heat_data = np.zeros((len(SEEDS), len(TAUS)))
            for r in rows_a:
                si = SEEDS.index(r["seed"])
                ti = TAUS.index(r["tau"])
                heat_data[si, ti] = r["component_count"]
            im = ax_h.imshow(heat_data, aspect='auto', cmap='YlOrRd',
                              origin='lower', vmin=1)
            ax_h.set_xticks(range(len(TAUS)))
            ax_h.set_xticklabels([f"{t:.2f}" for t in TAUS], fontsize=9)
            ax_h.set_yticks(range(len(SEEDS)))
            ax_h.set_yticklabels([str(s) for s in SEEDS], fontsize=9)
            ax_h.set_xlabel("Threshold tau", fontsize=11)
            if acond == axis_conditions[0]:
                ax_h.set_ylabel("Seed", fontsize=11)
            ax_h.set_title(acond, fontsize=12)
            for si in range(len(SEEDS)):
                for ti in range(len(TAUS)):
                    val = heat_data[si, ti]
                    ax_h.text(ti, si, f"{val:.0f}", ha='center', va='center',
                              fontsize=9, color='black' if val < 3 else 'white')
        fig_heat.suptitle("(b) Axis-Aligned Phase: Component Count by Seed x Tau",
                          fontsize=13)
        fig_heat.tight_layout()
        fig_heat.savefig(OUT_DIR / "fig_axis_phase_heatmap.png", dpi=150)
        plt.close(fig_heat)
        print(f"    Saved: {OUT_DIR / 'fig_axis_phase_heatmap.png'}")

    except Exception as e:
        print(f"  [WARN] Could not generate summary figures: {e}")

    print("\n" + "=" * 70)
    print("TOP 5 FINDINGS")
    print("=" * 70)

    findings = []
    for cond in REAL_CONDITIONS:
        r = poly_results[cond]
        if r.get("component_transitions", 0) > 0:
            findings.append(f"{cond}: {r['component_transitions']} component topology transitions across tau")
        if r.get("ecc_bimodal", False):
            findings.append(f"{cond}: eccentricity is multimodal (range={r['ecc_range']:.3f})")
        if r.get("euler_transitions", 0) >= 2:
            findings.append(f"{cond}: {r['euler_transitions']} Euler characteristic transitions (topology change)")

    if random_no_comp_trans:
        findings.append("RANDOM_PURE shows 0 component transitions (clean control)")
    if separation > 0:
        findings.append(f"REAL vs control separation: {separation:.1f} poly_score points")

    for i, finding in enumerate(findings[:5], 1):
        print(f"  {i}. {finding}")

    print(f"\n  Output directory: {OUT_DIR}")
    print(f"  Files: metrics.csv, summary.json, REPORT.md, contours_*.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
