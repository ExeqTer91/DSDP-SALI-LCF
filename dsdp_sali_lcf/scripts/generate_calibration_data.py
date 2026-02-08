"""Generate calibration datasets for the 5 conditions.

RANDOM_PURE: Gaussian noise, no structure
NEAR_NULL: Random + epsilon weak identity axis
REAL_NORMAL: sqrt(phi) lattice alignment (iterated hub-snapping) + temporal phasing
REAL_SURVIVAL: Derived from REAL_NORMAL with contraction + tightening
REAL_BIBLICAL: Maximally constrained canonical regime — preserved identity,
               uniform lattice snap (no temporal dynamics), rigid conservation
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

PHI = 1.6180339887498949
SQRT_PHI = np.sqrt(PHI)
LOG_STEP = np.log(SQRT_PHI)


def generate_condition(condition: str, n: int = 5000, d: int = 64,
                       seed: int = 42) -> np.ndarray:
    if condition == "RANDOM_PURE":
        return _gen_random_pure(n, d, seed)
    elif condition == "NEAR_NULL":
        return _gen_near_null(n, d, seed)
    elif condition == "REAL_NORMAL":
        return _gen_real_normal(n, d, seed)
    elif condition == "REAL_SURVIVAL":
        return _gen_real_survival(n, d, seed)
    elif condition == "REAL_BIBLICAL":
        return _gen_real_biblical(n, d, seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")


def _gen_random_pure(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, d) * 0.5


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


def _gen_near_null(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    k = 6
    dirs = _make_magistrale_directions(d, k, rng)
    E = rng.randn(n, d) * 0.5
    center = np.zeros(d)
    radii = np.linalg.norm(E - center, axis=1)
    log_step = np.log(SQRT_PHI)
    log_r = np.log(radii + 1e-12)
    nearest_band = np.round(log_r / log_step) * log_step
    target_r = np.exp(nearest_band)
    directions = E - center
    dir_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    dir_norms = np.maximum(dir_norms, 1e-12)
    unit_dirs = directions / dir_norms
    epsilon = 0.05
    new_radii = radii * (1 - epsilon) + target_r * epsilon
    cluster_assign = rng.randint(0, k, size=n)
    angular_pull = 0.05
    for i in range(n):
        unit_dirs[i] = unit_dirs[i] * (1 - angular_pull) + dirs[cluster_assign[i]] * angular_pull
        norm = np.linalg.norm(unit_dirs[i])
        if norm > 1e-12:
            unit_dirs[i] /= norm
    E_out = center + unit_dirs * new_radii[:, np.newaxis]
    E_out += rng.randn(n, d) * 0.02
    return E_out


def _find_hub(E: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(E, axis=1)
    i_max = np.argmax(norms)
    dists = np.linalg.norm(E - E[i_max], axis=1)
    j_max = np.argmax(dists)
    return (E[i_max] + E[j_max]) / 2.0


def _snap_radii_from_hub(E: np.ndarray, hub: np.ndarray,
                          t_frac: np.ndarray, rng,
                          snap_strength_base: float = 0.85,
                          temporal_phase: bool = True) -> np.ndarray:
    diff = E - hub
    r = np.linalg.norm(diff, axis=1)
    r_safe = np.maximum(r, 1e-12)
    unit_dirs = diff / r_safe[:, np.newaxis]
    y = np.log(r_safe)
    nearest_band_y = np.round(y / LOG_STEP) * LOG_STEP
    if temporal_phase:
        entropy_phase = np.where(t_frac < 0.30, 1.0,
                        np.where(t_frac < 0.50, 1.0 - (t_frac - 0.30) / 0.20, 0.0))
        wave_phase = np.where(t_frac < 0.40, 0.0,
                     np.where(t_frac < 0.65, (t_frac - 0.40) / 0.25, 1.0))
        snap = 0.30 + 0.65 * wave_phase
        jitter = rng.randn(len(E)) * LOG_STEP * 0.35 * entropy_phase
    else:
        snap = np.full(len(E), snap_strength_base)
        jitter = rng.randn(len(E)) * LOG_STEP * 0.05
    y_target = nearest_band_y + jitter
    y_new = y * (1.0 - snap) + y_target * snap
    new_r = np.exp(y_new)
    E_new = hub + unit_dirs * new_r[:, np.newaxis]
    return E_new


def _gen_real_normal_with_meta(n: int, d: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    from dsdp_sali_lcf.src.data_loader import center_normalize
    E_norm, _, _ = center_normalize(E_raw)
    t_frac = np.linspace(0, 1, n)
    rng_snap = np.random.RandomState(seed + 500)
    for iteration in range(3):
        hub = _find_hub(E_norm)
        E_norm = _snap_radii_from_hub(E_norm, hub, t_frac, rng_snap,
                                        temporal_phase=True)
        E_norm = E_norm - E_norm.mean(axis=0)
        E_norm = E_norm / (np.std(E_norm) + 1e-12)
        rng_snap = np.random.RandomState(seed + 500)
    return E_norm, dirs, cluster_assign


def _gen_real_normal(n: int, d: int, seed: int) -> np.ndarray:
    E, _, _ = _gen_real_normal_with_meta(n, d, seed)
    return E


def _gen_real_survival(n: int, d: int, seed: int) -> np.ndarray:
    E_real, dirs, cluster_assign = _gen_real_normal_with_meta(n, d, seed)
    rng = np.random.RandomState(seed + 999)
    center = E_real.mean(axis=0)
    diff = E_real - center
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-12)
    unit_dirs_actual = diff / norms_safe
    norms_flat = norms.flatten()
    mean_r = norms_flat.mean()
    contraction = 0.95
    new_norms = norms_flat * contraction + mean_r * (1 - contraction)
    tighten = 0.15
    for i in range(len(E_real)):
        c_dir = dirs[cluster_assign[i]]
        unit_dirs_actual[i] = unit_dirs_actual[i] * (1 - tighten) + c_dir * tighten
        norm = np.linalg.norm(unit_dirs_actual[i])
        if norm > 1e-12:
            unit_dirs_actual[i] /= norm
    E_survival = center + unit_dirs_actual * new_norms[:, np.newaxis]
    E_survival += rng.randn(len(E_real), E_real.shape[1]) * 0.003
    return E_survival


def _gen_real_biblical(n: int, d: int, seed: int) -> np.ndarray:
    E_real, dirs, cluster_assign = _gen_real_normal_with_meta(n, d, seed)
    rng = np.random.RandomState(seed + 777)
    perm = rng.permutation(n)
    E_biblical = E_real[perm]
    return E_biblical


ALL_CONDITIONS = ["RANDOM_PURE", "NEAR_NULL", "REAL_NORMAL", "REAL_SURVIVAL",
                  "REAL_BIBLICAL"]


def generate_all(n: int = 5000, d: int = 64, seed: int = 42,
                 save_dir: str = None) -> Dict[str, np.ndarray]:
    datasets = {}
    for cond in ALL_CONDITIONS:
        datasets[cond] = generate_condition(cond, n, d, seed)

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for cond, E in datasets.items():
            path = out / f"{cond.lower()}_embeddings.npz"
            np.savez(path, embeddings=E)
            print(f"Saved {cond}: {path} shape={E.shape} "
                  f"mean_norm={np.mean(np.linalg.norm(E, axis=1)):.4f}")

    return datasets


if __name__ == "__main__":
    generate_all(save_dir="dsdp_sali_lcf/data/calibration")
