"""Identity vector metrics for coupling analysis with label alignment."""
import numpy as np
from typing import Dict, Any
from scipy.optimize import linear_sum_assignment


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def align_centers(centers_a: np.ndarray,
                   centers_b: np.ndarray) -> np.ndarray:
    k = centers_a.shape[0]
    cost_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cost_matrix[i, j] = 1.0 - cosine_similarity(centers_a[i], centers_b[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return centers_b[col_ind]


def compute_identity_similarity(centers_a: np.ndarray,
                                  centers_b: np.ndarray) -> float:
    centers_b_aligned = align_centers(centers_a, centers_b)
    sims = []
    for i in range(centers_a.shape[0]):
        sims.append(cosine_similarity(centers_a[i], centers_b_aligned[i]))
    return float(np.mean(sims))


def compute_identity_vector(centers: np.ndarray) -> np.ndarray:
    flat = centers.flatten()
    norm = np.linalg.norm(flat)
    if norm < 1e-12:
        return flat
    return flat / norm


def compare_identities(centers_a: np.ndarray,
                        centers_b: np.ndarray) -> Dict[str, Any]:
    aligned_sim = compute_identity_similarity(centers_a, centers_b)
    return {
        "identity_similarity": aligned_sim,
        "identity_distance": 1.0 - aligned_sim,
    }


def compute_coupling_strength(alignment: float, wave_persistence: int,
                                fsi: float) -> float:
    alignment_norm = min(1.0, max(0.0, alignment))
    wave_norm = min(1.0, wave_persistence / 10.0)
    fsi_norm = min(1.0, max(0.0, fsi))
    return float(0.4 * alignment_norm + 0.3 * wave_norm + 0.3 * fsi_norm)
