"""Geometry module: anchor pair, hub, magistrales via spherical k-means."""
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


def approximate_farthest_pair(E: np.ndarray, n_projections: int = 50, seed: int = 42) -> Tuple[int, int]:
    """Find approximate farthest pair using random projections. O(n * n_projections)."""
    rng = np.random.RandomState(seed)
    n, d = E.shape
    best_dist = -1.0
    best_pair = (0, 1)
    
    for _ in range(n_projections):
        direction = rng.randn(d)
        direction /= np.linalg.norm(direction) + 1e-12
        proj = E @ direction
        i_min = int(np.argmin(proj))
        i_max = int(np.argmax(proj))
        dist = np.linalg.norm(E[i_max] - E[i_min])
        if dist > best_dist:
            best_dist = dist
            best_pair = (i_min, i_max)
    
    logger.info(f"Farthest pair: indices {best_pair}, distance={best_dist:.4f}")
    return best_pair


def compute_hub(E: np.ndarray, anchor_pair: Tuple[int, int]) -> np.ndarray:
    """Compute hub as midpoint of anchor pair."""
    hub = (E[anchor_pair[0]] + E[anchor_pair[1]]) / 2.0
    return hub


def compute_unit_directions(E: np.ndarray, hub: np.ndarray) -> np.ndarray:
    """Compute unit direction vectors from hub."""
    diff = E - hub
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    u = diff / norms
    return u


def compute_radii(E: np.ndarray, hub: np.ndarray) -> np.ndarray:
    """Compute radial distances from hub."""
    diff = E - hub
    r = np.linalg.norm(diff, axis=1)
    return r


def spherical_kmeans(u: np.ndarray, k: int = 6, seed: int = 42, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Spherical k-means clustering on unit vectors (cosine distance)."""
    try:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=max_iter)
        labels = km.fit_predict(u)
        centers = km.cluster_centers_
        center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.maximum(center_norms, 1e-12)
        return labels, centers
    except Exception as e:
        logger.warning(f"Spherical k-means failed: {e}. Falling back to PCA-based binning.")
        return pca_binning(u, k, seed)


def pca_binning(u: np.ndarray, k: int = 6, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback: PCA-based directional binning into k sectors."""
    n_components = min(3, u.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    u_pca = pca.fit_transform(u)
    
    angles = np.arctan2(u_pca[:, 1] if n_components > 1 else np.zeros(len(u)),
                        u_pca[:, 0])
    bin_edges = np.linspace(-np.pi, np.pi, k + 1)
    labels = np.digitize(angles, bin_edges) - 1
    labels = np.clip(labels, 0, k - 1)
    
    centers = np.zeros((k, u.shape[1]))
    for i in range(k):
        mask = labels == i
        if mask.sum() > 0:
            c = u[mask].mean(axis=0)
            norm = np.linalg.norm(c)
            if norm > 1e-12:
                centers[i] = c / norm
    
    return labels, centers


def build_geometry(E: np.ndarray, n_magistrales: int = 6, seed: int = 42,
                   anchor_pair: Tuple[int, int] = None) -> Dict[str, Any]:
    """Full geometry pipeline: anchor, hub, directions, magistrales."""
    if anchor_pair is None:
        anchor_pair = approximate_farthest_pair(E, seed=seed)
    
    hub = compute_hub(E, anchor_pair)
    u = compute_unit_directions(E, hub)
    r = compute_radii(E, hub)
    labels, centers = spherical_kmeans(u, k=n_magistrales, seed=seed)
    
    mag_counts = [int((labels == i).sum()) for i in range(n_magistrales)]
    logger.info(f"Magistrale counts: {mag_counts}")
    
    return {
        "anchor_pair": anchor_pair,
        "hub": hub,
        "u": u,
        "r": r,
        "labels": labels,
        "centers": centers,
        "mag_counts": mag_counts,
    }
