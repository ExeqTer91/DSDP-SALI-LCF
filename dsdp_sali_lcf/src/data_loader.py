"""Data loading and validation for embedding files."""
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_embeddings(path: str, max_points: Optional[int] = None, seed: int = 42) -> Dict[str, Any]:
    """Load embeddings from .npz, .npy, or .csv file.
    
    Returns dict with keys: 'E' (embeddings), 't' (time index), 'meta' (diagnostics).
    """
    p = Path(path)
    meta: Dict[str, Any] = {"source": str(p), "format": p.suffix}
    
    try:
        if p.suffix == ".npz":
            data = np.load(p, allow_pickle=False)
            keys = list(data.keys())
            meta["npz_keys"] = keys
            E = None
            t = None
            for k in keys:
                arr = data[k]
                if arr.ndim == 2 and E is None:
                    E = arr.astype(np.float64)
                    meta["embedding_key"] = k
                elif arr.ndim == 1 and t is None and k.lower() in ("t", "time", "index", "timestamps"):
                    t = arr.astype(np.float64)
                    meta["time_key"] = k
            if E is None:
                for k in keys:
                    arr = data[k]
                    if arr.ndim == 2:
                        E = arr.astype(np.float64)
                        meta["embedding_key"] = k
                        break
            if E is None:
                raise ValueError(f"No 2D array found in {p}. Keys: {keys}")
        elif p.suffix == ".npy":
            E = np.load(p, allow_pickle=False).astype(np.float64)
            t = None
            if E.ndim != 2:
                raise ValueError(f"Expected 2D array, got {E.ndim}D from {p}")
        elif p.suffix == ".csv":
            E = np.loadtxt(p, delimiter=",", dtype=np.float64)
            t = None
            if E.ndim != 2:
                raise ValueError(f"Expected 2D array from CSV, got {E.ndim}D")
        else:
            raise ValueError(f"Unsupported format: {p.suffix}. Use .npz, .npy, or .csv")
    except Exception as e:
        logger.error(f"Failed to load {p}: {e}")
        raise

    n, d = E.shape
    meta["original_shape"] = (n, d)
    logger.info(f"Loaded {p}: shape=({n}, {d})")

    if max_points is not None and n > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_points, replace=False)
        idx.sort()
        E = E[idx]
        if t is not None:
            t = t[idx]
        meta["subsampled"] = True
        meta["subsample_n"] = max_points
        logger.info(f"Subsampled to {max_points} points")
    else:
        meta["subsampled"] = False

    return {"E": E, "t": t, "meta": meta}


def validate_embeddings(E: np.ndarray) -> Dict[str, Any]:
    """Validate embeddings for NaNs, infinities, shape issues."""
    diag: Dict[str, Any] = {}
    n, d = E.shape
    diag["shape"] = (n, d)
    diag["dtype"] = str(E.dtype)
    
    nan_count = int(np.isnan(E).sum())
    inf_count = int(np.isinf(E).sum())
    diag["nan_count"] = nan_count
    diag["inf_count"] = inf_count
    
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values - replacing with 0")
        E[np.isnan(E)] = 0.0
    if inf_count > 0:
        logger.warning(f"Found {inf_count} Inf values - clipping")
        E = np.clip(E, -1e15, 1e15)
    
    diag["mean_norm"] = float(np.mean(np.linalg.norm(E, axis=1)))
    diag["std_norm"] = float(np.std(np.linalg.norm(E, axis=1)))
    diag["min_val"] = float(E.min())
    diag["max_val"] = float(E.max())
    
    return diag


def prepare_time_index(t: Optional[np.ndarray], n: int, remap_by_proxy: bool = False,
                       E: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Prepare time index. Create arange if missing. Optionally remap by scalar proxy."""
    meta: Dict[str, Any] = {}
    
    if t is not None:
        meta["t_source"] = "from_file"
        t = t.copy()
    else:
        t = np.arange(n, dtype=np.float64)
        meta["t_source"] = "synthetic_arange"
        logger.info("No time index found, using arange(n)")
    
    if remap_by_proxy and E is not None:
        rng = np.random.RandomState(0)
        anchor = rng.randn(E.shape[1])
        anchor /= np.linalg.norm(anchor) + 1e-12
        projections = E @ anchor
        sort_order = np.argsort(projections)
        t_remapped = np.arange(n, dtype=np.float64)
        meta["t_remap"] = "proxy_projection"
        meta["t_original_order"] = t.copy().tolist()[:20]
        return t_remapped, meta
    
    return t, meta


def center_normalize(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Center and normalize embeddings. Returns (E_normed, center, scale)."""
    center = E.mean(axis=0)
    E_c = E - center
    scale = np.std(E_c) + 1e-12
    E_c /= scale
    return E_c, center, float(scale)
