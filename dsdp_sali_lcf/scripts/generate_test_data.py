"""Generate synthetic test data (fresh + echo embeddings) for pipeline testing."""
import numpy as np
from pathlib import Path
import sys

def generate_structured_embeddings(n: int = 5000, d: int = 64, seed: int = 42,
                                    structure_level: float = 0.3) -> np.ndarray:
    """Generate embeddings with some latent radial structure.
    
    structure_level: 0 = pure noise, 1 = strong discrete radial bands
    """
    rng = np.random.RandomState(seed)
    
    E = rng.randn(n, d) * 0.5
    
    center = np.zeros(d)
    radii = np.linalg.norm(E - center, axis=1)
    
    PHI = 1.6180339887498949
    SQRT_PHI = np.sqrt(PHI)
    log_step = np.log(SQRT_PHI)
    
    log_r = np.log(radii + 1e-12)
    nearest_band = np.round(log_r / log_step) * log_step
    target_r = np.exp(nearest_band)
    
    directions = (E - center)
    dir_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    dir_norms = np.maximum(dir_norms, 1e-12)
    unit_dirs = directions / dir_norms
    
    new_radii = radii * (1 - structure_level) + target_r * structure_level
    E_structured = center + unit_dirs * new_radii[:, np.newaxis]
    
    E_structured += rng.randn(n, d) * 0.02
    
    return E_structured


def generate_echo(E_fresh: np.ndarray, echo_noise: float = 0.3, seed: int = 123) -> np.ndarray:
    """Generate echo embeddings: degraded copy of fresh (anti-contamination).
    
    echo should have LESS structure than fresh.
    """
    rng = np.random.RandomState(seed)
    
    noise = rng.randn(*E_fresh.shape) * echo_noise
    
    perm = rng.permutation(len(E_fresh))
    E_echo = E_fresh[perm] * (1 - echo_noise * 0.5) + noise
    
    return E_echo


def main():
    output_dir = Path("dsdp_sali_lcf/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n = 5000
    d = 64
    
    print(f"Generating synthetic FRESH embeddings: n={n}, d={d}")
    E_fresh = generate_structured_embeddings(n=n, d=d, seed=42, structure_level=0.3)
    
    print(f"Generating synthetic ECHO embeddings (degraded copy)")
    E_echo = generate_echo(E_fresh, echo_noise=0.3, seed=123)
    
    fresh_path = output_dir / "fresh_embeddings.npz"
    echo_path = output_dir / "echo_embeddings.npz"
    
    np.savez(fresh_path, embeddings=E_fresh)
    np.savez(echo_path, embeddings=E_echo)
    
    print(f"Saved: {fresh_path} shape={E_fresh.shape}")
    print(f"Saved: {echo_path} shape={E_echo.shape}")
    print(f"\nFresh stats: mean_norm={np.mean(np.linalg.norm(E_fresh, axis=1)):.4f}")
    print(f"Echo stats:  mean_norm={np.mean(np.linalg.norm(E_echo, axis=1)):.4f}")


if __name__ == "__main__":
    main()
