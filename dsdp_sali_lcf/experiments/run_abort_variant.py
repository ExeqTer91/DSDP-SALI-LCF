"""Run abort variant: inject abort signal and re-run pipeline."""
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dsdp_sali_lcf.experiments.run_single_agent import run_pipeline_extract
from dsdp_sali_lcf.tests.test_intentional_abort import inject_abort_signal


def run_abort_experiment(E: np.ndarray, t: np.ndarray, y: np.ndarray,
                          abort_strength: float = 0.5,
                          n_mag: int = 6, tau: float = 0.03,
                          eps: float = 1e-12, seed: int = 42,
                          wave_quantile: float = 0.75,
                          hub: np.ndarray = None) -> Dict[str, Any]:
    E_abort = inject_abort_signal(E, y, abort_strength=abort_strength,
                                   seed=seed, hub=hub)
    return run_pipeline_extract(E_abort, t, n_mag, tau, eps, seed, wave_quantile)
