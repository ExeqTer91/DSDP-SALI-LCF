"""Iteration 1: Calibration / No-Error Pass.
Loads subset, validates, runs geometry + lattice + temporal + nulls + ablations on small scale.
"""
import sys
import os
import json
import argparse
import logging
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dsdp_sali_lcf.src.data_loader import (
    load_embeddings, validate_embeddings, prepare_time_index, center_normalize
)
from dsdp_sali_lcf.src.geometry import build_geometry, compute_radii
from dsdp_sali_lcf.src.lattice import (
    compute_log_radii, score_per_magistral, dual_grid_score, predict_next_band
)
from dsdp_sali_lcf.src.temporal import (
    compute_temporal_scores, wave_persistence, field_strength_index, compute_clustering_entropy
)
from dsdp_sali_lcf.src.nulls import run_null_suite
from dsdp_sali_lcf.src.ablations import seed_ablation, coupler_ablation, anchor_ablation
from dsdp_sali_lcf.src.plots import (
    plot_pca_magistrales, plot_residual_histogram, plot_wave_lengths, plot_temporal_scores
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    default = Path(__file__).resolve().parent.parent / "config.yaml"
    if default.exists():
        with open(default) as f:
            return yaml.safe_load(f)
    return {
        "random_seed": 42, "max_points": 50000, "n_magistrales": 6,
        "tau": 0.03, "eps": 1e-12, "n_surrogates_calib": 50,
        "wave_quantile_threshold": 0.75, "decoy_seeds": [1.70, 2.10, 2.60],
        "switching_penalty": 0.1,
    }


def run_calibration_single(data_path: str, label: str, config: dict, output_dir: Path) -> dict:
    """Run calibration on a single dataset."""
    logger.info(f"=== CALIBRATION: {label} ===")
    
    seed = int(config.get("random_seed", 42))
    np.random.seed(seed)
    max_points = int(config.get("max_points", 50000))
    n_mag = int(config.get("n_magistrales", 6))
    tau = float(config.get("tau", 0.03))
    eps = float(config.get("eps", 1e-12))
    
    logger.info(f"Loading {data_path}...")
    data = load_embeddings(data_path, max_points=max_points, seed=seed)
    E = data["E"]
    t_raw = data["t"]
    meta = data["meta"]
    
    logger.info("Validating embeddings...")
    val_diag = validate_embeddings(E)
    logger.info(f"Validation: {val_diag}")
    
    logger.info("Centering and normalizing...")
    E, center, scale = center_normalize(E)
    
    logger.info("Preparing time index...")
    t, t_meta = prepare_time_index(t_raw, len(E), remap_by_proxy=False, E=E)
    
    logger.info("Building geometry (anchor, hub, magistrales)...")
    geo = build_geometry(E, n_mag, seed)
    
    logger.info("Computing radii and log-space...")
    r = compute_radii(E, geo["hub"])
    y = compute_log_radii(r, eps)
    
    logger.info("Scoring lattice alignment...")
    lattice_scores = score_per_magistral(y, geo["labels"], n_mag, tau,
                                          switching_penalty=config.get("switching_penalty", 0.1))
    logger.info(f"Global alignment: {lattice_scores['global_alignment']:.4f}")
    logger.info(f"Global median residual: {lattice_scores['global_median_resid']:.6f}")
    
    logger.info("Computing temporal scores...")
    temporal = compute_temporal_scores(y, t, geo["labels"], n_mag, tau)
    waves = wave_persistence(temporal["scores"], config.get("wave_quantile_threshold", 0.75))
    logger.info(f"Waves: {waves['n_waves']} total, longest={waves['longest_wave']}")
    
    logger.info("Computing predictions (E6)...")
    preds = predict_next_band(y, geo["labels"], n_mag, tau)
    logger.info(f"Prediction median error: {preds['median_error']:.6f}")
    
    logger.info("Running null suite (B={})...".format(config.get("n_surrogates_calib", 50)))
    
    def score_fn(y_in, labels_in):
        sc = score_per_magistral(y_in, labels_in, n_mag, tau)
        return sc["global_alignment"]
    
    null_results = run_null_suite(
        y, geo["labels"], E, score_fn,
        n_surrogates=config.get("n_surrogates_calib", 50),
        n_magistrales=n_mag, seed=seed
    )
    
    logger.info("Computing clustering entropy & FSI...")
    entropy = compute_clustering_entropy(geo["labels"], n_mag)
    surprise_z = null_results.get("min_z", 0.0)
    fsi_data = field_strength_index(
        lattice_scores["global_alignment"], surprise_z, entropy,
        config.get("fsi_weights")
    )
    logger.info(f"FSI: {fsi_data['fsi']:.4f}")
    
    logger.info("Running ablations...")
    seed_abl = seed_ablation(y, geo["labels"], n_mag, tau, config.get("decoy_seeds"))
    coupler_abl = coupler_ablation(y, geo["labels"], n_mag, tau)
    anchor_abl = anchor_ablation(E, y, geo["labels"], n_mag, tau, seed=seed)
    
    logger.info("Generating plots...")
    plot_pca_magistrales(E, geo["labels"],
                         str(output_dir / f"{label}_pca_magistrales.png"),
                         f"PCA Magistrales - {label}")
    
    all_residuals = []
    step_a = np.log(np.sqrt(1.618033988749895))
    for m in range(n_mag):
        mask = geo["labels"] == m
        y_m = y[mask]
        if len(y_m) > 0:
            res = np.abs(y_m - np.round(y_m / step_a) * step_a)
            all_residuals.extend(res.tolist())
    
    if all_residuals:
        plot_residual_histogram(np.array(all_residuals), tau,
                                str(output_dir / f"{label}_residual_hist.png"),
                                f"Residual Histogram - {label}")
    
    plot_wave_lengths(waves["wave_lengths"],
                      str(output_dir / f"{label}_wave_lengths.png"),
                      f"Wave Lengths - {label}")
    
    plot_temporal_scores(temporal["bin_centers"], temporal["scores"],
                         str(output_dir / f"{label}_temporal_scores.png"),
                         f"S(t) - {label}")
    
    report = {
        "label": label,
        "source": data_path,
        "meta": {k: str(v) if isinstance(v, (np.ndarray, np.generic)) else v 
                 for k, v in meta.items()},
        "validation": val_diag,
        "time_index": t_meta,
        "geometry": {
            "anchor_pair": list(geo["anchor_pair"]),
            "mag_counts": geo["mag_counts"],
        },
        "lattice": {
            "global_alignment": lattice_scores["global_alignment"],
            "global_median_resid": lattice_scores["global_median_resid"],
            "per_magistral": lattice_scores["per_magistral"],
        },
        "temporal": {
            "n_waves": waves["n_waves"],
            "longest_wave": waves["longest_wave"],
            "mean_wave": waves["mean_wave"],
            "wave_lengths": waves["wave_lengths"],
        },
        "predictions": {
            "median_error": preds["median_error"],
            "n_valid": preds["n_valid"],
        },
        "null_suite": {
            k: v for k, v in null_results.items() 
            if k not in ("conservative_p", "min_z", "observed") or isinstance(v, (int, float, str))
        },
        "conservative_p": null_results["conservative_p"],
        "min_z": null_results["min_z"],
        "fsi": fsi_data,
        "ablations": {
            "seed": {
                "real_score": seed_abl["real_score"],
                "max_decoy": seed_abl["max_decoy_score"],
                "advantage": seed_abl["advantage"],
                "seed_is_special": seed_abl["seed_is_special"],
            },
            "coupler": {
                "with": coupler_abl["with_coupler"],
                "without": coupler_abl["without_coupler"],
                "diff": coupler_abl["difference"],
                "matters": coupler_abl["coupler_matters"],
            },
            "anchor": {
                "original": anchor_abl["original_score"],
                "mean_random": anchor_abl["mean_random"],
                "advantage": anchor_abl["stability_advantage"],
                "stable": anchor_abl["anchor_stable"],
            },
        },
        "status": "CALIBRATION_OK",
    }
    
    return report


def main():
    parser = argparse.ArgumentParser(description="DSDP Calibration Pass")
    parser.add_argument("--fresh", required=True, help="Path to fresh embeddings file")
    parser.add_argument("--echo", required=True, help="Path to echo embeddings file")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--output-dir", default="dsdp_sali_lcf/outputs/calib", help="Output directory")
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("DSDP SALI LCF - CALIBRATION PASS")
    logger.info("=" * 60)
    
    fresh_report = run_calibration_single(args.fresh, "fresh", config, output_dir)
    echo_report = run_calibration_single(args.echo, "echo", config, output_dir)
    
    combined = {
        "pipeline": "DSDP_SALI_LCF_CALIBRATION",
        "status": "OK" if fresh_report["status"] == "CALIBRATION_OK" and echo_report["status"] == "CALIBRATION_OK" else "FAILED",
        "fresh": fresh_report,
        "echo": echo_report,
    }
    
    report_path = output_dir / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    
    logger.info(f"\nCalibration report saved: {report_path}")
    
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Fresh: alignment={fresh_report['lattice']['global_alignment']:.4f}, "
          f"FSI={fresh_report['fsi']['fsi']:.4f}, "
          f"conservative_p={fresh_report['conservative_p']:.4f}")
    print(f"Echo:  alignment={echo_report['lattice']['global_alignment']:.4f}, "
          f"FSI={echo_report['fsi']['fsi']:.4f}, "
          f"conservative_p={echo_report['conservative_p']:.4f}")
    print(f"Seed ablation (fresh): advantage={fresh_report['ablations']['seed']['advantage']:.4f}")
    print(f"Coupler ablation (fresh): diff={fresh_report['ablations']['coupler']['diff']:.4f}")
    print(f"Anchor stability (fresh): advantage={fresh_report['ablations']['anchor']['advantage']:.4f}")
    print(f"\nStatus: {combined['status']}")
    print("=" * 60)
    
    return combined


if __name__ == "__main__":
    main()
