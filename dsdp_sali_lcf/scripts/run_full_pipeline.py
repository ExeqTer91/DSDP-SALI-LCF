"""Iteration 2: Full Pipeline Run.
Runs geometry + lattice + temporal + nulls (B=500) + Holm + ablations + perturbation + echo vs fresh + verdict.
"""
import sys
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
    compute_log_radii, score_per_magistral, predict_next_band, SQRT_PHI
)
from dsdp_sali_lcf.src.temporal import (
    compute_temporal_scores, wave_persistence, field_strength_index, compute_clustering_entropy
)
from dsdp_sali_lcf.src.nulls import run_null_suite, holm_correction
from dsdp_sali_lcf.src.ablations import seed_ablation, coupler_ablation, anchor_ablation, temporal_ablation
from dsdp_sali_lcf.src.perturbation import perturbation_curve
from dsdp_sali_lcf.src.plots import (
    plot_pca_magistrales, plot_residual_histogram, plot_wave_lengths,
    plot_temporal_scores, plot_band_predictions, plot_fsi_breakdown,
    plot_perturbation_curve, plot_null_pvalues, plot_echo_vs_fresh
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path=None):
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        default = Path(__file__).resolve().parent.parent / "config.yaml"
        if default.exists():
            with open(default) as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}
    cfg.setdefault("random_seed", 42)
    cfg.setdefault("max_points", None)
    cfg.setdefault("n_magistrales", 6)
    cfg.setdefault("tau", 0.03)
    cfg.setdefault("eps", 1e-12)
    cfg.setdefault("n_surrogates_full", 500)
    cfg.setdefault("n_splits_stability", 20)
    cfg.setdefault("decoy_seeds", [1.70, 2.10, 2.60])
    cfg.setdefault("switching_penalty", 0.1)
    cfg.setdefault("wave_quantile_threshold", 0.75)
    cfg.setdefault("perturbation_alphas", [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    return cfg


def run_single_dataset(data_path: str, label: str, config: dict, output_dir: Path) -> dict:
    """Run full pipeline on one dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"FULL PIPELINE: {label}")
    logger.info(f"{'='*60}")
    
    seed = int(config["random_seed"])
    np.random.seed(seed)
    n_mag = int(config["n_magistrales"])
    tau = float(config["tau"])
    eps = float(config["eps"])
    n_surr = int(config["n_surrogates_full"])
    
    data = load_embeddings(data_path, max_points=config.get("max_points"), seed=seed)
    E = data["E"]
    t_raw = data["t"]
    meta = data["meta"]
    
    val_diag = validate_embeddings(E)
    E, center, scale = center_normalize(E)
    t, t_meta = prepare_time_index(t_raw, len(E), E=E)
    
    logger.info("1) Geometry...")
    geo = build_geometry(E, n_mag, seed)
    r = compute_radii(E, geo["hub"])
    y = compute_log_radii(r, eps)
    
    logger.info("2) Lattice scoring...")
    lattice = score_per_magistral(y, geo["labels"], n_mag, tau,
                                   switching_penalty=config["switching_penalty"])
    
    logger.info("3) Predictions (E6)...")
    preds = predict_next_band(y, geo["labels"], n_mag, tau)
    
    logger.info("4) Temporal analysis...")
    temporal = compute_temporal_scores(y, t, geo["labels"], n_mag, tau)
    waves = wave_persistence(temporal["scores"], config["wave_quantile_threshold"])
    entropy = compute_clustering_entropy(geo["labels"], n_mag)
    
    logger.info(f"5) Null suite (B={n_surr})...")
    
    def alignment_score_fn(y_in, labels_in):
        sc = score_per_magistral(y_in, labels_in, n_mag, tau)
        return sc["global_alignment"]
    
    null_results = run_null_suite(y, geo["labels"], E, alignment_score_fn,
                                  n_surrogates=n_surr, n_magistrales=n_mag, seed=seed)
    
    surprise_z = null_results.get("min_z", 0.0)
    fsi_data = field_strength_index(lattice["global_alignment"], surprise_z, entropy,
                                     config.get("fsi_weights"))
    
    logger.info("6) Ablations...")
    seed_abl = seed_ablation(y, geo["labels"], n_mag, tau, config["decoy_seeds"])
    coupler_abl = coupler_ablation(y, geo["labels"], n_mag, tau)
    anchor_abl = anchor_ablation(E, y, geo["labels"], n_mag, tau, seed=seed)
    temp_abl = temporal_ablation(y, t, geo["labels"], n_mag, tau, seed=seed)
    
    logger.info("7) Perturbation curve...")
    
    def full_score_fn(E_p):
        geo_p = build_geometry(E_p, n_mag, seed)
        r_p = compute_radii(E_p, geo_p["hub"])
        y_p = compute_log_radii(r_p, eps)
        sc = score_per_magistral(y_p, geo_p["labels"], n_mag, tau)
        return sc["global_alignment"]
    
    perturb = perturbation_curve(E, full_score_fn, config["perturbation_alphas"], seed)
    
    logger.info("8) Split-half stability...")
    n_splits = config["n_splits_stability"]
    split_scores = []
    n = len(E)
    for sp in range(n_splits):
        rng = np.random.RandomState(seed + sp * 777)
        idx = rng.choice(n, n // 2, replace=False)
        E_half = E[idx]
        try:
            geo_h = build_geometry(E_half, n_mag, seed + sp)
            r_h = compute_radii(E_half, geo_h["hub"])
            y_h = compute_log_radii(r_h, eps)
            sc_h = score_per_magistral(y_h, geo_h["labels"], n_mag, tau)
            split_scores.append(sc_h["global_alignment"])
        except Exception as e:
            logger.warning(f"Split {sp} failed: {e}")
    
    split_stability = {
        "mean": float(np.mean(split_scores)) if split_scores else 0.0,
        "std": float(np.std(split_scores)) if split_scores else 0.0,
        "n_successful": len(split_scores),
        "stable": (float(np.std(split_scores)) < 0.1 * float(np.mean(split_scores))) if split_scores else False,
    }
    
    logger.info("9) Generating plots...")
    plot_pca_magistrales(E, geo["labels"],
                         str(output_dir / f"{label}_pca_magistrales.png"),
                         f"PCA Magistrales - {label}")
    
    step_a = np.log(SQRT_PHI)
    all_res = []
    for m in range(n_mag):
        mask = geo["labels"] == m
        y_m = y[mask]
        if len(y_m) > 0:
            res = np.abs(y_m - np.round(y_m / step_a) * step_a)
            all_res.extend(res.tolist())
    
    if all_res:
        plot_residual_histogram(np.array(all_res), tau,
                                str(output_dir / f"{label}_residual_hist.png"),
                                f"Residuals - {label}")
    
    plot_wave_lengths(waves["wave_lengths"],
                      str(output_dir / f"{label}_wave_lengths.png"), f"Waves - {label}")
    plot_temporal_scores(temporal["bin_centers"], temporal["scores"],
                         str(output_dir / f"{label}_temporal_st.png"), f"S(t) - {label}")
    plot_band_predictions(preds["predictions"],
                          str(output_dir / f"{label}_band_predictions.png"),
                          f"Band Predictions - {label}")
    plot_fsi_breakdown(fsi_data,
                       str(output_dir / f"{label}_fsi_breakdown.png"),
                       f"FSI - {label}")
    plot_perturbation_curve(perturb["alphas"], perturb["scores"], perturb["tipping_alpha"],
                            str(output_dir / f"{label}_perturbation.png"),
                            f"Perturbation - {label}")
    plot_null_pvalues(null_results,
                      str(output_dir / f"{label}_null_pvalues.png"),
                      f"Null P-values - {label}")
    
    result = {
        "label": label,
        "source": data_path,
        "shape": list(E.shape),
        "lattice": {
            "global_alignment": lattice["global_alignment"],
            "global_median_resid": lattice["global_median_resid"],
            "per_magistral": lattice["per_magistral"],
        },
        "predictions": {
            "median_error": preds["median_error"],
            "n_valid": preds["n_valid"],
            "details": preds["predictions"],
        },
        "temporal": {
            "n_waves": waves["n_waves"],
            "longest_wave": waves["longest_wave"],
            "mean_wave": waves["mean_wave"],
        },
        "null_suite": {
            k: {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in null_results.items()
        },
        "conservative_p": null_results["conservative_p"],
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
                "difference": coupler_abl["difference"],
                "matters": coupler_abl["coupler_matters"],
            },
            "anchor": {
                "original": anchor_abl["original_score"],
                "mean_random": anchor_abl["mean_random"],
                "advantage": anchor_abl["stability_advantage"],
                "stable": anchor_abl["anchor_stable"],
            },
            "temporal_order": {
                "original_longest_wave": temp_abl["original_longest_wave"],
                "shuffled_longest_wave": temp_abl["shuffled_longest_wave"],
                "matters": temp_abl["temporal_order_matters"],
            },
        },
        "perturbation": {
            "alphas": perturb["alphas"],
            "scores": perturb["scores"],
            "tipping_alpha": perturb["tipping_alpha"],
        },
        "split_half_stability": split_stability,
        "alignment": lattice["global_alignment"],
        "median_error": preds["median_error"],
    }
    
    return result


def compute_verdict(fresh: dict, echo: dict, config: dict) -> dict:
    """Strict verdict logic with kill-switches."""
    
    fresh_p = fresh["conservative_p"]
    echo_p = echo["conservative_p"]
    
    all_p = [fresh_p, echo_p]
    holm = holm_correction(all_p, alpha=0.05)
    
    fresh_alignment = fresh["lattice"]["global_alignment"]
    echo_alignment = echo["lattice"]["global_alignment"]
    alignment_ratio = echo_alignment / max(fresh_alignment, 1e-12)
    
    fresh_fsi = fresh["fsi"]["fsi"]
    echo_fsi = echo["fsi"]["fsi"]
    fsi_ratio = echo_fsi / max(fresh_fsi, 1e-12)
    
    fresh_error = fresh["predictions"]["median_error"]
    echo_error = echo["predictions"]["median_error"]
    error_ratio = echo_error / max(fresh_error, 1e-12) if fresh_error > 0 else float("inf")
    
    kills = []
    
    null_types = ["uniform", "gaussian_cov", "spacing"]
    sig_count = 0
    for nt in null_types:
        if nt in fresh["null_suite"]:
            if fresh["null_suite"][nt]["p_value"] < 0.05:
                sig_count += 1
    if sig_count > 0 and sig_count < len(null_types):
        kills.append(f"KILL: Significant under only {sig_count}/{len(null_types)} nulls")
    
    if not fresh["ablations"]["seed"]["seed_is_special"]:
        kills.append("KILL: Seed ablation shows sqrt(phi) is NOT special vs decoys")
    
    fresh_stable = fresh["split_half_stability"]["stable"]
    
    primary_sig = holm["rejected"][0] if len(holm["rejected"]) > 0 else False
    ablations_support = (fresh["ablations"]["seed"]["seed_is_special"] and
                         fresh["ablations"]["anchor"]["stable"])
    
    if kills:
        verdict = "FAIL - Kill-switch triggered"
        claim = False
    elif primary_sig and ablations_support and fresh_stable:
        verdict = "STRUCTURE DETECTED (conservative)"
        claim = True
    elif primary_sig and not ablations_support:
        verdict = "INCONCLUSIVE - Primary significant but ablations do not support"
        claim = False
    elif not primary_sig:
        verdict = "NO STRUCTURE DETECTED - Primary endpoint not significant"
        claim = False
    else:
        verdict = "INCONCLUSIVE"
        claim = False
    
    return {
        "verdict": verdict,
        "claim_structure": claim,
        "kill_switches": kills,
        "holm_correction": {
            "p_values": all_p,
            "adjusted": holm["adjusted"],
            "rejected": holm["rejected"],
        },
        "echo_vs_fresh": {
            "alignment_ratio": alignment_ratio,
            "fsi_ratio": fsi_ratio,
            "error_ratio": error_ratio,
            "anti_contamination": alignment_ratio < 1.0,
        },
        "fresh_primary_p": fresh_p,
        "echo_primary_p": echo_p,
        "split_half_stable": fresh_stable,
    }


def print_results_block(fresh, echo, verdict_data):
    """Print paper-ready results block."""
    print("\n" + "=" * 70)
    print("PAPER-READY RESULTS BLOCK")
    print("=" * 70)
    
    print(f"\n--- Primary Endpoint: Next-Band Prediction (E6) ---")
    print(f"  Fresh: median_error = {fresh['predictions']['median_error']:.6f}")
    print(f"  Echo:  median_error = {echo['predictions']['median_error']:.6f}")
    
    print(f"\n--- P-values (per null model, FRESH) ---")
    for nt in ["uniform", "gaussian_cov", "spacing"]:
        if nt in fresh["null_suite"] and isinstance(fresh["null_suite"][nt], dict):
            p = fresh["null_suite"][nt].get("p_value", "N/A")
            z = fresh["null_suite"][nt].get("z_score", "N/A")
            print(f"  {nt:20s}: p={p:.4f}, z={z:.2f}" if isinstance(p, float) else f"  {nt}: {p}")
    
    print(f"\n  Conservative p (max across nulls): {fresh['conservative_p']:.4f}")
    
    print(f"\n--- Holm-Adjusted Decision ---")
    print(f"  p-values: {verdict_data['holm_correction']['p_values']}")
    print(f"  adjusted: {verdict_data['holm_correction']['adjusted']}")
    print(f"  rejected: {verdict_data['holm_correction']['rejected']}")
    
    print(f"\n--- Key Ablation Outcomes (FRESH) ---")
    abl = fresh["ablations"]
    print(f"  Seed: sqrt(phi)={abl['seed']['real_score']:.4f} vs max_decoy={abl['seed']['max_decoy']:.4f} -> special={abl['seed']['seed_is_special']}")
    print(f"  Coupler: with={abl['coupler']['with']:.4f} vs without={abl['coupler']['without']:.4f} -> matters={abl['coupler']['matters']}")
    print(f"  Anchor: original={abl['anchor']['original']:.4f} vs random_mean={abl['anchor']['mean_random']:.4f} -> stable={abl['anchor']['stable']}")
    print(f"  Temporal: original_longest_wave={abl['temporal_order']['original_longest_wave']} vs shuffled={abl['temporal_order']['shuffled_longest_wave']} -> matters={abl['temporal_order']['matters']}")
    
    print(f"\n--- Echo vs Fresh ---")
    evf = verdict_data["echo_vs_fresh"]
    print(f"  Alignment ratio (echo/fresh): {evf['alignment_ratio']:.4f}")
    print(f"  FSI ratio (echo/fresh):       {evf['fsi_ratio']:.4f}")
    print(f"  Error ratio (echo/fresh):     {evf['error_ratio']:.4f}")
    print(f"  Anti-contamination (ratio<1): {evf['anti_contamination']}")
    
    print(f"\n--- Split-Half Stability ---")
    sh = fresh["split_half_stability"]
    print(f"  Mean={sh['mean']:.4f}, Std={sh['std']:.4f}, Stable={sh['stable']}")
    
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict_data['verdict']}")
    if verdict_data["kill_switches"]:
        for ks in verdict_data["kill_switches"]:
            print(f"  >> {ks}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="DSDP Full Pipeline")
    parser.add_argument("--fresh", required=True, help="Path to fresh embeddings")
    parser.add_argument("--echo", required=True, help="Path to echo embeddings")
    parser.add_argument("--config", default=None, help="Config YAML")
    parser.add_argument("--output-dir", default="dsdp_sali_lcf/outputs/full", help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Fast mode: B=50 surrogates instead of 500")
    args = parser.parse_args()
    
    config = load_config(args.config)
    config["calibration_mode"] = False
    if args.fast:
        config["n_surrogates_full"] = 50
        logger.info("FAST MODE: B=50 surrogates (reduced from 500)")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("DSDP SALI LCF - FULL PIPELINE RUN")
    logger.info("=" * 60)
    
    fresh_result = run_single_dataset(args.fresh, "fresh", config, output_dir)
    echo_result = run_single_dataset(args.echo, "echo", config, output_dir)
    
    plot_echo_vs_fresh(
        {"alignment": fresh_result["alignment"], "fsi": fresh_result["fsi"]["fsi"],
         "median_error": fresh_result["median_error"]},
        {"alignment": echo_result["alignment"], "fsi": echo_result["fsi"]["fsi"],
         "median_error": echo_result["median_error"]},
        str(output_dir / "echo_vs_fresh_comparison.png")
    )
    
    verdict_data = compute_verdict(fresh_result, echo_result, config)
    
    combined = {
        "pipeline": "DSDP_SALI_LCF_FULL",
        "fresh": fresh_result,
        "echo": echo_result,
        "verdict": verdict_data,
    }
    
    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    
    fresh_path = output_dir / "fresh_report.json"
    with open(fresh_path, "w") as f:
        json.dump(fresh_result, f, indent=2, default=str)
    
    echo_path = output_dir / "echo_report.json"
    with open(echo_path, "w") as f:
        json.dump(echo_result, f, indent=2, default=str)
    
    print_results_block(fresh_result, echo_result, verdict_data)
    
    logger.info(f"Full report: {report_path}")
    return combined


if __name__ == "__main__":
    main()
