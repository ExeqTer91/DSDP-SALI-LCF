"""Phase 1 Experimental Harness: H1 + H2 + H5.

Runs all three hypothesis tests on provided embeddings and produces
a consolidated results report with per-test verdicts.
"""
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dsdp_sali_lcf.src.data_loader import load_embeddings, center_normalize, prepare_time_index
from dsdp_sali_lcf.experiments.run_single_agent import run_pipeline_extract
from dsdp_sali_lcf.experiments.run_abort_variant import run_abort_experiment
from dsdp_sali_lcf.tests.test_coherence_threshold import test_coherence_threshold
from dsdp_sali_lcf.tests.test_peripheral_coupling import (
    split_by_radius, perturb_subset, test_peripheral_vs_core
)
from dsdp_sali_lcf.tests.test_intentional_abort import test_abort_spin_control
from dsdp_sali_lcf.metrics.entropy import compute_per_bin_entropy
from dsdp_sali_lcf.metrics.wave import compute_per_bin_wave_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_h1_coherence_threshold(base_result: dict, n_bins: int = 50,
                                 tau: float = 0.03,
                                 wave_quantile: float = 0.75) -> dict:
    logger.info("=" * 60)
    logger.info("H1: COHERENCE THRESHOLD TEST")
    logger.info("=" * 60)

    entropies = compute_per_bin_entropy(
        base_result["y"], base_result["t"], base_result["labels"],
        n_bins=n_bins, tau=tau
    )

    wave_signals = compute_per_bin_wave_signal(
        base_result["temporal_scores"], wave_quantile
    )

    alignment_scores = base_result["temporal_scores"]

    n = min(len(entropies), len(wave_signals), len(alignment_scores))
    entropies = entropies[:n]
    wave_signals = wave_signals[:n]
    alignment_scores = alignment_scores[:n]

    result = test_coherence_threshold(entropies, wave_signals, alignment_scores,
                                       n_perm=500, seed=42)
    logger.info(f"H1 Result: lag-1 corr = {result.get('entropy_wave_lag1_correlation', 'N/A')}")
    if result.get("permutation_test"):
        pt = result["permutation_test"]
        logger.info(f"H1 Permutation: p={pt['p_value']:.4f}, z={pt['z_score']:.2f}")
    logger.info(f"H1 Verdict: {result.get('interpretation', 'N/A')}")
    return result


def run_h2_peripheral_coupling(E: np.ndarray, t: np.ndarray,
                                 base_result: dict,
                                 alpha: float = 0.3,
                                 n_mag: int = 6, tau: float = 0.03,
                                 eps: float = 1e-12, seed: int = 42,
                                 wave_quantile: float = 0.75) -> dict:
    logger.info("=" * 60)
    logger.info("H2: PERIPHERAL COUPLING TEST")
    logger.info("=" * 60)

    hub = base_result["hub"]
    peri_mask, core_mask = split_by_radius(E, hub, quantile=0.5)
    logger.info(f"Peripheral points: {peri_mask.sum()}, Core points: {core_mask.sum()}")

    E_peri = perturb_subset(E, peri_mask, alpha=alpha, seed=seed)
    peri_result = run_pipeline_extract(E_peri, t, n_mag, tau, eps, seed, wave_quantile)

    E_core = perturb_subset(E, core_mask, alpha=alpha, seed=seed + 100)
    core_result = run_pipeline_extract(E_core, t, n_mag, tau, eps, seed, wave_quantile)

    base_summary = {
        "alignment_score": base_result["alignment"],
        "identity_vector": base_result["centers"],
        "wave_persistence": base_result["longest_wave"],
        "fsi": base_result["fsi"],
    }
    peri_summary = {
        "alignment_score": peri_result["alignment"],
        "identity_vector": peri_result["centers"],
        "wave_persistence": peri_result["longest_wave"],
        "fsi": peri_result["fsi"],
    }
    core_summary = {
        "alignment_score": core_result["alignment"],
        "identity_vector": core_result["centers"],
        "wave_persistence": core_result["longest_wave"],
        "fsi": core_result["fsi"],
    }

    result = test_peripheral_vs_core(base_summary, peri_summary, core_summary)
    logger.info(f"H2 Core identity sim: {result['identity_core_similarity']:.4f}")
    logger.info(f"H2 Peripheral identity sim: {result['identity_peripheral_similarity']:.4f}")
    logger.info(f"H2 Alignment change peri: {result['alignment_change_peripheral']:.4f}")
    logger.info(f"H2 Alignment change core: {result['alignment_change_core']:.4f}")
    logger.info(f"H2 Verdict: {result['interpretation']}")
    return result


def run_h5_intentional_abort(E: np.ndarray, t: np.ndarray,
                               base_result: dict,
                               abort_strength: float = 0.5,
                               n_mag: int = 6, tau: float = 0.03,
                               eps: float = 1e-12, seed: int = 42,
                               wave_quantile: float = 0.75) -> dict:
    logger.info("=" * 60)
    logger.info("H5: INTENTIONAL SPIN CONTROL (ABORT) TEST")
    logger.info("=" * 60)

    abort_result = run_abort_experiment(
        E, t, base_result["y"],
        abort_strength=abort_strength,
        n_mag=n_mag, tau=tau, eps=eps, seed=seed, wave_quantile=wave_quantile,
        hub=base_result.get("hub")
    )

    normal_summary = {
        "alignment_score": base_result["alignment"],
        "identity_vector": base_result["centers"],
        "wave_persistence": base_result["longest_wave"],
        "fsi": base_result["fsi"],
    }
    abort_summary = {
        "alignment_score": abort_result["alignment"],
        "identity_vector": abort_result["centers"],
        "wave_persistence": abort_result["longest_wave"],
        "fsi": abort_result["fsi"],
    }

    result = test_abort_spin_control(normal_summary, abort_summary)
    logger.info(f"H5 Wave drop: {result['wave_drop']}")
    logger.info(f"H5 Alignment drop: {result['alignment_drop']:.4f}")
    logger.info(f"H5 Identity preserved: {result['identity_preserved_similarity']:.4f}")
    logger.info(f"H5 Verdict: {result['interpretation']}")
    return result


def print_phase1_report(h1, h2, h5):
    print("\n" + "=" * 70)
    print("PHASE 1 EXPERIMENTAL HARNESS - CONSOLIDATED RESULTS")
    print("=" * 70)

    print("\n--- H1: Coherence Threshold (SPIN -> RESONANCE) ---")
    print(f"  Entropy-Wave lag-1 correlation: {h1.get('entropy_wave_lag1_correlation', 'N/A'):.4f}"
          if h1.get('status') == 'ok' else f"  Status: {h1.get('status')}")
    if h1.get('status') == 'ok':
        print(f"  Entropy-Wave lag-0 correlation: {h1['entropy_wave_lag0_correlation']:.4f}")
        print(f"  Entropy-Alignment lag-1 corr:   {h1['entropy_alignment_lag1_correlation']:.4f}")
        print(f"  Optimal lag: {h1['optimal_lag']} (corr={h1['optimal_correlation']:.4f})")
        print(f"  Cross-correlation profile:")
        for lag, corr in sorted(h1["cross_correlation_profile"].items()):
            bar = "*" * max(0, int(abs(corr) * 20))
            sign = "+" if corr >= 0 else "-"
            print(f"    lag {lag:+2d}: {corr:+.4f} {sign}{bar}")
        if h1.get("permutation_test"):
            pt = h1["permutation_test"]
            print(f"  Permutation null (500 perms):")
            print(f"    observed lag-1: {pt['observed']:.4f}")
            print(f"    null mean:      {pt['null_mean']:.4f} +/- {pt['null_std']:.4f}")
            print(f"    p-value:        {pt['p_value']:.4f}")
            print(f"    z-score:        {pt['z_score']:.2f}")
    print(f"  PASS: {h1['pass']}")
    print(f"  >> {h1.get('interpretation', 'N/A')}")

    print("\n--- H2: Peripheral Coupling (PERIPHERY = COUPLING ZONE) ---")
    print(f"  Identity core similarity:       {h2['identity_core_similarity']:.4f}")
    print(f"  Identity peripheral similarity: {h2['identity_peripheral_similarity']:.4f}")
    print(f"  Alignment change (peripheral):  {h2['alignment_change_peripheral']:.4f}")
    print(f"  Alignment change (core):        {h2['alignment_change_core']:.4f}")
    print(f"  Wave change (peripheral):       {h2['wave_change_peripheral']:.1f}")
    print(f"  Wave change (core):             {h2['wave_change_core']:.1f}")
    print(f"  FSI change (peripheral):        {h2['fsi_change_peripheral']:.4f}")
    print(f"  FSI change (core):              {h2['fsi_change_core']:.4f}")
    print(f"  Total impact (peripheral):      {h2['total_impact_peripheral']:.4f}")
    print(f"  Total impact (core):            {h2['total_impact_core']:.4f}")
    print(f"  Peripheral more impact: {h2['peripheral_has_more_impact']}")
    print(f"  Core identity preserved: {h2['core_identity_preserved']}")
    print(f"  PASS: {h2['pass']}")
    print(f"  >> {h2['interpretation']}")

    print("\n--- H5: Intentional Spin Control / ABORT ---")
    print(f"  Wave drop:          {h5['wave_drop']:.1f}")
    print(f"  Alignment drop:     {h5['alignment_drop']:.4f}")
    print(f"  FSI drop:           {h5['fsi_drop']:.4f}")
    print(f"  Coupling effect:    {h5['coupling_effect']:.4f}")
    print(f"  Identity similarity:{h5['identity_preserved_similarity']:.4f}")
    print(f"  Coupling dropped:   {h5['coupling_dropped']}")
    print(f"  Identity preserved: {h5['identity_preserved']}")
    print(f"  PASS: {h5['pass']}")
    print(f"  >> {h5['interpretation']}")

    n_pass = sum([h1["pass"], h2["pass"], h5["pass"]])
    print(f"\n{'='*70}")
    print(f"PHASE 1 SUMMARY: {n_pass}/3 hypotheses confirmed")
    if n_pass == 3:
        print(">> ALL CONFIRMED: Strong evidence for coherence threshold + peripheral coupling + spin control")
        print(">> Phase 2 (H3 + H4) is well-motivated.")
    elif n_pass >= 2:
        print(">> PARTIAL: Majority of Phase 1 hypotheses confirmed.")
        print(">> Phase 2 may proceed with caveats.")
    elif n_pass == 1:
        print(">> WEAK: Only one hypothesis confirmed. Review methodology before Phase 2.")
    else:
        print(">> NONE CONFIRMED: Phase 1 hypotheses not supported on this data.")
        print(">> Reassess before proceeding to Phase 2.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Experimental Harness")
    parser.add_argument("--fresh", required=True, help="Path to fresh embeddings")
    parser.add_argument("--n-mag", type=int, default=6)
    parser.add_argument("--tau", type=float, default=0.03)
    parser.add_argument("--perturb-alpha", type=float, default=0.3,
                        help="Perturbation strength for H2")
    parser.add_argument("--abort-strength", type=float, default=0.5,
                        help="Abort signal strength for H5")
    parser.add_argument("--output-dir", default="dsdp_sali_lcf/outputs/phase1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 1 EXPERIMENTAL HARNESS")
    logger.info("=" * 60)

    data = load_embeddings(args.fresh, seed=args.seed)
    E = data["E"]
    t_raw = data["t"]
    E, center, scale = center_normalize(E)
    t, t_meta = prepare_time_index(t_raw, len(E), E=E)

    logger.info("Running base pipeline...")
    base = run_pipeline_extract(E, t, args.n_mag, args.tau, seed=args.seed)

    h1 = run_h1_coherence_threshold(base, tau=args.tau)
    h2 = run_h2_peripheral_coupling(E, t, base, alpha=args.perturb_alpha,
                                      n_mag=args.n_mag, tau=args.tau, seed=args.seed)
    h5 = run_h5_intentional_abort(E, t, base, abort_strength=args.abort_strength,
                                    n_mag=args.n_mag, tau=args.tau, seed=args.seed)

    print_phase1_report(h1, h2, h5)

    report = {
        "pipeline": "DSDP_SALI_LCF_PHASE1",
        "source": args.fresh,
        "n_points": len(E),
        "n_dim": E.shape[1],
        "H1_coherence_threshold": {
            k: v for k, v in h1.items()
            if k != "cross_correlation_profile"
        },
        "H1_cross_correlation": h1.get("cross_correlation_profile", {}),
        "H2_peripheral_coupling": h2,
        "H5_intentional_abort": h5,
        "phase1_summary": {
            "n_pass": sum([h1["pass"], h2["pass"], h5["pass"]]),
            "h1_pass": h1["pass"],
            "h2_pass": h2["pass"],
            "h5_pass": h5["pass"],
        },
    }

    report_path = output_dir / "phase1_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Phase 1 report saved: {report_path}")

    return report


if __name__ == "__main__":
    main()
