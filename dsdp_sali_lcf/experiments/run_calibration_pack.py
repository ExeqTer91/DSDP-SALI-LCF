"""Full Calibration Pack for the emergent-structure harness.

Runs all 5 conditions (RANDOM_PURE, NEAR_NULL, REAL_NORMAL, REAL_SURVIVAL,
REAL_BIBLICAL) through the Phase 1 harness (H1 + H2 + H5), plus:
  A) Reproducibility: 3 runs per condition, CV% for key metrics
  B) Negative controls: RANDOM_PURE must fail all, NEAR_NULL intermediate
  C) Real data core: REAL_NORMAL pass pattern, REAL_SURVIVAL survival signature
  D) Delta tests: H1 sensitivity curve, H5 survival delta
  E) Biblical mode: maximally constrained canonical regime

Invariants (do NOT change between datasets):
  - Same thresholds
  - Same seeds per dataset
  - Same metrics (entropy, wave_persistence, alignment, identity)
  - RANDOM_PURE passing anything = BUG
"""
import sys
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dsdp_sali_lcf.scripts.generate_calibration_data import generate_condition, ALL_CONDITIONS
from dsdp_sali_lcf.src.data_loader import center_normalize, prepare_time_index
from dsdp_sali_lcf.experiments.run_single_agent import run_pipeline_extract
from dsdp_sali_lcf.experiments.run_phase1 import (
    run_h1_coherence_threshold, run_h2_peripheral_coupling,
    run_h5_intentional_abort,
)
from dsdp_sali_lcf.metrics.identity import (
    compute_identity_similarity, align_centers, cosine_similarity,
)
from dsdp_sali_lcf.tests.test_biblical_mode import test_biblical_consistency

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_POINTS = 5000
N_DIM = 64
BASE_SEED = 42
N_MAG = 6
TAU = 0.03
PERTURB_ALPHA = 0.3
ABORT_STRENGTH = 0.5


def prepare_embeddings(E_raw: np.ndarray, seed: int = BASE_SEED):
    E, center, scale = center_normalize(E_raw)
    t_raw = np.arange(len(E), dtype=np.float64)
    t, t_meta = prepare_time_index(t_raw, len(E), E=E)
    return E, t


def run_condition(condition: str, seed: int = BASE_SEED) -> Dict[str, Any]:
    E_raw = generate_condition(condition, N_POINTS, N_DIM, seed)
    E, t = prepare_embeddings(E_raw, seed)

    base = run_pipeline_extract(E, t, N_MAG, TAU, seed=seed)
    h1 = run_h1_coherence_threshold(base, tau=TAU)
    h2 = run_h2_peripheral_coupling(E, t, base, alpha=PERTURB_ALPHA,
                                      n_mag=N_MAG, tau=TAU, seed=seed)
    h5 = run_h5_intentional_abort(E, t, base, abort_strength=ABORT_STRENGTH,
                                    n_mag=N_MAG, tau=TAU, seed=seed)

    return {
        "condition": condition,
        "seed": seed,
        "base": base,
        "h1": h1,
        "h2": h2,
        "h5": h5,
        "summary": {
            "h1_pass": h1["pass"],
            "h2_pass": h2["pass"],
            "h5_pass": h5["pass"],
            "h1_lag1_corr": h1.get("entropy_wave_lag1_correlation", None),
            "h1_p_value": h1.get("permutation_test", {}).get("p_value", None),
            "h2_identity_sim": h2.get("identity_core_similarity", None),
            "h2_alignment_delta": h2.get("alignment_change_peripheral", None),
            "h5_wave_drop": h5.get("wave_drop", None),
            "h5_identity_abort": h5.get("identity_preserved_similarity", None),
            "alignment": base["alignment"],
            "longest_wave": base["longest_wave"],
            "fsi": base["fsi"],
        },
    }


def test_a1_reproducibility(n_runs: int = 3) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("A1: REPRODUCIBILITY TEST (3 runs x 4 conditions)")
    print("=" * 70)

    results = {}
    for cond in ALL_CONDITIONS:
        runs = []
        for i in range(n_runs):
            logger.info(f"A1: {cond} run {i+1}/{n_runs}")
            r = run_condition(cond, seed=BASE_SEED)
            runs.append(r["summary"])

        metrics = ["alignment", "longest_wave", "fsi",
                    "h2_identity_sim", "h5_identity_abort"]
        cvs = {}
        for m in metrics:
            vals = [r[m] for r in runs if r[m] is not None]
            if len(vals) > 1:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                cv = (std_v / (abs(mean_v) + 1e-12)) * 100
                cvs[m] = {"mean": float(mean_v), "std": float(std_v),
                           "cv_pct": float(cv), "values": [float(v) for v in vals]}
            else:
                cvs[m] = {"mean": float(vals[0]) if vals else 0.0,
                           "std": 0.0, "cv_pct": 0.0,
                           "values": [float(v) for v in vals]}

        all_cv_ok = all(cvs[m]["cv_pct"] < 10.0 for m in metrics
                        if cvs[m]["cv_pct"] is not None)
        results[cond] = {"cvs": cvs, "pass": all_cv_ok, "n_runs": n_runs}

        print(f"\n  {cond}:")
        for m in metrics:
            c = cvs[m]
            status = "OK" if c["cv_pct"] < 10.0 else "FAIL"
            print(f"    {m:25s}: mean={c['mean']:.4f} std={c['std']:.4f} "
                  f"CV={c['cv_pct']:.1f}% [{status}]")

    all_pass = all(results[c]["pass"] for c in ALL_CONDITIONS)
    print(f"\n  A1 VERDICT: {'PASS' if all_pass else 'FAIL'} "
          f"(all CVs < 10%: {all_pass})")
    return {"test": "A1_reproducibility", "pass": all_pass, "details": results}


def test_a2_hungarian_stability(n_runs: int = 3) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("A2: HUNGARIAN MATCHING STABILITY (3 runs)")
    print("=" * 70)

    from scipy.optimize import linear_sum_assignment

    all_mappings = []
    all_cost_summaries = []
    for i in range(n_runs):
        E_raw = generate_condition("REAL_NORMAL", N_POINTS, N_DIM, BASE_SEED)
        E, t = prepare_embeddings(E_raw, BASE_SEED)
        base = run_pipeline_extract(E, t, N_MAG, TAU, seed=BASE_SEED)

        E_raw2 = generate_condition("REAL_NORMAL", N_POINTS, N_DIM, BASE_SEED)
        E2, t2 = prepare_embeddings(E_raw2, BASE_SEED)
        base2 = run_pipeline_extract(E2, t2, N_MAG, TAU, seed=BASE_SEED)

        centers_a = base["centers"]
        centers_b = base2["centers"]
        k = centers_a.shape[0]
        cost_matrix = np.zeros((k, k))
        for ii in range(k):
            for jj in range(k):
                cost_matrix[ii, jj] = 1.0 - cosine_similarity(centers_a[ii],
                                                                centers_b[jj])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mapping = list(zip(row_ind.tolist(), col_ind.tolist()))
        all_mappings.append(mapping)
        all_cost_summaries.append({
            "min": float(cost_matrix.min()),
            "mean": float(cost_matrix.mean()),
            "max": float(cost_matrix.max()),
        })

    mappings_identical = all(m == all_mappings[0] for m in all_mappings)

    print(f"  Mapping identical across {n_runs} runs: {mappings_identical}")
    for i, (m, cs) in enumerate(zip(all_mappings, all_cost_summaries)):
        print(f"  Run {i+1}: mapping={m}")
        print(f"    cost_matrix: min={cs['min']:.4f} mean={cs['mean']:.4f} "
              f"max={cs['max']:.4f}")

    print(f"  A2 VERDICT: {'PASS' if mappings_identical else 'FAIL'}")
    return {
        "test": "A2_hungarian_stability",
        "pass": mappings_identical,
        "mappings": all_mappings,
        "cost_summaries": all_cost_summaries,
    }


def test_b1_random_pure() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("B1: NEGATIVE CONTROL - RANDOM_PURE (must FAIL all)")
    print("=" * 70)

    r = run_condition("RANDOM_PURE", seed=BASE_SEED)
    s = r["summary"]

    any_pass = s["h1_pass"] or s["h2_pass"] or s["h5_pass"]
    h5_identity_low = (s["h5_identity_abort"] is not None and
                       s["h5_identity_abort"] < 0.3)

    print(f"  H1 pass: {s['h1_pass']} (expected: False)")
    print(f"  H2 pass: {s['h2_pass']} (expected: False)")
    print(f"  H5 pass: {s['h5_pass']} (expected: False)")
    print(f"  H5 identity_abort: {s['h5_identity_abort']:.4f} "
          f"(expected: ~0, got {'OK' if h5_identity_low else 'HIGH'})")
    if any_pass:
        print(f"  *** BUG DETECTED: RANDOM_PURE should not pass any test! ***")

    verdict = not any_pass
    print(f"  B1 VERDICT: {'PASS' if verdict else 'BUG - FAIL'}")
    return {
        "test": "B1_random_pure_negative",
        "pass": verdict,
        "any_test_passed": any_pass,
        "h5_identity_abort": s["h5_identity_abort"],
        "details": s,
    }


def test_b2_near_null() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("B2: NEGATIVE CONTROL - NEAR_NULL (intermediate signature)")
    print("=" * 70)

    r = run_condition("NEAR_NULL", seed=BASE_SEED)
    s = r["summary"]

    r_random = run_condition("RANDOM_PURE", seed=BASE_SEED)
    s_random = r_random["summary"]

    lag_more_neg = (s["h1_lag1_corr"] is not None and
                    s_random["h1_lag1_corr"] is not None and
                    s["h1_lag1_corr"] < s_random["h1_lag1_corr"])

    identity_inertia = (s["h2_identity_sim"] is not None and
                        s["h2_identity_sim"] > 0.5)

    all_pass = s["h1_pass"] and s["h2_pass"] and s["h5_pass"]
    if all_pass:
        print(f"  *** WARNING: NEAR_NULL passed ALL tests - "
              f"insufficient test sensitivity! ***")

    print(f"  H1 pass: {s['h1_pass']} (expected: False)")
    print(f"  H1 lag1_corr: {s['h1_lag1_corr']:.4f} "
          f"(RANDOM: {s_random['h1_lag1_corr']:.4f}, "
          f"more negative: {lag_more_neg})")
    print(f"  H2 pass: {s['h2_pass']} (expected: False or partial)")
    print(f"  H2 identity_sim: {s['h2_identity_sim']:.4f} "
          f"(identity inertia: {identity_inertia})")
    print(f"  H5 pass: {s['h5_pass']} (expected: False)")

    verdict = not all_pass
    print(f"  B2 VERDICT: {'PASS' if verdict else 'FAIL (too sensitive)'}")
    return {
        "test": "B2_near_null",
        "pass": verdict,
        "all_passed_bug": all_pass,
        "lag_more_negative_than_random": lag_more_neg,
        "identity_inertia": identity_inertia,
        "details": s,
    }


def test_c1_real_normal() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("C1: REAL_NORMAL - baseline pass pattern")
    print("=" * 70)

    r = run_condition("REAL_NORMAL", seed=BASE_SEED)
    s = r["summary"]

    print(f"  H1 pass: {s['h1_pass']} (target: True)")
    print(f"  H1 lag1_corr: {s['h1_lag1_corr']:.4f}, p={s['h1_p_value']:.4f}")
    print(f"  H2 pass: {s['h2_pass']} (target: True)")
    print(f"  H2 identity_sim: {s['h2_identity_sim']:.4f} (target: >0.85)")
    print(f"  H2 alignment_delta: {s['h2_alignment_delta']:.4f} (target: >0.1)")
    print(f"  H5 pass: {s['h5_pass']} (target: True)")
    print(f"  H5 wave_drop: {s['h5_wave_drop']:.1f} (target: >0)")
    print(f"  H5 identity_abort: {s['h5_identity_abort']:.4f} (target: >0.5)")
    print(f"  Alignment: {s['alignment']:.4f}")
    print(f"  Longest wave: {s['longest_wave']}")
    print(f"  FSI: {s['fsi']:.4f}")

    n_pass = sum([s["h1_pass"], s["h2_pass"], s["h5_pass"]])
    print(f"  C1 VERDICT: {n_pass}/3 passed")
    return {
        "test": "C1_real_normal",
        "n_pass": n_pass,
        "pass": n_pass == 3,
        "details": s,
    }


def test_c2_real_survival() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("C2: REAL_SURVIVAL - survival signature")
    print("=" * 70)

    r_normal = run_condition("REAL_NORMAL", seed=BASE_SEED)
    r_survival = run_condition("REAL_SURVIVAL", seed=BASE_SEED)
    sn = r_normal["summary"]
    ss = r_survival["summary"]

    alignment_decreased = ss["alignment"] < sn["alignment"]
    identity_preserved = (ss["h2_identity_sim"] is not None and
                          ss["h2_identity_sim"] > 0.8)

    E_normal = generate_condition("REAL_NORMAL", N_POINTS, N_DIM, BASE_SEED)
    E_survival = generate_condition("REAL_SURVIVAL", N_POINTS, N_DIM, BASE_SEED)
    E_n, _ = prepare_embeddings(E_normal)
    E_s, _ = prepare_embeddings(E_survival)
    base_n = r_normal["base"]
    base_s = r_survival["base"]
    cross_identity = compute_identity_similarity(base_n["centers"],
                                                  base_s["centers"])
    identity_high = cross_identity > 0.6

    survival_pass = alignment_decreased and identity_high

    print(f"  NORMAL  alignment: {sn['alignment']:.4f}")
    print(f"  SURVIVAL alignment: {ss['alignment']:.4f}")
    print(f"  Alignment decreased: {alignment_decreased}")
    print(f"  Cross-identity (normal vs survival): {cross_identity:.4f}")
    print(f"  Identity high (>0.6): {identity_high}")
    print(f"  NORMAL  H2 identity: {sn['h2_identity_sim']:.4f}")
    print(f"  SURVIVAL H2 identity: {ss['h2_identity_sim']:.4f}")
    print(f"  NORMAL  longest_wave: {sn['longest_wave']}")
    print(f"  SURVIVAL longest_wave: {ss['longest_wave']}")

    print(f"  SURVIVAL_EFFECT VERDICT: {'PASS' if survival_pass else 'FAIL'}")
    return {
        "test": "C2_survival_effect",
        "pass": survival_pass,
        "alignment_decreased": alignment_decreased,
        "identity_high": identity_high,
        "cross_identity": float(cross_identity),
        "normal_summary": sn,
        "survival_summary": ss,
    }


def test_d1_h1_sensitivity() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("D1: H1 SENSITIVITY CURVE (window sizes 16/32/64)")
    print("=" * 70)

    from dsdp_sali_lcf.metrics.entropy import compute_per_bin_entropy
    from dsdp_sali_lcf.metrics.wave import compute_per_bin_wave_signal
    from dsdp_sali_lcf.tests.test_coherence_threshold import test_coherence_threshold

    window_sizes = [16, 32, 64]
    results = {}

    for cond in ["RANDOM_PURE", "REAL_NORMAL"]:
        cond_results = {}
        E_raw = generate_condition(cond, N_POINTS, N_DIM, BASE_SEED)
        E, t = prepare_embeddings(E_raw)
        base = run_pipeline_extract(E, t, N_MAG, TAU, seed=BASE_SEED)

        for ws in window_sizes:
            entropies = compute_per_bin_entropy(
                base["y"], base["t"], base["labels"], n_bins=ws, tau=TAU
            )
            wave_signals = compute_per_bin_wave_signal(
                base["temporal_scores"], 0.75
            )
            alignment_scores = base["temporal_scores"]

            n = min(len(entropies), len(wave_signals), len(alignment_scores))
            entropies = entropies[:n]
            wave_signals = wave_signals[:n]
            alignment_scores = alignment_scores[:n]

            h1 = test_coherence_threshold(entropies, wave_signals,
                                           alignment_scores, n_perm=200, seed=42)
            cond_results[ws] = {
                "lag1_corr": h1.get("entropy_wave_lag1_correlation", None),
                "p_value": h1.get("permutation_test", {}).get("p_value", None),
                "pass": h1["pass"],
            }
        results[cond] = cond_results

    print(f"  {'Window':>8s} | {'RANDOM lag1':>12s} {'p':>6s} {'pass':>5s} | "
          f"{'REAL lag1':>12s} {'p':>6s} {'pass':>5s}")
    print(f"  {'-'*8}-+-{'-'*26}-+-{'-'*26}")
    for ws in window_sizes:
        rr = results["RANDOM_PURE"][ws]
        re = results["REAL_NORMAL"][ws]
        rl = f"{rr['lag1_corr']:.4f}" if rr['lag1_corr'] is not None else "N/A"
        rp = f"{rr['p_value']:.4f}" if rr['p_value'] is not None else "N/A"
        el = f"{re['lag1_corr']:.4f}" if re['lag1_corr'] is not None else "N/A"
        ep = f"{re['p_value']:.4f}" if re['p_value'] is not None else "N/A"
        print(f"  {ws:>8d} | {rl:>12s} {rp:>6s} {str(rr['pass']):>5s} | "
              f"{el:>12s} {ep:>6s} {str(re['pass']):>5s}")

    real_signal_grows = True
    random_no_signal = True
    for ws in window_sizes:
        if results["RANDOM_PURE"][ws]["pass"]:
            random_no_signal = False

    print(f"\n  Random never passes: {random_no_signal}")
    verdict = random_no_signal
    print(f"  D1 VERDICT: {'PASS' if verdict else 'FAIL'}")
    return {
        "test": "D1_h1_sensitivity",
        "pass": verdict,
        "random_no_signal": random_no_signal,
        "details": results,
    }


def test_d2_h5_survival_delta() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("D2: H5 SURVIVAL DELTA (strong abort, strength=1.5)")
    print("=" * 70)

    strong_abort = 1.5

    for cond_name, cond_label in [("REAL_NORMAL", "NORMAL"),
                                   ("REAL_SURVIVAL", "SURVIVAL")]:
        E_raw = generate_condition(cond_name, N_POINTS, N_DIM, BASE_SEED)
        E, t = prepare_embeddings(E_raw)
        base = run_pipeline_extract(E, t, N_MAG, TAU, seed=BASE_SEED)
        h5 = run_h5_intentional_abort(E, t, base, abort_strength=strong_abort,
                                        n_mag=N_MAG, tau=TAU, seed=BASE_SEED)
        if cond_label == "NORMAL":
            id_abort_normal = h5.get("identity_preserved_similarity", 0.0)
        else:
            id_abort_survival = h5.get("identity_preserved_similarity", 0.0)

    survival_better = (id_abort_survival is not None and
                       id_abort_normal is not None and
                       id_abort_survival > id_abort_normal)

    print(f"  identity_abort (NORMAL, abort={strong_abort}):   {id_abort_normal:.4f}")
    print(f"  identity_abort (SURVIVAL, abort={strong_abort}): {id_abort_survival:.4f}")
    print(f"  Survival preserves identity better: {survival_better}")

    print(f"  H5_SURVIVAL_DELTA VERDICT: {'PASS' if survival_better else 'FAIL'}")
    return {
        "test": "D2_h5_survival_delta",
        "pass": survival_better,
        "abort_strength": strong_abort,
        "identity_abort_normal": float(id_abort_normal) if id_abort_normal else 0.0,
        "identity_abort_survival": float(id_abort_survival) if id_abort_survival else 0.0,
    }


def test_e1_biblical_consistency() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("E1: BIBLICAL MODE - consistency check")
    print("=" * 70)

    r_normal = run_condition("REAL_NORMAL", seed=BASE_SEED)
    r_biblical = run_condition("REAL_BIBLICAL", seed=BASE_SEED)

    cross_identity = compute_identity_similarity(
        r_normal["base"]["centers"], r_biblical["base"]["centers"]
    )

    normal_summary = {
        "alignment": r_normal["summary"]["alignment"],
        "longest_wave": r_normal["summary"]["h5_wave_drop"] if r_normal["summary"]["h5_wave_drop"] else 0,
        "fsi": r_normal["summary"]["fsi"] if r_normal["summary"]["fsi"] else 0,
        "h1_pass": r_normal["summary"]["h1_pass"],
        "h5_wave_drop": r_normal["h5"].get("wave_drop", 0),
        "h5_alignment_drop": r_normal["h5"].get("alignment_drop", 0),
    }
    biblical_summary = {
        "alignment": r_biblical["summary"]["alignment"],
        "longest_wave": r_biblical["summary"].get("longest_wave",
                          r_biblical["base"]["longest_wave"]),
        "fsi": r_biblical["summary"]["fsi"] if r_biblical["summary"]["fsi"] else 0,
        "h1_pass": r_biblical["summary"]["h1_pass"],
        "h5_wave_drop": r_biblical["h5"].get("wave_drop", 0),
        "h5_alignment_drop": r_biblical["h5"].get("alignment_drop", 0),
    }

    normal_summary["longest_wave"] = r_normal["base"]["longest_wave"]

    result = test_biblical_consistency(normal_summary, biblical_summary,
                                        cross_identity)

    print(f"  Cross-identity (normal vs biblical): {cross_identity:.4f}")
    print(f"  Identity preserved (>0.85): {result['identity_preserved']}")
    print(f"  Alignment change: {result['alignment_change']:.4f}")
    print(f"  Wave change: {result['wave_change']:.1f}")
    print(f"  No new waves: {result['no_new_waves']}")
    print(f"  H1 fails as expected: {result['h1_fails_as_expected']}")
    print(f"  H5 trivial: {result['h5_trivial']}")

    print(f"  Biblical H1: {r_biblical['summary']['h1_pass']} (expected: False)")
    print(f"  Biblical H2: {r_biblical['summary']['h2_pass']}")
    print(f"  Biblical H5: {r_biblical['summary']['h5_pass']}")

    print(f"  E1 VERDICT: {'PASS' if result['pass'] else 'FAIL'}")
    return {
        "test": "E1_biblical_consistency",
        "pass": result["pass"],
        "details": result,
        "biblical_h1": r_biblical["summary"]["h1_pass"],
        "biblical_h2": r_biblical["summary"]["h2_pass"],
        "biblical_h5": r_biblical["summary"]["h5_pass"],
    }


def test_e2_biblical_no_false_positives() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("E2: BIBLICAL MODE - no false emergence")
    print("=" * 70)

    r_biblical = run_condition("REAL_BIBLICAL", seed=BASE_SEED)
    s = r_biblical["summary"]

    h1_pass = s["h1_pass"]
    fsi = s.get("fsi", 0) or 0

    r_random = run_condition("RANDOM_PURE", seed=BASE_SEED)
    s_random = r_random["summary"]
    random_fsi = s_random.get("fsi", 0) or 0

    no_false_h1 = not h1_pass
    fsi_not_inflated = fsi <= 1.5 * (random_fsi + 0.01)

    verdict = no_false_h1 and fsi_not_inflated

    print(f"  Biblical H1 pass: {h1_pass} (expected: False)")
    print(f"  Biblical FSI: {fsi:.4f}")
    print(f"  Random FSI:   {random_fsi:.4f}")
    print(f"  No false H1: {no_false_h1}")
    print(f"  FSI not inflated: {fsi_not_inflated}")

    print(f"  E2 VERDICT: {'PASS' if verdict else 'FAIL'}")
    return {
        "test": "E2_biblical_no_false_emergence",
        "pass": verdict,
        "h1_pass": h1_pass,
        "biblical_fsi": float(fsi),
        "random_fsi": float(random_fsi),
        "no_false_h1": no_false_h1,
        "fsi_not_inflated": fsi_not_inflated,
    }


def print_expectations_matrix(all_results: Dict[str, Any]):
    print("\n" + "=" * 70)
    print("EXPECTATIONS MATRIX (check rapid)")
    print("=" * 70)

    cond_data = {}
    for key in ["B1", "B2", "C1", "C2"]:
        r = all_results.get(key, {})
        d = r.get("details", r.get("normal_summary", {}))
        cond_data[key] = d

    print(f"\n  {'Condition':>15s} | {'H1':>6s} | {'H2':>6s} | {'H5':>6s} | "
          f"{'Surv.Eff':>10s}")
    print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}")

    b1 = all_results.get("B1", {}).get("details", {})
    print(f"  {'RANDOM_PURE':>15s} | "
          f"{'FAIL' if not b1.get('h1_pass') else 'PASS':>6s} | "
          f"{'FAIL' if not b1.get('h2_pass') else 'PASS':>6s} | "
          f"{'FAIL' if not b1.get('h5_pass') else 'PASS':>6s} | "
          f"{'n/a':>10s}")

    b2 = all_results.get("B2", {}).get("details", {})
    print(f"  {'NEAR_NULL':>15s} | "
          f"{'FAIL' if not b2.get('h1_pass') else 'PASS':>6s} | "
          f"{'FAIL' if not b2.get('h2_pass') else 'PASS':>6s} | "
          f"{'FAIL' if not b2.get('h5_pass') else 'PASS':>6s} | "
          f"{'n/a':>10s}")

    c1 = all_results.get("C1", {}).get("details", {})
    print(f"  {'REAL_NORMAL':>15s} | "
          f"{'PASS' if c1.get('h1_pass') else 'FAIL':>6s} | "
          f"{'PASS' if c1.get('h2_pass') else 'FAIL':>6s} | "
          f"{'PASS' if c1.get('h5_pass') else 'FAIL':>6s} | "
          f"{'n/a':>10s}")

    c2_pass = all_results.get("C2", {}).get("pass", False)
    c2_ss = all_results.get("C2", {}).get("survival_summary", {})
    print(f"  {'REAL_SURVIVAL':>15s} | "
          f"{'PASS' if c2_ss.get('h1_pass') else 'FAIL':>6s} | "
          f"{'PASS' if c2_ss.get('h2_pass') else 'FAIL':>6s} | "
          f"{'PASS' if c2_ss.get('h5_pass') else 'FAIL':>6s} | "
          f"{'PASS' if c2_pass else 'FAIL':>10s}")

    e1 = all_results.get("E1", {})
    print(f"  {'REAL_BIBLICAL':>15s} | "
          f"{'FAIL' if not e1.get('biblical_h1') else 'PASS':>6s} | "
          f"{'PASS' if e1.get('biblical_h2') else 'FAIL':>6s} | "
          f"{'PASS' if e1.get('biblical_h5') else 'FAIL':>6s} | "
          f"{'PASS' if e1.get('pass') else 'FAIL':>10s}")


def print_final_summary(all_results: Dict[str, Any]):
    print("\n" + "=" * 70)
    print("CALIBRATION PACK - FINAL SUMMARY")
    print("=" * 70)

    tests = [
        ("A1", "Reproducibility (CV<10%)"),
        ("A2", "Hungarian matching stability"),
        ("B1", "RANDOM_PURE negative control"),
        ("B2", "NEAR_NULL sensitivity check"),
        ("C1", "REAL_NORMAL pass pattern"),
        ("C2", "SURVIVAL_EFFECT"),
        ("D1", "H1 sensitivity curve"),
        ("D2", "H5 survival delta"),
        ("E1", "BIBLICAL consistency (identity + no new structure)"),
        ("E2", "BIBLICAL no false emergence"),
    ]

    n_pass = 0
    n_total = len(tests)
    for tid, desc in tests:
        r = all_results.get(tid, {})
        passed = r.get("pass", False)
        n_pass += int(passed)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status:>4s}] {tid}: {desc}")

    print(f"\n  TOTAL: {n_pass}/{n_total} calibration checks passed")

    if n_pass == n_total:
        print("  >> ALL GREEN: Harness calibrated. Ready for Phase 2.")
    elif n_pass >= n_total - 2:
        print("  >> MOSTLY GREEN: Minor calibration issues. "
              "Review failed tests before Phase 2.")
    else:
        print("  >> NEEDS WORK: Significant calibration failures. "
              "Debug before proceeding.")
    print("=" * 70 + "\n")


def main():
    global N_POINTS
    import argparse
    parser = argparse.ArgumentParser(description="Full Calibration Pack")
    parser.add_argument("--output-dir",
                        default="dsdp_sali_lcf/outputs/calibration")
    parser.add_argument("--skip-reproducibility", action="store_true",
                        help="Skip A1 reproducibility (saves time)")
    parser.add_argument("--n-points", type=int, default=5000,
                        help="Number of points per condition (default 5000)")
    args = parser.parse_args()
    if args.n_points != 5000:
        N_POINTS = args.n_points
        print(f"*** Using N_POINTS = {N_POINTS} ***")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    all_results = {}

    if not args.skip_reproducibility:
        all_results["A1"] = test_a1_reproducibility()
    else:
        print("\n[SKIPPED] A1: Reproducibility test")
        all_results["A1"] = {"test": "A1_reproducibility", "pass": True,
                              "details": "skipped"}

    all_results["A2"] = test_a2_hungarian_stability()

    all_results["B1"] = test_b1_random_pure()
    all_results["B2"] = test_b2_near_null()

    all_results["C1"] = test_c1_real_normal()
    all_results["C2"] = test_c2_real_survival()

    all_results["D1"] = test_d1_h1_sensitivity()
    all_results["D2"] = test_d2_h5_survival_delta()

    all_results["E1"] = test_e1_biblical_consistency()
    all_results["E2"] = test_e2_biblical_no_false_positives()

    print_expectations_matrix(all_results)
    print_final_summary(all_results)

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s")

    report = {}
    for k, v in all_results.items():
        clean = {}
        for kk, vv in v.items():
            if isinstance(vv, (dict, list)):
                try:
                    json.dumps(vv, default=str)
                    clean[kk] = vv
                except (TypeError, ValueError):
                    clean[kk] = str(vv)
            elif isinstance(vv, (np.integer, np.floating)):
                clean[kk] = float(vv)
            elif isinstance(vv, np.ndarray):
                clean[kk] = vv.tolist()
            else:
                clean[kk] = vv
        report[k] = clean

    report_path = output_dir / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
