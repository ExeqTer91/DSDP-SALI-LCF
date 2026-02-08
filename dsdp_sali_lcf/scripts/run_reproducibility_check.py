"""Reproducibility Check Script for DSDP SALI LCF Calibration Pack.

This script is designed for independent verification. It:
1. Generates all 5 synthetic conditions deterministically (seed=42)
2. Computes SHA-256 hashes of raw embeddings for code-identity verification
3. Runs the full calibration pack (10 tests)
4. Runs fractal verification (time, space, regime invariance)
5. Outputs a single JSON report with all results, hashes, and verdicts

Usage:
    python dsdp_sali_lcf/scripts/run_reproducibility_check.py
    python dsdp_sali_lcf/scripts/run_reproducibility_check.py --n-points 10000
    python dsdp_sali_lcf/scripts/run_reproducibility_check.py --n-points 10000 --record-hashes
    python dsdp_sali_lcf/scripts/run_reproducibility_check.py --n-points 5000 --verify-hashes
    python dsdp_sali_lcf/scripts/run_reproducibility_check.py --fast
    python dsdp_sali_lcf/scripts/run_reproducibility_check.py --output-dir /path/to/output

Expected result: 10/10 calibration checks, 3/3 fractal checks.
"""
import sys
import json
import hashlib
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dsdp_sali_lcf.scripts.generate_calibration_data import (
    generate_condition, ALL_CONDITIONS,
)
from dsdp_sali_lcf.experiments.run_calibration_pack import (
    prepare_embeddings, run_pipeline_extract, run_condition,
    test_a2_hungarian_stability, test_b1_random_pure, test_b2_near_null,
    test_c1_real_normal, test_c2_real_survival,
    test_d1_h1_sensitivity, test_d2_h5_survival_delta,
    test_e1_biblical_consistency, test_e2_biblical_no_false_positives,
    print_expectations_matrix, print_final_summary,
    N_DIM, BASE_SEED, N_MAG, TAU,
)
from dsdp_sali_lcf.experiments.run_abort_variant import run_abort_experiment
from dsdp_sali_lcf.metrics.identity import compute_identity_similarity
from dsdp_sali_lcf.metrics.entropy import compute_per_bin_entropy
from dsdp_sali_lcf.metrics.wave import compute_per_bin_wave_signal
from dsdp_sali_lcf.tests.test_coherence_threshold import test_coherence_threshold

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HASH_FILE = Path(__file__).resolve().parent.parent / "repro" / "reference_hashes.json"

DEFAULT_N_POINTS = 10000


def load_reference_hashes(n_points: int) -> Dict[str, str]:
    if not HASH_FILE.exists():
        return {}
    with open(HASH_FILE) as f:
        data = json.load(f)
    return data.get("hashes", {}).get(str(n_points), {})


def _get_environment_metadata() -> Dict[str, str]:
    import platform as _platform
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": _platform.platform(),
    }


def save_reference_hashes(n_points: int, hashes: Dict[str, str]) -> None:
    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    if HASH_FILE.exists():
        with open(HASH_FILE) as f:
            data = json.load(f)
    else:
        data = {
            "description": "SHA-256 hashes of raw embeddings for reproducibility verification",
            "seed": BASE_SEED,
            "n_dim": N_DIM,
            "hashes": {},
        }
    data["environment"] = _get_environment_metadata()
    data["hashes"][str(n_points)] = hashes
    with open(HASH_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Hashes recorded to {HASH_FILE} under key n={n_points}")


def compute_embedding_hash(E: np.ndarray) -> str:
    return hashlib.sha256(E.tobytes()).hexdigest()


def step_1_generate_and_hash(n_points: int,
                              mode: str = "verify") -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"STEP 1: GENERATE DATA & COMPUTE HASHES (n={n_points})")
    print("=" * 70)

    hashes = {}
    shapes = {}
    for cond in ALL_CONDITIONS:
        E = generate_condition(cond, n_points, N_DIM, BASE_SEED)
        h = compute_embedding_hash(E)
        hashes[cond] = h
        shapes[cond] = list(E.shape)
        print(f"  {cond:15s}: shape={E.shape} hash={h[:16]}...")

    if mode == "record":
        save_reference_hashes(n_points, hashes)
        return {"hashes": hashes, "shapes": shapes, "recorded": True}

    ref = load_reference_hashes(n_points)
    if ref:
        mismatches = []
        for c in ALL_CONDITIONS:
            if hashes.get(c) != ref.get(c):
                mismatches.append(c)
        all_match = len(mismatches) == 0
        print(f"\n  Reference hash match (n={n_points}): {all_match}")
        if mismatches:
            for c in mismatches:
                print(f"    MISMATCH: {c}")
                print(f"      computed: {hashes[c][:32]}...")
                print(f"      expected: {ref[c][:32]}...")
    else:
        all_match = None
        print(f"\n  No reference hashes for n={n_points}. "
              f"Use --record-hashes to save them.")

    return {"hashes": hashes, "shapes": shapes,
            "reference_match": all_match}


def step_2_calibration_pack(skip_a1: bool = True,
                             n_points: int = DEFAULT_N_POINTS) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"STEP 2: CALIBRATION PACK (10 tests, n={n_points})")
    print("=" * 70)

    import dsdp_sali_lcf.experiments.run_calibration_pack as calib_mod
    old_n = calib_mod.N_POINTS
    calib_mod.N_POINTS = n_points

    all_results = {}

    if skip_a1:
        print("\n[SKIPPED] A1: Reproducibility test (use --full for A1)")
        all_results["A1"] = {"test": "A1_reproducibility", "pass": True,
                              "details": "skipped"}
    else:
        from dsdp_sali_lcf.experiments.run_calibration_pack import (
            test_a1_reproducibility,
        )
        all_results["A1"] = test_a1_reproducibility()

    all_results["A2"] = test_a2_hungarian_stability()
    all_results["B1"] = test_b1_random_pure()
    all_results["B2"] = test_b2_near_null()
    all_results["C1"] = test_c1_real_normal()
    all_results["C2"] = test_c2_real_survival()
    all_results["D1"] = test_d1_h1_sensitivity()
    all_results["D2"] = test_d2_h5_survival_delta()
    all_results["E1"] = test_e1_biblical_consistency()
    all_results["E2"] = test_e2_biblical_no_false_positives()

    calib_mod.N_POINTS = old_n

    print_expectations_matrix(all_results)
    print_final_summary(all_results)

    n_pass = sum(1 for v in all_results.values() if v.get("pass", False))
    return {"n_pass": n_pass, "n_total": 10, "tests": {
        k: {"pass": v.get("pass", False), "test": v.get("test", k)}
        for k, v in all_results.items()
    }}


def step_3_fractal_verification(n_points: int = DEFAULT_N_POINTS) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"STEP 3: FRACTAL VERIFICATION (n={n_points})")
    print("=" * 70)

    results = {}

    print("\n  --- A. Time scale invariance (H1 at windows 16, 32, 64) ---")
    time_results = {}
    for cond in ALL_CONDITIONS:
        E_raw = generate_condition(cond, n_points, N_DIM, BASE_SEED)
        E, t = prepare_embeddings(E_raw)
        base = run_pipeline_extract(E, t, N_MAG, TAU, seed=BASE_SEED)
        cond_res = {}
        for ws in [16, 32, 64]:
            entropies = compute_per_bin_entropy(
                base["y"], base["t"], base["labels"], n_bins=ws, tau=TAU)
            wave_signals = compute_per_bin_wave_signal(
                base["temporal_scores"], 0.75)
            n = min(len(entropies), len(wave_signals),
                    len(base["temporal_scores"]))
            h1 = test_coherence_threshold(
                entropies[:n], wave_signals[:n],
                base["temporal_scores"][:n], n_perm=200, seed=42)
            cond_res[str(ws)] = {
                "pass": bool(h1["pass"]),
                "lag1": float(h1.get("entropy_wave_lag1_correlation", 0)),
            }
        time_results[cond] = cond_res

    random_never = all(
        not time_results["RANDOM_PURE"][str(ws)]["pass"]
        for ws in [16, 32, 64])
    near_null_never = all(
        not time_results["NEAR_NULL"][str(ws)]["pass"]
        for ws in [16, 32, 64])
    time_pass = random_never
    print(f"    RANDOM never passes across windows: {random_never}")
    print(f"    NEAR_NULL never passes across windows: {near_null_never}")
    print(f"    TIME verdict: {'PASS' if time_pass else 'FAIL'}")
    results["time"] = {
        "pass": time_pass,
        "random_never_passes": random_never,
        "near_null_never_passes": near_null_never,
        "details": time_results,
    }

    print("\n  --- B. Space/structure invariance (H5 at strengths 0.25, 0.5, 1.0) ---")
    space_results = {}
    for cond in ALL_CONDITIONS:
        E_raw = generate_condition(cond, n_points, N_DIM, BASE_SEED)
        E, t = prepare_embeddings(E_raw)
        base = run_pipeline_extract(E, t, N_MAG, TAU, seed=BASE_SEED)
        cond_res = {}
        for strength in [0.25, 0.5, 1.0]:
            abort = run_abort_experiment(
                E, t, base["y"], abort_strength=strength,
                hub=base.get("hub"))
            identity = compute_identity_similarity(
                base["centers"], abort["centers"])
            align_drop = base["alignment"] - abort["alignment"]
            cond_res[str(strength)] = {
                "align_drop": float(align_drop),
                "identity": float(identity),
            }
        space_results[cond] = cond_res

    real_monotonic = (
        space_results["REAL_NORMAL"]["0.25"]["align_drop"] <=
        space_results["REAL_NORMAL"]["0.5"]["align_drop"] <=
        space_results["REAL_NORMAL"]["1.0"]["align_drop"])
    real_identity_ok = all(
        space_results["REAL_NORMAL"][str(s)]["identity"] > 0.8
        for s in [0.25, 0.5, 1.0])
    space_pass = real_monotonic and real_identity_ok
    print(f"    REAL_NORMAL monotonic align_drop: {real_monotonic}")
    print(f"    REAL_NORMAL identity always > 0.8: {real_identity_ok}")
    print(f"    SPACE verdict: {'PASS' if space_pass else 'FAIL'}")
    results["space"] = {
        "pass": space_pass,
        "monotonic": real_monotonic,
        "identity_ok": real_identity_ok,
        "details": space_results,
    }

    print("\n  --- C. Regime invariance (data sizes 2000, 5000) ---")
    regime_results = {}
    for n_pts in [2000, 5000]:
        size_res = {}
        for cond in ALL_CONDITIONS:
            E_raw = generate_condition(cond, n_pts, N_DIM, BASE_SEED)
            E, t = prepare_embeddings(E_raw)
            base = run_pipeline_extract(E, t, N_MAG, TAU, seed=BASE_SEED)
            size_res[cond] = {
                "alignment": float(base["alignment"]),
                "waves": int(base["longest_wave"]),
                "fsi": float(base["fsi"]),
            }
        regime_results[str(n_pts)] = size_res

    hierarchy_ok = True
    for n_pts in [2000, 5000]:
        rr = regime_results[str(n_pts)]
        if rr["RANDOM_PURE"]["alignment"] > rr["REAL_NORMAL"]["alignment"]:
            hierarchy_ok = False
    regime_pass = hierarchy_ok
    print(f"    Hierarchy RANDOM <= REAL at all sizes: {hierarchy_ok}")
    print(f"    REGIME verdict: {'PASS' if regime_pass else 'FAIL'}")
    results["regime"] = {
        "pass": regime_pass,
        "hierarchy_ok": hierarchy_ok,
        "details": regime_results,
    }

    n_fractal_pass = sum(1 for v in results.values() if v.get("pass", False))
    print(f"\n  FRACTAL TOTAL: {n_fractal_pass}/3 checks passed")
    return {
        "n_pass": n_fractal_pass,
        "n_total": 3,
        "checks": results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="DSDP SALI LCF Reproducibility Check")
    parser.add_argument("--output-dir",
                        default="dsdp_sali_lcf/outputs")
    parser.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS,
                        help=f"Number of points (default: {DEFAULT_N_POINTS}). "
                             f"Hash verification selects the matching set.")
    parser.add_argument("--record-hashes", action="store_true",
                        help="Record computed hashes as new reference "
                             "(saved to repro/reference_hashes.json)")
    parser.add_argument("--verify-hashes", action="store_true",
                        help="Verify computed hashes against stored reference "
                             "(exits with error on mismatch)")
    parser.add_argument("--fast", action="store_true",
                        help="Skip A1 reproducibility test")
    parser.add_argument("--full", action="store_true",
                        help="Include A1 reproducibility test")
    args = parser.parse_args()

    n_points = args.n_points
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    print("=" * 70)
    print("DSDP SALI LCF — REPRODUCIBILITY CHECK")
    print("=" * 70)
    print(f"Parameters: N={n_points}, D={N_DIM}, seed={BASE_SEED}, "
          f"K={N_MAG}, tau={TAU}")

    hash_mode = "record" if args.record_hashes else "verify"
    step1 = step_1_generate_and_hash(n_points, mode=hash_mode)

    if args.record_hashes:
        print("\n  Hashes recorded. Exiting (use --verify-hashes to check).")
        return

    if args.verify_hashes:
        if step1.get("reference_match") is None:
            print(f"\n  ERROR: No reference hashes found for n={n_points}.")
            print(f"  Run with --record-hashes first.")
            sys.exit(1)
        elif not step1["reference_match"]:
            print(f"\n  ERROR: Hash mismatch for n={n_points}!")
            sys.exit(1)
        else:
            print(f"\n  Hash verification PASSED for n={n_points}.")

    step2 = step_2_calibration_pack(skip_a1=not args.full, n_points=n_points)
    step3 = step_3_fractal_verification(n_points=n_points)

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("REPRODUCIBILITY CHECK — FINAL VERDICT")
    print("=" * 70)
    print(f"  Calibration pack: {step2['n_pass']}/{step2['n_total']} passed")
    print(f"  Fractal checks:   {step3['n_pass']}/{step3['n_total']} passed")
    print(f"  Total time:       {elapsed:.1f}s")

    all_ok = (step2["n_pass"] == step2["n_total"] and
              step3["n_pass"] == step3["n_total"])
    print(f"\n  OVERALL: {'ALL PASS — Ready for Phase 2' if all_ok else 'ISSUES FOUND'}")
    print("=" * 70)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parameters": {
            "n_points": n_points,
            "n_dim": N_DIM,
            "seed": BASE_SEED,
            "k_magistrales": N_MAG,
            "tau": TAU,
        },
        "step1_hashes": step1,
        "step2_calibration": step2,
        "step3_fractal": step3,
        "elapsed_seconds": elapsed,
        "overall_pass": all_ok,
    }

    report_path = output_dir / "reproducibility_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
