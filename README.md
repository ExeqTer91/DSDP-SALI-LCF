# DSDP SALI LCF

**Discrete Structure Detection Pipeline** -- operationalizes "morphic" as latent structure detectability rather than an ontological field.

Quantifies persistence (temporal waves), coherence (FSI), and predictive discrete structure (next-band prediction). Validation relies strictly on conservative nulls, ablations, perturbations, and kill-switch rules.

---

## Calibration Results (n=10,000, seed=42)

| Test | Description | Status |
|------|-------------|--------|
| A1 | Reproducibility (bit-exact across runs) | PASS |
| A2 | Hungarian label stability | PASS |
| B1 | RANDOM_PURE fails all hypotheses | PASS |
| B2 | NEAR_NULL produces no false positives | PASS |
| C1 | REAL_NORMAL passes H1 + H2 + H5 (3/3) | PASS |
| C2 | Cross-identity similarity > 0.85 | PASS (0.986) |
| D1 | RANDOM never passes H1 at any window size | PASS |
| D2 | SURVIVAL resists abort better than NORMAL | PASS |
| E1 | BIBLICAL: H1 fails, identity preserved | PASS (id=1.0) |
| E2 | BIBLICAL: no false wave emergence | PASS |

**H1 criterion**: conservative conjunction -- requires both p < 0.05 (permutation null) AND |lag-1 correlation| > 0.2 (effect size).

---

## Hypotheses Tested

| Hypothesis | Description | Criterion |
|------------|-------------|-----------|
| H1 | Coherence Threshold | Entropy-wave lag-1 anticorrelation with p < 0.05 AND \|r\| > 0.2 |
| H2 | Peripheral Coupling | Peripheral alignment drops monotonically with abort strength |
| H5 | Intentional Abort | Wave count drops under anti-axial perturbation |

---

## Project Structure

```
dsdp_sali_lcf/
  config.yaml                    # Pipeline parameters
  METHODS.md                     # Full methods document (Nature-grade)
  CALIBRATION_REPORT.md          # Calibration summary

  src/                           # Core pipeline modules
    data_loader.py               #   Load, validate, normalize embeddings
    geometry.py                  #   Anchor pairs, hubs, magistrales (spherical k-means)
    lattice.py                   #   Log-space lattice scoring (dual grid + coupler)
    temporal.py                  #   S(t) scores, wave persistence, FSI
    nulls.py                     #   Null models (uniform, gaussian, spacing) + Holm correction
    ablations.py                 #   Seed, coupler, anchor, temporal ablations
    perturbation.py              #   Perturbation curves, tipping points
    plots.py                     #   Visualization

  metrics/                       # Metric computation
    entropy.py                   #   Per-bin residual, alignment, magistrale entropy
    wave.py                      #   Wave profiles, per-bin wave signals
    identity.py                  #   Identity vectors (Hungarian alignment + coupling)

  tests/                         # Hypothesis test implementations
    test_coherence_threshold.py  #   H1: entropy-wave lag structure
    test_peripheral_coupling.py  #   H2: peripheral alignment under abort
    test_intentional_abort.py    #   H5: wave disruption under anti-axial attack
    test_biblical_mode.py        #   Biblical mode consistency checks

  experiments/                   # Experiment runners
    run_calibration_pack.py      #   10-test calibration suite
    run_phase1.py                #   Phase 1 hypothesis testing
    run_single_agent.py          #   Single-condition pipeline run
    run_real_embeddings.py       #   BERT embedding validation
    run_abort_variant.py         #   Abort experiment variants
    verify_*.py                  #   Theoretical verification suites

  scripts/                       # Utility scripts
    generate_calibration_data.py #   Synthetic data generation
    run_reproducibility_check.py #   Full reproducibility verification
    generate_final_pack.py       #   Consolidate results

  repro/                         # Reproducibility
    reference_hashes.json        #   Versioned SHA-256 hashes (n=5k + n=10k)

  data/                          # Embedding data (.npz)
  outputs/                       # Results, reports, plots
```

---

## Quick Start

```bash
# Install dependencies
pip install numpy scipy scikit-learn pyyaml

# Run calibration pack (primary, n=10,000)
python dsdp_sali_lcf/experiments/run_calibration_pack.py --skip-reproducibility --n-points 10000

# Verify reproducibility (hash check + calibration + fractal)
python dsdp_sali_lcf/scripts/run_reproducibility_check.py --n-points 10000 --verify-hashes

# Run Phase 1 hypotheses only
python dsdp_sali_lcf/experiments/run_phase1.py

# Run BERT embedding validation (requires sentence-transformers + onnxruntime)
python dsdp_sali_lcf/experiments/run_real_embeddings.py
```

---

## Reproducibility

Reference SHA-256 hashes are stored in `repro/reference_hashes.json` with version keys by sample size (n=5,000 and n=10,000). The reproducibility script supports:

| Flag | Behavior |
|------|----------|
| `--record-hashes` | Compute and save hashes for the given `--n-points` |
| `--verify-hashes` | Compare computed hashes against stored reference |
| *(default)* | Verify if reference exists, warn if not |

Environment metadata (git commit, Python/NumPy versions, platform) is recorded alongside hashes for full traceability.

---

## Real Neural Network Validation (BERT)

Tested with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) on 5,000 topically-ordered scientific English sentences:

- **Spatial structure confirmed**: alignment 0.2181 vs 0.0000 (random), FSI 188x
- **H1 correctly fails**: p=0.976 on static embeddings (no temporal dynamics to detect)
- **No false positives** on any null condition
- **Identity preserved**: 0.877 similarity

This demonstrates the pipeline detects genuine spatial structure in real neural network representations while correctly rejecting temporal hypotheses when temporal dynamics are absent.

---

## Key Design Choices

- **Conservative H1 criterion**: Requires BOTH statistical significance (p < 0.05) AND meaningful effect size (|r| > 0.2), eliminating false positives
- **n=10,000 primary**: Increased from n=5,000 for robust significance (H1 p improved from 0.064 to 0.036)
- **5 synthetic conditions**: RANDOM_PURE, NEAR_NULL, REAL_NORMAL, REAL_SURVIVAL, REAL_BIBLICAL
- **Anti-contamination**: "Echo < Fresh" as outcome, not evidence for non-local transfer

---

## License

This project is provided for academic and research purposes.
