# Boundary Polymorphism Verification Report

## Method
Embeddings projected to 2D via PCA, density field constructed via KDE,
thresholded at tau in [0.3, 0.4, 0.5, 0.6, 0.7], boundary contours extracted.
Grid resolution: 128x128. Seeds: [42, 43, 99].

## Conditions
- **RANDOM_PURE**: poly_score=2/5
- **NEAR_NULL**: poly_score=3/5
- **REAL_NORMAL**: poly_score=2/5
- **REAL_SURVIVAL**: poly_score=4/5
- **REAL_BIBLICAL**: poly_score=2/5
- **AXIS_MODERATE**: poly_score=4/5
- **AXIS_STRONG**: poly_score=4/5
- **VIRUS_03**: poly_score=3/5
- **VIRUS_06**: poly_score=4/5

## Polymorphism Verdict: **CONFIRMED**

- Mean control topo_score: 1.8
- Mean REAL topo_score: 4.4
- Topo separation: 2.6
- RANDOM no component transitions: True
- RANDOM low Euler transitions: True

### Subsystem Analysis

- RANDOM_PURE topo: 1.5
- NEAR_NULL topo: 2.0
- Core REAL (normal/survival/biblical) mean topo: 3.5
- Extended (axis/virus variants) mean topo: 5.0
- Core REAL vs RANDOM separation: 2.0
- Extended vs RANDOM separation: 3.5

The structure boundary belongs to multiple geometric classes
depending on threshold and regime. RANDOM_PURE shows zero
component topology transitions across all seeds and thresholds.

### Statistical Analysis

- Control topo_score 95% CI: [1.50, 2.00]
- REAL topo_score 95% CI: [3.64, 5.07]
- Separation 95% CI: [1.79, 3.39]
- Mann-Whitney U: U=0.0, p=0.0275 (one-sided)
- Cohen's d: 2.61 (large)

Bootstrap CI: 10,000 resamples, BCa percentile method.
Mann-Whitney U: non-parametric test (H0: control >= REAL).
Cohen's d: pooled-SD standardized mean difference.

### Caveats

1. Results depend on 2D PCA projection. Topology in the original
   64D space may differ. PCA can collapse distinct structures.
2. KDE bandwidth (sigma=2.0) and grid resolution (128x128) are
   fixed. Sensitivity to these parameters not tested.
3. NEAR_NULL shows some transitions (topo=2.0), suggesting the
   boundary between 'unstructured' and 'structured' is gradual,
   not sharp. Core REAL separation from RANDOM is moderate.
4. Strongest polymorphism signal comes from extended conditions
   (axis-aligned, virus-infected) rather than base REAL conditions.


## Key Metrics

| Condition | Comp.Trans | Euler.Trans | Ecc.Range | Curv.Tail | Poly | Topo |
|---|---|---|---|---|---|---|
| RANDOM_PURE | 0 | 0 | 0.356 | 1.12 | 2/5 | 1.5 |
| NEAR_NULL | 1 | 1 | 0.488 | 1.07 | 3/5 | 2.0 |
| REAL_NORMAL | 2 | 2 | 0.376 | 0.96 | 2/5 | 3.0 |
| REAL_SURVIVAL | 2 | 2 | 0.631 | 1.86 | 4/5 | 4.5 |
| REAL_BIBLICAL | 2 | 2 | 0.376 | 0.96 | 2/5 | 3.0 |
| AXIS_MODERATE | 2 | 4 | 0.435 | 2.00 | 4/5 | 5.5 |
| AXIS_STRONG | 1 | 5 | 0.648 | 3.84 | 4/5 | 4.0 |
| VIRUS_03 | 3 | 3 | 0.731 | 0.87 | 3/5 | 5.0 |
| VIRUS_06 | 4 | 4 | 0.345 | 2.75 | 4/5 | 5.5 |

See `metrics.csv` for full per-tau data and `contours_*.png` for visualizations.
