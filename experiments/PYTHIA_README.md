# Pythia Temporal Emergence Experiment

## What This Does

Tests whether emergence metrics (H1 temporal coherence, topological separation, 
lattice alignment) develop during neural network training, using EleutherAI's 
Pythia checkpoints.

**Key insight:** Pythia provides public checkpoints at regular intervals from 
random initialization (step 0) to fully trained (step 143,000). We don't need 
to train anything — just extract hidden states at each checkpoint and measure.

## Predictions (pre-registered)

1. **H1**: Transitions from FAIL → PASS as training progresses
2. **topo_sep**: Increases monotonically (more topological structure)
3. **alignment**: Increases (representations become lattice-compatible)

If confirmed: paper gets a "prediction confirmed on real neural network" section.
If falsified: reported as null result.

## Requirements

```bash
pip install torch transformers numpy scipy scikit-learn
```

## Hardware

| Model | VRAM | Time (est.) | Where |
|-------|------|-------------|-------|
| pythia-70m | <1GB | ~30 min | Replit CPU ok |
| pythia-160m | ~2GB | ~1 hr | Replit GPU or RunPod |
| pythia-410m | ~4GB | ~2 hrs | RunPod |

**Recommendation:** Start with 70m on Replit. If it works, replicate on 160m.

## Run

```bash
python pythia_temporal_emergence.py
```

Results saved to `results/` directory:
- `pythia_emergence_results.json` — full results (saved incrementally)
- `pythia_emergence_plot_data.json` — plot-ready data

## Configuration

Edit the CONFIG dict at the top of the script:
- `model_name`: Which Pythia model (default: 70m)
- `checkpoints`: Which training steps to sample (15 points spanning 0-143k)
- `extract_layer`: Which layer's hidden states (default: layer 3 of 6)
- `n_sentences`: Corpus size (default: 2000)

## If Results Are Positive

These go into the Nature manuscript as:
- New Results subsection: "Temporal Coherence During Neural Network Training"
- One figure showing metrics vs training step
- Extended Data Table 9

## If Results Are Null

Report in Limitations:
"H1 temporal coherence was not detected in Pythia training checkpoints, 
suggesting that the temporal metric requires iterative constraint dynamics 
distinct from gradient-based optimization."
