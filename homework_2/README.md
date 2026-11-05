# DATA 37711 — Homework 2

Neural networks are written in **JAX + optax** (the assignment allows PyTorch or JAX).
All random splits and initializations use fixed seeds, so results are reproducible.

```
src/
  common.py        paths, small JAX MLP, Adam trainer with early stopping on validation loss
  p1_taxi.py       Q1: heteroscedastic Gaussian MLP (mean + std) on NYC taxi durations, baselines,
                   P(duration < 45 min) with Brier / log-loss / accuracy@0.5
  p2_retail.py     Q2: cardinality, out-of-fold target encoding + gradient boosting,
                   embedding MLP, embeddings + target encoding, error by item frequency
  p3_rks_mlp.py    Q3: Random Kitchen Sinks + ridge (sweep sigma) vs 1-hidden-layer MLP (sweep width),
                   test-set predictions, fitted surfaces
  dataloader.py    provided starter loader (unchanged)
data/              datasets from Canvas
figures/           generated plots
results/           metrics (json) and LaTeX table rows
predictions/       rks_test_predictions.csv, mlp_test_predictions.csv (Q3d, same row order as test.csv)
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all        # or: make p1 / make p2 / make p3   (each takes ~1-3 min on a laptop CPU)
```
