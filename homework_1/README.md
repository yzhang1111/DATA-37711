# DATA 37711 — Homework 1

Code for HW1. Framework: **JAX + optax** for Problem 5 (autodiff + `optax.sgd`).

```
src/
  data.py                  loaders, 80/20 split, standardization (fit on train only)
  sgd.py                   generic mini-batch SGD trainer for linear models
  p1_kelly_check.py        numeric check of the bet-size answers in Q1
  p4_residual_normality.py Q4: 5 regressors on ALE, QQ plots + normaltest/Shapiro
  p5_classification.py     Q5 classification: sklearn baseline, logistic, probit
  p5_regression.py         Q5 regression: OLS baseline, MSE+SGD, Student-t(5)+SGD
  make_tables.py           results/*.json -> LaTeX table rows (for the write-up)
data/       put ale.csv here (see data/README.md)
figures/    generated plots
results/    generated json + .tex tables
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
make p4      # needs ALE data (auto-download via ucimlrepo or data/ale.csv)
make p5      # classification + regression + tables
```

Seed is fixed (`SEED = 0` in `data.py`), so splits are reproducible.
