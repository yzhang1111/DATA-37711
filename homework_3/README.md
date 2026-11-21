# DATA 37711 — Homework 3

Problems 1 and 2(a)–(b) are written answers (see the LaTeX write-up). This folder has the code for
the empirical parts. All splits and models use fixed seeds, so results are reproducible.

```
src/
  p2_wine.py       Q2(c): 70/10/20 split of white wine; grid search (600 settings each) for
                   KNeighborsRegressor and RandomForestRegressor, select on validation MSE,
                   report test MSE, compare validation vs test risk (incl. 2000 re-splits of the
                   held-out data to measure the optimism caused by selection)
  p3_boosting.py   Q3: GradientBoostingClassifier with 1, 10, 100, 300, 1000 trees on dset0 and dset1
data/              winequality-white.csv, dset0.csv, dset1.csv
figures/           generated plots
results/           grid-search tables (csv), held-out predictions (npy), summaries (json)
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all          # or: make p2 / make p3
```

`p2_wine.py knn` or `p2_wine.py rf` runs a single model class; the random-forest grid is the slow
part (a few minutes on a laptop).
