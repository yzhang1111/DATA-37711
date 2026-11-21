"""Problem 2(c): hyperparameter selection with a train / validation / test split on white wine.

Split 70 / 10 / 20 (seed 0). Two model classes: RandomForestRegressor and KNeighborsRegressor.
For each class, grid search over more parameter sets than there are validation points:
fit on train, score MSE on validation, pick the best, report MSE on test.

Usage:  python p2_wine.py            (runs both model classes, caches per-class results)
        python p2_wine.py knn | rf   (runs one class)
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, RES = ROOT / "data", ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
SEED = 0


def load_split():
    df = pd.read_csv(DATA / "winequality-white.csv", sep=";")
    X, y = df.drop(columns="quality").to_numpy(float), df["quality"].to_numpy(float)
    perm = np.random.default_rng(SEED).permutation(len(y))
    n_tr, n_va = int(0.7 * len(y)), int(0.1 * len(y))
    idx = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    return [(X[i], y[i]) for i in idx]


# ------------------------------------------------------------------ grids
KNN_GRID = {
    "n_neighbors": list(range(1, 51)),            # 50
    "weights": ["uniform", "distance"],           # 2
    "metric": ["manhattan", "euclidean", "chebyshev"],  # 3
    "scale": [True, False],                       # 2   -> 600 parameter sets
}
RF_GRID = {
    "n_estimators": [50, 150],                    # 2
    "max_depth": [None, 4, 8, 12, 16, 24],        # 6
    "max_features": [0.2, 0.4, 0.6, 0.8, 1.0],    # 5
    "min_samples_leaf": [1, 2, 4, 8, 16],         # 5
    "bootstrap": [True, False],                   # 2   -> 600 parameter sets
}


def make_model(kind, p):
    if kind == "knn":
        knn = KNeighborsRegressor(n_neighbors=p["n_neighbors"], weights=p["weights"], metric=p["metric"])
        return make_pipeline(StandardScaler(), knn) if p["scale"] else knn
    return RandomForestRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                 max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"],
                                 bootstrap=p["bootstrap"], random_state=SEED, n_jobs=1)


def evaluate(kind, p, splits):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    m = make_model(kind, p).fit(Xtr, ytr)
    pva, pte = m.predict(Xva), m.predict(Xte)
    row = {**p, "train_mse": mean_squared_error(ytr, m.predict(Xtr)),
           "val_mse": mean_squared_error(yva, pva), "test_mse": mean_squared_error(yte, pte)}
    return row, np.concatenate([pva, pte]).astype(np.float32)   # held-out predictions, kept for resampling


def run(kind, splits, batch=40):
    """Evaluate every parameter set; partial results are cached so an interrupted run can resume."""
    grid = KNN_GRID if kind == "knn" else RF_GRID
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    path, part = RES / f"p2_grid_{kind}.csv", RES / f"p2_grid_{kind}.partial.csv"
    ppath, ppart = RES / f"p2_preds_{kind}.npy", RES / f"p2_preds_{kind}.partial.npy"
    done = pd.read_csv(part).to_dict("records") if part.exists() else []
    preds = list(np.load(ppart)) if done else []
    print(f"{kind}: {len(combos)} parameter sets (validation set has {len(splits[1][1])} points); "
          f"{len(done)} cached")
    t, budget = time.time(), float(os.environ.get("TIME_BUDGET", "inf"))
    while len(done) < len(combos):
        if time.time() - t > budget:
            print(f"  time budget reached at {len(done)}/{len(combos)}; rerun to continue")
            return None
        todo = combos[len(done):len(done) + batch]
        res = Parallel(n_jobs=-1)(delayed(evaluate)(kind, p, splits) for p in todo)
        done += [r for r, _ in res]; preds += [q for _, q in res]
        pd.DataFrame(done).to_csv(part, index=False); np.save(ppart, np.array(preds))
    print(f"  done ({time.time() - t:.0f}s this run)")
    df = pd.DataFrame(done)
    df.to_csv(path, index=False); np.save(ppath, np.array(preds))
    part.unlink(missing_ok=True); ppart.unlink(missing_ok=True)
    return df


def selection_bias(kind, splits, n_resample=2000):
    """Re-partition the 30% held-out data into validation/test many times (same trained models).

    For each random partition: select the setting with the lowest validation MSE and record
    (min validation MSE) and (test MSE of that setting). For any *fixed* setting, validation and test
    MSE have the same expectation, so the average difference measures the optimism caused by selection.
    """
    P = np.load(RES / f"p2_preds_{kind}.npy")                     # (n_settings, n_val + n_test)
    y = np.concatenate([splits[1][1], splits[2][1]])
    sq = (P - y) ** 2
    n_val = len(splits[1][1])
    rng = np.random.default_rng(SEED)
    vmin, tsel, tbest, fixed_gap = [], [], [], []
    for _ in range(n_resample):
        perm = rng.permutation(len(y)); v, t = perm[:n_val], perm[n_val:]
        val, test = sq[:, v].mean(1), sq[:, t].mean(1)
        j = val.argmin()
        vmin.append(val[j]); tsel.append(test[j]); tbest.append(test.min())
        fixed_gap.append(test[0] - val[0])
    vmin, tsel, tbest = map(np.array, (vmin, tsel, tbest))
    return {"mean_min_val_mse": float(vmin.mean()), "mean_test_mse_of_selected": float(tsel.mean()),
            "mean_optimism": float((tsel - vmin).mean()), "sd_optimism": float((tsel - vmin).std()),
            "frac_val_below_test": float(np.mean(vmin < tsel)),
            "mean_regret_vs_oracle": float((tsel - tbest).mean()),
            "fixed_setting_mean_gap": float(np.mean(fixed_gap)), "n_resample": n_resample}


def summarize(splits):
    n_val = len(splits[1][1])
    out, fig, ax = {}, *plt.subplots(1, 2, figsize=(11, 4.3))
    names = {"knn": "KNeighborsRegressor", "rf": "RandomForestRegressor"}
    for a, kind in zip(ax, ["knn", "rf"]):
        df = pd.read_csv(RES / f"p2_grid_{kind}.csv")
        best = df.loc[df["val_mse"].idxmin()]
        oracle = df.loc[df["test_mse"].idxmin()]
        params = {k: (None if pd.isna(best[k]) else best[k]) for k in (KNN_GRID if kind == "knn" else RF_GRID)}
        out[kind] = {
            "n_param_sets": int(len(df)), "n_val": n_val,
            "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()},
            "best_val_mse": float(best["val_mse"]), "test_mse_of_selected": float(best["test_mse"]),
            "train_mse_of_selected": float(best["train_mse"]),
            "oracle_best_test_mse": float(oracle["test_mse"]),
            "rank_of_selected_on_test": int((df["test_mse"] < best["test_mse"]).sum() + 1),
            "corr_val_test": float(np.corrcoef(df["val_mse"], df["test_mse"])[0, 1]),
            "spearman_val_test": float(df["val_mse"].rank().corr(df["test_mse"].rank())),
            "mean_val_minus_test": float((df["val_mse"] - df["test_mse"]).mean()),
            "mean_test_over_val_ratio": float((df["test_mse"] / df["val_mse"]).mean()),
            "resampling": selection_bias(kind, splits),
        }
        a.scatter(df["val_mse"], df["test_mse"], s=6, alpha=0.5, label="all parameter sets")
        a.scatter([best["val_mse"]], [best["test_mse"]], c="r", s=40, marker="*", label="selected (min val MSE)")
        lo, hi = min(df["val_mse"].min(), df["test_mse"].min()), max(df["val_mse"].max(), df["test_mse"].max())
        a.plot([lo, hi], [lo, hi], "k--", lw=1, label="val = test")
        a.set(xlabel="validation MSE", ylabel="test MSE", title=f"{names[kind]} ({len(df)} settings)")
        a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "p2_val_vs_test.pdf"); fig.savefig(FIG / "p2_val_vs_test.png", dpi=150)

    # baseline: predict the training mean
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    out["baseline_mean"] = {"val_mse": float(np.mean((yva - ytr.mean()) ** 2)),
                            "test_mse": float(np.mean((yte - ytr.mean()) ** 2))}
    out["n"] = [len(ytr), len(yva), len(yte)]
    (RES / "p2_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    splits = load_split()
    kinds = sys.argv[1:] or ["knn", "rf"]
    for kind in kinds:
        if kind in ("knn", "rf"):
            run(kind, splits)
    if all((RES / f"p2_grid_{k}.csv").exists() for k in ("knn", "rf")):
        summarize(splits)


if __name__ == "__main__":
    main()
