"""Problem 3: generalization of gradient boosting as the number of trees grows, on dset0 and dset1.

Column 0 is a binary label; columns 1.. are features. Each dataset is split 70 / 30 (stratified, seed 0).
GradientBoostingClassifier with default settings (learning rate 0.1, depth-3 trees, no subsampling)
is fit with 1, 10, 100, 300 and 1000 trees. Because default gradient boosting is deterministic and
adds trees sequentially, the m-tree model is exactly the first m stages of the 1000-tree model;
we use staged predictions to draw the full curve and check the five requested sizes against
separately fitted models. To check that the pattern is not an artifact of one split, the test curves are
also averaged over 20 further random splits. Finally, simple vs flexible models are compared by 10-fold
CV to characterize each dataset (part c).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, RES = ROOT / "data", ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
SEED, SIZES, MAX_T, N_REPEATS = 0, [1, 10, 100, 300, 1000], 1000, 20


def load_xy(name):
    d = pd.read_csv(DATA / f"{name}.csv", header=None).to_numpy(float)
    return d[:, 1:], d[:, 0].astype(int)


def load(name, seed=SEED):
    X, y = load_xy(name)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def repeated_curves(name):
    """Test accuracy / log-loss curves averaged over N_REPEATS random 70/30 splits."""
    acc, ll = [], []
    for r in range(N_REPEATS):
        Xtr, Xte, ytr, yte = load(name, seed=100 + r)
        gb = GradientBoostingClassifier(n_estimators=MAX_T, random_state=SEED).fit(Xtr, ytr)
        a, l = curves(gb, Xte, yte); acc.append(a); ll.append(l)
    return np.array(acc), np.array(ll)


def diagnostics(name):
    """10-fold CV accuracy of simple vs flexible models: how complex is the signal?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    X, y = load_xy(name)
    cv = StratifiedKFold(10, shuffle=True, random_state=SEED)
    models = {"logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
              "single depth-3 tree": DecisionTreeClassifier(max_depth=3, random_state=SEED),
              "random forest (300 trees)": RandomForestClassifier(300, random_state=SEED)}
    out = {k: float(cross_val_score(m, X, y, cv=cv).mean()) for k, m in models.items()}
    out["corr_with_label"] = [float(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])]
    return out


def curves(model, X, y):
    acc = [accuracy_score(y, p) for p in model.staged_predict(X)]
    ll = [log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]) for p in model.staged_predict_proba(X)]
    return np.array(acc), np.array(ll)


def main():
    out, rows = {}, []
    rep, diag = {}, {}
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for j, name in enumerate(["dset0", "dset1"]):
        Xtr, Xte, ytr, yte = load(name)
        gb = GradientBoostingClassifier(n_estimators=MAX_T, random_state=SEED).fit(Xtr, ytr)
        tr_acc, tr_ll = curves(gb, Xtr, ytr)
        te_acc, te_ll = curves(gb, Xte, yte)

        # check: separately fitted models with m trees match the staged predictions
        for m in SIZES:
            g = GradientBoostingClassifier(n_estimators=m, random_state=SEED).fit(Xtr, ytr)
            assert abs(accuracy_score(yte, g.predict(Xte)) - te_acc[m - 1]) < 1e-12
            rows.append({"dataset": name, "trees": m,
                         "train_acc": float(tr_acc[m - 1]), "test_acc": float(te_acc[m - 1]),
                         "train_logloss": float(tr_ll[m - 1]), "test_logloss": float(te_ll[m - 1])})
        out[name] = {"n_train": len(ytr), "n_test": len(yte), "n_features": Xtr.shape[1],
                     "class_balance": float(ytr.mean()),
                     "best_test_acc": float(te_acc.max()), "best_test_acc_trees": int(te_acc.argmax() + 1),
                     "best_test_logloss": float(te_ll.min()), "best_test_logloss_trees": int(te_ll.argmin() + 1),
                     "feature_importances": gb.feature_importances_.round(4).tolist()}

        ra, rl = repeated_curves(name)
        rep[name] = {"trees": SIZES,
                     "test_acc_mean": [float(ra[:, m - 1].mean()) for m in SIZES],
                     "test_acc_sd": [float(ra[:, m - 1].std()) for m in SIZES],
                     "test_logloss_mean": [float(rl[:, m - 1].mean()) for m in SIZES],
                     "best_mean_acc_trees": int(ra.mean(0).argmax() + 1),
                     "best_mean_acc": float(ra.mean(0).max()),
                     "best_mean_logloss_trees": int(rl.mean(0).argmin() + 1)}
        diag[name] = diagnostics(name)

        t = np.arange(1, MAX_T + 1)
        ax[0, j].plot(t, 1 - tr_acc, label="train (split 0)"); ax[0, j].plot(t, 1 - te_acc, label="test (split 0)")
        ax[0, j].plot(t, 1 - ra.mean(0), "k", lw=1.2, label=f"test, mean of {N_REPEATS} splits")
        ax[0, j].set(ylabel="misclassification error", title=name, xscale="log")
        ax[1, j].plot(t, tr_ll, label="train (split 0)"); ax[1, j].plot(t, te_ll, label="test (split 0)")
        ax[1, j].plot(t, rl.mean(0), "k", lw=1.2, label=f"test, mean of {N_REPEATS} splits")
        ax[1, j].set(ylabel="log-loss", xlabel="number of trees", xscale="log")
        for a in ax[:, j]:
            for m in SIZES:
                a.axvline(m, color="grey", lw=0.5, ls=":")
            a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "p3_boosting.pdf"); fig.savefig(FIG / "p3_boosting.png", dpi=150)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out["repeated_splits"], out["diagnostics"] = rep, diag
    print(json.dumps({"repeated_splits": rep, "diagnostics": diag}, indent=2))
    df.to_csv(RES / "p3_boosting.csv", index=False)
    (RES / "p3_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
