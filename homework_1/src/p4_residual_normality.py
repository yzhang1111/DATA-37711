"""Problem 4: fit several regressors to ALE and test residuals for normality."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from data import ROOT, load_ale_split

MODELS = {
    "Linear regression": LinearRegression(),
    "SVR (RBF)": SVR(C=10.0, epsilon=0.01),
    "Decision tree": DecisionTreeRegressor(max_depth=4, random_state=0),
    "Random forest": RandomForestRegressor(n_estimators=300, random_state=0),
    "MLP": MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=5000, random_state=0),
}


def main():
    Xtr, Xte, ytr, yte = load_ale_split()
    rows = []
    fig, axes = plt.subplots(2, len(MODELS), figsize=(4 * len(MODELS), 7))
    for j, (name, m) in enumerate(MODELS.items()):
        m.fit(Xtr, ytr)
        r_tr, r_te = ytr - m.predict(Xtr), yte - m.predict(Xte)
        row = {"model": name, "test_mae": float(np.mean(np.abs(r_te)))}
        for tag, r in [("train", r_tr), ("test", r_te)]:
            k2, p_dag = stats.normaltest(r)          # D'Agostino-Pearson
            _, p_sw = stats.shapiro(r)               # Shapiro-Wilk
            row[f"{tag}_dagostino_p"] = float(p_dag)
            row[f"{tag}_shapiro_p"] = float(p_sw)
            row[f"{tag}_skew"] = float(stats.skew(r))
            row[f"{tag}_kurtosis"] = float(stats.kurtosis(r))
        rows.append(row)
        stats.probplot(r_tr, dist="norm", plot=axes[0, j])
        axes[0, j].set_title(f"{name}\ntrain residuals")
        stats.probplot(r_te, dist="norm", plot=axes[1, j])
        axes[1, j].set_title("test residuals")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "p4_qqplots.pdf")

    (ROOT / "results" / "p4.json").write_text(json.dumps(rows, indent=2))
    f = lambda p: f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"
    lines = [f"{r['model']} & {r['test_mae']:.3f} & {f(r['train_dagostino_p'])} & "
             f"{f(r['train_shapiro_p'])} & {f(r['test_dagostino_p'])} & "
             f"{f(r['test_shapiro_p'])} \\\\" for r in rows]
    (ROOT / "results" / "p4_table.tex").write_text("\n".join(lines) + "\n")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
