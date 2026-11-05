"""Problem 1: heteroscedastic Gaussian MLP for NYC taxi trip duration.

Model: g(x) in R^2,  mu(x) = g_1(x),  sigma(x) = softplus(g_2(x)) + 1e-3.
Loss:  Gaussian NLL  log sigma + (y - mu)^2 / (2 sigma^2)   (+ const).
"""
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from common import DATA, FIG, init_mlp, mlp_apply, save_json, train

SEED, THRESH, HIDDEN = 0, 45.0, [128, 128]
LOG2PI = 0.5 * np.log(2 * np.pi)


def link(out):
    """R^2 -> (mu, sigma): identity for the mean, softplus (+ floor) for the std."""
    return out[:, 0], jax.nn.softplus(out[:, 1]) + 1e-3


def gauss_nll(mu, sigma, y):
    return jnp.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2 + LOG2PI


def load_split():
    df = pd.read_parquet(DATA / "taxi_trips.parquet")
    y = df.pop("trip_duration").to_numpy(np.float32)
    X = df.to_numpy(np.float32)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    n_tr, n_va = int(0.7 * len(y)), int(0.15 * len(y))
    idx = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    return [(X[i], y[i]) for i in idx], list(df.columns)


def prob_metrics(p, y):
    """p = predicted P(T < 45); label = 1{T < 45}."""
    lab = (y < THRESH).astype(float)
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    return {"brier": float(np.mean((p - lab) ** 2)),
            "log_loss": float(-np.mean(lab * np.log(pc) + (1 - lab) * np.log(1 - pc))),
            "accuracy@0.5": float(np.mean((p >= 0.5) == lab)),
            "auc": float(roc_auc_score(lab, p)) if 0 < lab.mean() < 1 else None}


def evaluate(name, mu, sigma, y):
    p = norm.cdf((THRESH - mu) / sigma)
    r = {"model": name,
         "nll": float(np.mean(np.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2 + LOG2PI)),
         "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
         "mae": float(np.mean(np.abs(y - mu))),
         "coverage90": float(np.mean(np.abs(y - mu) <= 1.6449 * sigma))}
    r.update(prob_metrics(p, y))
    return r, p


def main():
    splits, cols = load_split()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    print(f"n_train={len(ytr)}  n_val={len(yva)}  n_test={len(yte)}   P(T<45) train={np.mean(ytr < THRESH):.4f}")
    rows, probs = [], {}

    # --- Baseline 0: base rate for the probability task
    p0 = np.full_like(yte, np.mean(ytr < THRESH))
    r = {"model": "Base rate P(T<45) from train"}
    r.update(prob_metrics(p0, yte)); rows.append(r)

    # --- Baseline 1: unconditional Gaussian  N(mean_train, sd_train^2)
    mu, sd = ytr.mean(), ytr.std()
    r, probs["const"] = evaluate("Constant Gaussian", np.full_like(yte, mu), np.full_like(yte, sd), yte); rows.append(r)

    # --- Baseline 2: linear mean, constant sigma (= train residual std)
    lin = LinearRegression().fit(Xtr, ytr)
    s = np.std(ytr - lin.predict(Xtr))
    r, probs["linear"] = evaluate("Linear mean + const sigma", lin.predict(Xte), np.full_like(yte, s), yte); rows.append(r)

    # --- Baseline 3: MLP mean trained with MSE, constant sigma (homoscedastic MLP)
    key = jax.random.PRNGKey(SEED)
    sizes = [Xtr.shape[1], *HIDDEN, 1]
    mse = lambda p, X, y: jnp.mean((mlp_apply(p, X)[:, 0] - y) ** 2)
    print("Training homoscedastic MLP (MSE)")
    pm, hist_mse = train(mse, init_mlp(key, sizes), (Xtr, ytr), (Xva, yva), lr=1e-3, batch_size=1024,
                         max_epochs=60, patience=5)
    mu_tr = np.asarray(mlp_apply(pm, Xtr)[:, 0]); s = np.std(ytr - mu_tr)
    mu_te = np.asarray(mlp_apply(pm, Xte)[:, 0])
    r, probs["mlp_mse"] = evaluate("MLP mean (MSE) + const sigma", mu_te, np.full_like(yte, s), yte); rows.append(r)

    # --- Main model: heteroscedastic MLP, outputs (mu, sigma), Gaussian NLL
    sizes = [Xtr.shape[1], *HIDDEN, 2]
    p_init = init_mlp(jax.random.PRNGKey(SEED + 1), sizes)
    # start sigma near the marginal sd and mu near the marginal mean for stable early training
    W, b = p_init[-1]
    p_init[-1] = (W * 0.1, b.at[0].set(float(ytr.mean())).at[1].set(float(np.log(np.expm1(ytr.std())))))

    def nll(p, X, y):
        m, s = link(mlp_apply(p, X))
        return jnp.mean(gauss_nll(m, s, y))

    print("Training heteroscedastic MLP (Gaussian NLL)")
    ph, hist_nll = train(nll, p_init, (Xtr, ytr), (Xva, yva), lr=1e-3, batch_size=1024,
                         max_epochs=60, patience=5)
    m, s = (np.asarray(a) for a in link(mlp_apply(ph, Xte)))
    r, probs["mlp_nll"] = evaluate("Heteroscedastic MLP (NLL)", m, s, yte); rows.append(r)

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    save_json({"results": rows, "n": [len(ytr), len(yva), len(yte)],
               "hist_mse": hist_mse, "hist_nll": hist_nll, "hidden": HIDDEN}, "p1_taxi.json")

    # ---------------- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot([h["epoch"] for h in hist_nll], [h["train"] for h in hist_nll], label="train")
    ax[0].plot([h["epoch"] for h in hist_nll], [h["val"] for h in hist_nll], label="val")
    ax[0].set(xlabel="epoch", ylabel="Gaussian NLL", title="Heteroscedastic MLP training"); ax[0].legend()
    ax[1].scatter(m[:5000], np.abs(yte - m)[:5000], s=2, alpha=0.3, label=r"$|y-\hat\mu|$")
    o = np.argsort(m[:5000]); ax[1].plot(m[:5000][o], s[:5000][o], "r.", ms=2, label=r"$\hat\sigma(x)$")
    ax[1].set(xlabel=r"predicted mean $\hat\mu(x)$ (min)", ylabel="minutes", ylim=(0, 60),
              title="Predicted sigma grows with predicted mean"); ax[1].legend()
    # standardized residuals of the heteroscedastic model vs N(0,1): checks the Gaussian assumption
    zres = (yte - m) / s
    ax[2].hist(np.clip(zres, -5, 8), bins=130, density=True, alpha=0.6, label=r"$(y-\hat\mu)/\hat\sigma$ (test)")
    g = np.linspace(-5, 8, 400); ax[2].plot(g, norm.pdf(g), "k", lw=1.5, label="N(0,1)")
    ax[2].set(xlabel="standardized residual (clipped to [-5, 8])", ylabel="density",
              title=f"Residuals: skew {float(pd.Series(zres).skew()):.2f}"); ax[2].legend()
    fig.tight_layout(); fig.savefig(FIG / "p1_taxi.pdf"); fig.savefig(FIG / "p1_taxi.png", dpi=150)

    # LaTeX rows
    f = lambda v, d=4: "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"
    lines = [f"{r['model']} & {f(r.get('nll'), 3)} & {f(r.get('rmse'), 2)} & {f(r.get('coverage90'), 3)} & "
             f"{f(r['brier'], 5)} & {f(r['log_loss'], 4)} & {f(r['accuracy@0.5'], 4)} & {f(r['auc'], 3)} \\\\"
             for r in rows]
    from common import RES
    (RES / "p1_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
