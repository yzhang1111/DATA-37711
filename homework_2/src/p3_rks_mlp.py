"""Problem 3: Random Kitchen Sinks + ridge vs. a 1-hidden-layer MLP on the 2-D toy dataset.

RKS:  z(x) = [sin(w_j^T x + b_j), cos(w_j^T x + b_j)]_{j=1..h},  w_j ~ N(0, I/sigma^2), b_j ~ U[0, 2pi],
      then ridge regression  min_beta (1/n)||y - beta_0 - Z beta||^2 + lambda ||beta||^2  (intercept unpenalized).
Sweep: kernel scale sigma (D = 2h = 1000, lambda = 1e-3 fixed).
MLP:  x -> ReLU(W1 x + b1) -> w2^T h + b2, Adam, MSE.  Sweep: hidden width (lr = 3e-3, batch 64, 400 epochs).
"""
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd

from common import DATA, FIG, PRED, RES, init_mlp, mlp_apply, save_json
from dataloader import HW2Dataset

SEEDS = [0, 1, 2]


# ============================================================================ RKS
class RandomKitchenSinks:
    def __init__(self, D=1000, sigma=1.0, ridge_lambda=1e-3, seed=0):
        assert D % 2 == 0
        self.h, self.sigma, self.lam, self.rng = D // 2, sigma, ridge_lambda, np.random.default_rng(seed)
        self.W = self.b = None

    def _features(self, X):
        if self.W is None:
            self.W = self.rng.normal(0.0, 1.0 / self.sigma, size=(X.shape[1], self.h))   # w_j ~ N(0, I/sigma^2)
            self.b = self.rng.uniform(0.0, 2 * np.pi, size=self.h)
        P = X @ self.W + self.b
        return np.hstack([np.sin(P), np.cos(P)])

    def fit(self, X, y):
        Z = self._features(X)
        self.z_mean, self.y_mean = Z.mean(0), y.mean()
        Zc = Z - self.z_mean
        n, D = Zc.shape
        # normal equations of (1/n)||y - Zc beta||^2 + lam ||beta||^2
        self.coef_ = np.linalg.solve(Zc.T @ Zc / n + self.lam * np.eye(D), Zc.T @ (y - self.y_mean) / n)
        return self

    def predict(self, X):
        return (self._features(X) - self.z_mean) @ self.coef_ + self.y_mean


# ============================================================================ MLP
class MLP:
    def __init__(self, d_in=2, d_hidden=128, lr=3e-3, batch_size=64, epochs=400, seed=0):
        self.sizes, self.lr, self.bs, self.epochs, self.seed = [d_in, d_hidden, 1], lr, batch_size, epochs, seed

    def fit(self, X, y):
        params = init_mlp(jax.random.PRNGKey(self.seed), self.sizes)
        opt = optax.adam(self.lr); state = opt.init(params)
        loss = lambda p, X, y: jnp.mean((mlp_apply(p, X)[:, 0] - y) ** 2)

        @jax.jit
        def epoch(params, state, perm):
            def body(carry, idx):
                p, s = carry
                g = jax.grad(loss)(p, X[idx], y[idx])
                u, s = opt.update(g, s, p)
                return (optax.apply_updates(p, u), s), None
            (params, state), _ = jax.lax.scan(body, (params, state), perm)
            return params, state

        X, y = jnp.asarray(X, jnp.float32), jnp.asarray(y, jnp.float32)
        rng = np.random.default_rng(self.seed)
        nb = len(y) // self.bs
        for _ in range(self.epochs):
            perm = rng.permutation(len(y))[: nb * self.bs].reshape(nb, self.bs)
            params, state = epoch(params, state, jnp.asarray(perm))
        self.params = params
        return self

    def predict(self, X):
        return np.asarray(mlp_apply(self.params, jnp.asarray(X, jnp.float32))[:, 0])


mse = lambda a, b: float(np.mean((a - b) ** 2))


def sweep(make, values, ds):
    """Fit one model per (value, seed); return mean/std train and val MSE per value."""
    out = []
    for v in values:
        tr, va = [], []
        for s in SEEDS:
            m = make(v, s).fit(ds.train.X, ds.train.y)
            tr.append(mse(m.predict(ds.train.X), ds.train.y)); va.append(mse(m.predict(ds.val.X), ds.val.y))
        out.append({"value": v, "train": float(np.mean(tr)), "train_sd": float(np.std(tr)),
                    "val": float(np.mean(va)), "val_sd": float(np.std(va))})
        print(f"  {v:>8}: train {out[-1]['train']:.4f}  val {out[-1]['val']:.4f}")
    return out


def plot_sweep(res, xlabel, title, fname, logx=True):
    v = [r["value"] for r in res]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, k in zip(ax, ["train", "val"]):
        a.errorbar(v, [r[k] for r in res], yerr=[r[k + "_sd"] for r in res], marker="o", capsize=3)
        a.set(xlabel=xlabel, ylabel=f"{'training' if k == 'train' else 'validation'} MSE",
              title=f"{title}: {'training' if k == 'train' else 'validation'} loss")
        if logx: a.set_xscale("log")
        a.set_yscale("log"); a.grid(alpha=0.3)
    best = min(res, key=lambda r: r["val"])
    ax[1].axvline(best["value"], color="r", ls="--", lw=1, label=f"best = {best['value']}"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / f"{fname}.pdf"); fig.savefig(FIG / f"{fname}.png", dpi=150)
    return best


def grid_predict(fn, n=200):
    xs = np.linspace(-3, 3, n)
    X1, X2 = np.meshgrid(xs, xs)
    return X1, X2, fn(np.stack([X1.ravel(), X2.ravel()], 1)).reshape(n, n)


def main():
    ds = HW2Dataset.from_csv(str(DATA / "train.csv"), str(DATA / "val.csv"), str(DATA / "test.csv"))
    print("train", ds.train.X.shape, "val", ds.val.X.shape, "test", ds.test.X.shape)

    # (a) visualize training data: 2-D colour plot + 3-D scatter
    fig = plt.figure(figsize=(11, 4.2))
    a = fig.add_subplot(1, 2, 1)
    sc = a.scatter(*ds.train.X.T, c=ds.train.y, cmap="viridis", s=14)
    plt.colorbar(sc, ax=a, label="y"); a.set(xlabel="x1", ylabel="x2", title="Training data", aspect="equal")
    a = fig.add_subplot(1, 2, 2, projection="3d")
    a.scatter(*ds.train.X.T, ds.train.y, c=ds.train.y, cmap="viridis", s=6)
    a.set(xlabel="x1", ylabel="x2", zlabel="y", title="Training data (3-D)"); a.view_init(25, -60)
    fig.tight_layout(); fig.savefig(FIG / "p3a_data.pdf"); fig.savefig(FIG / "p3a_data.png", dpi=150)

    # (b) RKS sweep over sigma
    sigmas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0]
    print("RKS sweep over sigma (D=1000, lambda=1e-3)")
    rks_res = sweep(lambda s_, seed: RandomKitchenSinks(1000, s_, 1e-3, seed), sigmas, ds)
    best_rks = plot_sweep(rks_res, r"kernel scale $\sigma$", "RKS", "p3b_rks_sweep")

    # (c) MLP sweep over hidden width
    widths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    print("MLP sweep over hidden width (lr=3e-3, batch=64, 400 epochs)")
    mlp_res = sweep(lambda w, seed: MLP(2, w, seed=seed), widths, ds)
    best_mlp = plot_sweep(mlp_res, "hidden width", "1-hidden-layer MLP", "p3c_mlp_sweep")

    # (d) refit best models (seed 0), validation MSE, test predictions, fitted surfaces
    rks = RandomKitchenSinks(1000, best_rks["value"], 1e-3, seed=0).fit(ds.train.X, ds.train.y)
    mlp = MLP(2, best_mlp["value"], seed=0).fit(ds.train.X, ds.train.y)
    final = {"rks": {"sigma": best_rks["value"], "train_mse": mse(rks.predict(ds.train.X), ds.train.y),
                     "val_mse": mse(rks.predict(ds.val.X), ds.val.y)},
             "mlp": {"hidden": best_mlp["value"], "train_mse": mse(mlp.predict(ds.train.X), ds.train.y),
                     "val_mse": mse(mlp.predict(ds.val.X), ds.val.y)},
             "val_y_var": float(np.var(ds.val.y))}
    print(final)
    for name, m in [("rks", rks), ("mlp", mlp)]:
        pd.DataFrame({"y_pred": m.predict(ds.test.X)}).to_csv(PRED / f"{name}_test_predictions.csv", index=False)

    surfaces = [grid_predict(rks.predict), grid_predict(mlp.predict)]
    vmin = min(ds.train.y.min(), *(s[2].min() for s in surfaces))
    vmax = max(ds.train.y.max(), *(s[2].max() for s in surfaces))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    sc = ax[0].scatter(*ds.train.X.T, c=ds.train.y, cmap="viridis", s=12, vmin=vmin, vmax=vmax)
    ax[0].set(title="Training data", xlim=(-3, 3), ylim=(-3, 3))
    for a, (X1, X2, Z), t in zip(ax[1:], surfaces,
                                 [f"RKS ($\\sigma$={best_rks['value']}, D=1000)  val MSE {final['rks']['val_mse']:.3f}",
                                  f"MLP (width {best_mlp['value']})  val MSE {final['mlp']['val_mse']:.3f}"]):
        a.contourf(X1, X2, Z, levels=30, vmin=vmin, vmax=vmax, cmap="viridis"); a.set_title(t, fontsize=10)
    for a in ax:
        a.set(xlabel="x1", ylabel="x2", aspect="equal")
    fig.colorbar(sc, ax=ax, label="y / prediction", shrink=0.85)
    fig.savefig(FIG / "p3d_surfaces.pdf", bbox_inches="tight"); fig.savefig(FIG / "p3d_surfaces.png", dpi=150, bbox_inches="tight")

    save_json({"rks_sweep": rks_res, "mlp_sweep": mlp_res, "final": final}, "p3.json")
    fmt = lambda res: "\n".join(f"{r['value']} & {r['train']:.4f} & {r['val']:.4f} \\\\" for r in res) + "\n"
    (RES / "p3_rks_table.tex").write_text(fmt(rks_res)); (RES / "p3_mlp_table.tex").write_text(fmt(mlp_res))


if __name__ == "__main__":
    main()
