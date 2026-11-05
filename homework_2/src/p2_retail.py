"""Problem 2: encoding high-cardinality categoricals (StockCode, CustomerID) to predict Quantity.

Quantity is extremely heavy-tailed, so every model is trained on z = log(1 + Quantity) with squared
error (i.e. a log-normal model for Quantity). Metrics on the test set:
  * RMSE on the log scale (the training objective),
  * MAE on the original scale using expm1(prediction) (the implied median),
  * Spearman rank correlation between prediction and Quantity.
"""
import category_encoders as ce
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from common import DATA, FIG, RES, init_mlp, mlp_apply, save_json, train

SEED, EMB_DIM = 0, 16
CATS = ["StockCode", "CustomerID"]


def load():
    df = pd.read_csv(DATA / "retail.csv", dtype={"StockCode": str})
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    for c in CATS:
        df[c] = df[c].astype("category")
    df["z"] = np.log1p(df["Quantity"]).astype(np.float32)
    perm = np.random.default_rng(SEED).permutation(len(df))
    n_tr, n_va = int(0.7 * len(df)), int(0.15 * len(df))
    tr, va, te = (df.iloc[i].reset_index(drop=True)
                  for i in (perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]))
    return df, tr, va, te


# ------------------------------------------------------------------ target encoding
def target_encode(tr, others, n_folds=5):
    """Out-of-fold TE for training rows (avoids target leakage); full-train TE for val/test."""
    oof = np.zeros((len(tr), len(CATS)), np.float32)
    for fit_idx, enc_idx in KFold(n_folds, shuffle=True, random_state=SEED).split(tr):
        enc = ce.TargetEncoder(cols=CATS, smoothing=10).fit(tr.loc[fit_idx, CATS], tr.loc[fit_idx, "z"])
        oof[enc_idx] = enc.transform(tr.loc[enc_idx, CATS]).to_numpy(np.float32)
    enc = ce.TargetEncoder(cols=CATS, smoothing=10).fit(tr[CATS], tr["z"])
    leaky_tr = enc.transform(tr[CATS]).to_numpy(np.float32)       # in-sample TE, for illustration
    return oof, leaky_tr, [enc.transform(o[CATS]).to_numpy(np.float32) for o in others]


# ------------------------------------------------------------------ embedding model
def make_index(tr, frames):
    """Integer ids per category; 0 is reserved for values unseen in training."""
    maps = {c: {v: i + 1 for i, v in enumerate(tr[c].cat.categories)} for c in CATS}
    return maps, [np.stack([f[c].map(maps[c]).astype(float).fillna(0).to_numpy(np.int32) for c in CATS], 1)
                  for f in frames]


def init_emb_model(key, n_cats, d_extra, hidden=(64, 64)):
    k1, k2, k3 = jax.random.split(key, 3)
    embs = [0.05 * jax.random.normal(k, (n + 1, EMB_DIM)) for k, n in zip((k1, k2), n_cats)]
    return {"emb": embs, "mlp": init_mlp(k3, [EMB_DIM * len(n_cats) + d_extra, *hidden, 1])}


def emb_forward(params, ids, extra):
    h = [params["emb"][j][ids[:, j]] for j in range(ids.shape[1])]
    if extra.shape[1]:
        h.append(extra)
    return mlp_apply(params["mlp"], jnp.concatenate(h, 1))[:, 0]


def fit_emb(ids_tr, ex_tr, z_tr, ids_va, ex_va, z_va, n_cats, seed):
    loss = lambda p, i, e, z: jnp.mean((emb_forward(p, i, e) - z) ** 2)
    params = init_emb_model(jax.random.PRNGKey(seed), n_cats, ex_tr.shape[1])
    params["mlp"][-1] = (params["mlp"][-1][0], params["mlp"][-1][1] + float(z_tr.mean()))
    return train(loss, params, (ids_tr, ex_tr, z_tr), (ids_va, ex_va, z_va), lr=1e-3,
                 batch_size=1024, max_epochs=40, patience=3, weight_decay=1e-4, seed=seed, verbose=False)


# ------------------------------------------------------------------ evaluation
def metrics(name, zhat, te):
    q, z = te["Quantity"].to_numpy(), te["z"].to_numpy()
    return {"model": name, "rmse_log": float(np.sqrt(np.mean((zhat - z) ** 2))),
            "mae_qty": float(np.mean(np.abs(np.expm1(zhat) - q))),
            "spearman": float(spearmanr(zhat, q).statistic) if np.std(zhat) > 0 else float("nan")}


def main():
    df, tr, va, te = load()
    card = {c: int(df[c].nunique()) for c in CATS}
    unseen = {c: float((~te[c].isin(set(tr[c]))).mean()) for c in CATS}
    print("cardinality:", card, " test rows with category unseen in train:", unseen)
    print(f"n_train={len(tr)} n_val={len(va)} n_test={len(te)}")
    rows, preds = [], {}

    rows.append(metrics("Global mean of log(1+Q)", np.full(len(te), tr["z"].mean()), te))

    # (b) target encoding + gradient boosting
    te_tr, leaky_tr, (te_va, te_te) = target_encode(tr, [va, te])
    hgb = lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=True,
                                                validation_fraction=0.1, random_state=SEED)
    m = hgb().fit(te_tr, tr["z"]); preds["te"] = m.predict(te_te)
    rows.append(metrics("TE (out-of-fold) + gradient boosting", preds["te"], te))
    m_leak = hgb().fit(leaky_tr, tr["z"])
    rows.append(metrics("TE (in-sample, leaky) + gradient boosting", m_leak.predict(te_te), te))

    # (c) embeddings, (d) embeddings + TE
    maps, (id_tr, id_va, id_te) = make_index(tr, [tr, va, te])
    n_cats = [len(maps[c]) for c in CATS]
    mu, sd = te_tr.mean(0), te_tr.std(0)
    std = lambda a: ((a - mu) / sd).astype(np.float32)
    zt, zv = tr["z"].to_numpy(np.float32), va["z"].to_numpy(np.float32)
    none = lambda n: np.zeros((n, 0), np.float32)

    hist = {}
    for key, name, ex in [("emb", "Embeddings (16+16) + MLP", (none(len(tr)), none(len(va)), none(len(te)))),
                          ("emb_te", "Embeddings + TE features + MLP", (std(te_tr), std(te_va), std(te_te)))]:
        print("training", name)
        p, hist[key] = fit_emb(id_tr, ex[0], zt, id_va, ex[1], zv, n_cats, SEED)
        preds[key] = np.asarray(emb_forward(p, id_te, ex[2]))
        rows.append(metrics(name, preds[key], te))
        if key == "emb":
            emb_params = p

    res = pd.DataFrame(rows); print(res.to_string(index=False))

    # (e) error by how often the test item's StockCode appeared in training
    cnt = te["StockCode"].map(tr["StockCode"].value_counts()).fillna(0).to_numpy()
    buckets = [("unseen", cnt == 0), ("1-10", (cnt >= 1) & (cnt <= 10)),
               ("11-100", (cnt > 10) & (cnt <= 100)), (">100", cnt > 100)]
    z = te["z"].to_numpy()
    by_freq = []
    for b, mask in buckets:
        r = {"bucket": b, "n_test": int(mask.sum())}
        for k in ["te", "emb", "emb_te"]:
            r[k] = float(np.sqrt(np.mean((preds[k][mask] - z[mask]) ** 2))) if mask.any() else None
        by_freq.append(r)
    print(pd.DataFrame(by_freq).to_string(index=False))

    # embedding geometry: does the StockCode embedding encode more than the item mean?
    E = np.asarray(emb_params["emb"][0])[1:]
    item_mean = tr.groupby("StockCode", observed=False)["z"].mean().reindex(tr["StockCode"].cat.categories).to_numpy()
    item_n = tr["StockCode"].value_counts().reindex(tr["StockCode"].cat.categories).to_numpy()
    U, S, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    pcs = U[:, :2] * S[:2]
    ok = item_n >= 20
    corr = [float(abs(np.corrcoef(pcs[ok, j], item_mean[ok])[0, 1])) for j in range(2)]
    var_expl = (S ** 2 / np.sum(S ** 2))[:5].tolist()
    print("corr(|PC1|,|PC2| with item mean log-qty):", corr, " variance explained:", var_expl)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sc = ax[0].scatter(pcs[ok, 0], pcs[ok, 1], c=item_mean[ok], s=4, cmap="viridis")
    plt.colorbar(sc, ax=ax[0], label="item mean log(1+Q) in train")
    ax[0].set(xlabel="PC1", ylabel="PC2", title="StockCode embeddings (items with >=20 rows)")
    lab = [b for b, _ in buckets]
    x = np.arange(len(lab))
    for j, (k, nm) in enumerate([("te", "target enc. + GBM"), ("emb", "embeddings"), ("emb_te", "emb. + TE")]):
        ax[1].bar(x + (j - 1) * 0.27, [r[k] for r in by_freq], 0.27, label=nm)
    ax[1].set(xticks=x, xticklabels=lab, xlabel="# training rows with the same StockCode",
              ylabel="test RMSE on log(1+Q)", title="Error by item frequency"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "p2_retail.pdf"); fig.savefig(FIG / "p2_retail.png", dpi=150)

    save_json({"cardinality": card, "unseen_frac_test": unseen, "n": [len(tr), len(va), len(te)],
               "results": rows, "by_freq": by_freq, "pc_corr_item_mean": corr,
               "emb_var_explained": var_expl, "hist": hist}, "p2_retail.json")
    (RES / "p2_table.tex").write_text("\n".join(
        f"{r['model']} & {r['rmse_log']:.4f} & {r['mae_qty']:.2f} & " + ("--" if np.isnan(r['spearman']) else f"{r['spearman']:.3f}") + " \\\\" for r in rows) + "\n")
    (RES / "p2_freq_table.tex").write_text("\n".join(
        f"{r['bucket']} & {r['n_test']} & {r['te']:.3f} & {r['emb']:.3f} & {r['emb_te']:.3f} \\\\" for r in by_freq) + "\n")


if __name__ == "__main__":
    main()
