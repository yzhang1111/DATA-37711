"""Shared helpers: paths, a tiny JAX MLP, and a mini-batch trainer with early stopping."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, RES, PRED = (ROOT / d for d in ("data", "figures", "results", "predictions"))
for d in (FIG, RES, PRED):
    d.mkdir(exist_ok=True)


def save_json(obj, name):
    (RES / name).write_text(json.dumps(obj, indent=2))


# --------------------------------------------------------------------------- MLP
def init_mlp(key, sizes, scale="he"):
    """sizes = [d_in, h1, ..., d_out]; returns list of (W, b)."""
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (m, n) in zip(keys, zip(sizes[:-1], sizes[1:])):
        W = jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / m)
        params.append((W, jnp.zeros(n)))
    return params


def mlp_apply(params, x, act=jax.nn.relu):
    for W, b in params[:-1]:
        x = act(x @ W + b)
    W, b = params[-1]
    return x @ W + b


# ----------------------------------------------------------------------- trainer
def train(loss_fn, params, train_arrays, val_arrays, lr=1e-3, batch_size=1024,
          max_epochs=50, patience=5, seed=0, weight_decay=0.0, verbose=True):
    """Generic Adam(W) trainer.

    loss_fn(params, *arrays) -> scalar mean loss. Early stopping on validation loss;
    returns the parameters from the best validation epoch and the loss history.
    """
    opt = optax.adamw(lr, weight_decay=weight_decay) if weight_decay else optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(params, state, *batch):
        loss, g = jax.value_and_grad(loss_fn)(params, *batch)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, loss

    eval_loss = jax.jit(loss_fn)

    def full_loss(params, arrays, bs=65536):
        n = len(arrays[0])
        tot = 0.0
        for i in range(0, n, bs):
            chunk = [a[i:i + bs] for a in arrays]
            tot += float(eval_loss(params, *chunk)) * len(chunk[0])
        return tot / n

    rng = np.random.default_rng(seed)
    n = len(train_arrays[0])
    best, best_params, bad, hist = np.inf, params, 0, []
    for epoch in range(max_epochs):
        perm = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            params, state, _ = step(params, state, *[a[idx] for a in train_arrays])
        tr, va = full_loss(params, train_arrays), full_loss(params, val_arrays)
        hist.append({"epoch": epoch + 1, "train": tr, "val": va})
        if verbose:
            print(f"  epoch {epoch + 1:3d}  train {tr:.4f}  val {va:.4f}")
        if va < best - 1e-5:
            best, best_params, bad = va, params, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_params, hist
