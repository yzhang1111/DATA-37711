"""Minimal mini-batch SGD trainer for linear models in JAX + optax."""
import jax
import jax.numpy as jnp
import numpy as np
import optax


def init_params(p, key=0):
    k = jax.random.PRNGKey(key)
    return {"w": 0.01 * jax.random.normal(k, (p,)), "b": jnp.zeros(())}


def linear(params, X):
    return X @ params["w"] + params["b"]


def train(loss_fn, X, y, lr=0.05, epochs=200, batch_size=32, seed=0):
    """loss_fn(params, X, y) -> scalar mean loss. Returns params and loss history."""
    X, y = jnp.asarray(X), jnp.asarray(y)
    params = init_params(X.shape[1], seed)
    opt = optax.sgd(lr)
    state = opt.init(params)

    @jax.jit
    def step(params, state, xb, yb):
        loss, g = jax.value_and_grad(loss_fn)(params, xb, yb)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, loss

    rng = np.random.default_rng(seed)
    n, hist = X.shape[0], []
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            params, state, _ = step(params, state, X[idx], y[idx])
        hist.append(float(loss_fn(params, X, y)))
    return params, hist
