"""Problem 5 (classification): logistic and probit regression with autodiff + SGD."""
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from data import ROOT, load_cancer_split
from sgd import linear, train

EPS = 1e-7


def bce(p, y):
    p = jnp.clip(p, EPS, 1 - EPS)   # avoid log(0)
    return -jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))


def logistic_prob(params, X):
    return jax.nn.sigmoid(linear(params, X))


def probit_prob(params, X):
    return norm.cdf(linear(params, X))


def acc(p, y):
    return float(np.mean((np.asarray(p) >= 0.5) == y))


def main(lr=0.01, epochs=200, batch_size=32):
    Xtr, Xte, ytr, yte = load_cancer_split()
    res = {}

    base = LogisticRegression(max_iter=10_000).fit(Xtr, ytr)
    res["sklearn logistic (baseline)"] = {"test_acc": float(base.score(Xte, yte))}

    for name, prob in [("JAX logistic + SGD", logistic_prob), ("JAX probit + SGD", probit_prob)]:
        loss = lambda params, X, y, prob=prob: bce(prob(params, X), y)
        params, hist = train(loss, Xtr, ytr, lr=lr, epochs=epochs, batch_size=batch_size)
        res[name] = {"test_acc": acc(prob(params, Xte), yte),
                     "train_acc": acc(prob(params, Xtr), ytr),
                     "final_train_bce": hist[-1],
                     "test_bce": float(bce(prob(params, Xte), yte))}
    res["_config"] = {"lr": lr, "epochs": epochs, "batch_size": batch_size,
                      "n_train": len(ytr), "n_test": len(yte)}
    (ROOT / "results" / "p5_classification.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
