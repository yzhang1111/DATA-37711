"""Problem 5 (regression): linear regression with MSE vs Student-t (nu=5) NLL."""
import json
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import t as student_t
from sklearn.linear_model import LinearRegression

from data import ROOT, load_ale_split
from sgd import linear, train

NU = 5.0


def mse(params, X, y):
    return jnp.mean((y - linear(params, X)) ** 2)


def t_nll(params, X, y):
    return -jnp.mean(student_t.logpdf(y - linear(params, X), NU))


def mae(params, X, y):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(linear(params, X)))))


def main(lr=0.01, epochs=500, batch_size=16):
    Xtr, Xte, ytr, yte = load_ale_split()
    res = {}
    base = LinearRegression().fit(Xtr, ytr)
    res["sklearn OLS (baseline)"] = {
        "test_mae": float(np.mean(np.abs(yte - base.predict(Xte)))),
        "coef": base.coef_.tolist(), "intercept": float(base.intercept_)}

    for name, loss in [("JAX linear, MSE + SGD", mse), ("JAX linear, Student-t(5) NLL + SGD", t_nll)]:
        params, hist = train(loss, Xtr, ytr, lr=lr, epochs=epochs, batch_size=batch_size)
        res[name] = {"test_mae": mae(params, Xte, yte), "train_mae": mae(params, Xtr, ytr),
                     "final_train_loss": hist[-1],
                     "coef": np.asarray(params["w"]).tolist(), "intercept": float(params["b"])}
    res["_config"] = {"lr": lr, "epochs": epochs, "batch_size": batch_size, "nu": NU,
                      "n_train": len(ytr), "n_test": len(yte)}
    (ROOT / "results" / "p5_regression.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
