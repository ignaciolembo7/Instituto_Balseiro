import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

import pytest
import pickle

from sklearn.metrics import mean_squared_error

from ej1.estimators.linear_regressor import LinearRegressor
from ej1.estimators.logistic_classifier import LogisticClassifier
from ej1.estimators.tree_stump_regressor import TreeStumpRegressor
from ej1.estimators.base import GradientBoostingEstimator


def rmae(model, X, y):
    return np.linalg.norm(model.predict(X) - y) / np.linalg.norm(y)


def test_linear_regressor():
    with open("ej1/dataset_1.pkl", "rb") as f:
        dataset = pickle.load(f)

    X, y = dataset["X"], dataset["y"]
    X_train, X_test, y_train, y_test = X[:8000], X[8000:], y[:8000], y[8000:]

    model = LinearRegressor(learning_rate=1e-3, batch_size=10, num_epochs=100)
    model.fit(X_train, y_train)
    model_error = mean_squared_error(y_test, model.predict(X_test))

    skmodel = LinearRegression()
    skmodel.fit(X_train, y_train)
    skmodel_error = mean_squared_error(y_test, skmodel.predict(X_test))

    assert model_error == pytest.approx(skmodel_error, rel=5e-1)


def test_logistic_classifier():
    with open("ej1/dataset_2.pkl", "rb") as f:
        dataset = pickle.load(f)

    X, y = dataset["X"], dataset["y"]
    X_train, X_test, y_train, y_test = X[:8000], X[8000:], y[:8000], y[8000:]

    model = LogisticClassifier(learning_rate=1e-1, batch_size=10, num_epochs=1000)
    model.fit(X_train, y_train)
    model_error = mean_squared_error(y_test, model.predict(X_test))

    skmodel = LogisticRegression(penalty="none")
    skmodel.fit(X_train, y_train)
    skmodel_error = mean_squared_error(y_test, skmodel.predict(X_test))

    assert model_error == pytest.approx(skmodel_error, rel=5e-1)


def test_gradient_boosting():
    with open("ej1/dataset_3.pkl", "rb") as f:
        dataset = pickle.load(f)

    X, y = dataset["X"], dataset["y"]
    X_train, X_test, y_train, y_test = X[:8000], X[8000:], y[:8000], y[8000:]

    model = GradientBoostingEstimator(
        n_estimators=5,
        learning_rate=0.75,
        model_class=TreeStumpRegressor,
        model_kwargs={},
    )
    model.fit(X_train, y_train)
    model_error = mean_squared_error(y_test, model.predict(X_test))

    skmodel = LinearRegression()
    skmodel.fit(X_train, y_train)
    skmodel_error = mean_squared_error(y_test, skmodel.predict(X_test))

    assert model_error < skmodel_error
