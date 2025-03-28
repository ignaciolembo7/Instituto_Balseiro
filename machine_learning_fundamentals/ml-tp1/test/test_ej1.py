import os
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from ej1.ej1 import (
    read_from_csv,
    mse,
    precision,
    recall,
    f1_score as ej1_f1_score,
    cross_entropy,
)


def test_read_from_csv():
    file = "../ej1/ej1.csv"
    assert pd.read_csv(file)["y"] == read_from_csv(file)[0]
    assert pd.read_csv(file)["y_pred"] == read_from_csv(file)[1]


def test_mse():
    y = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([1, 1, 0, 1])
    assert mse(y, y_pred) == mean_squared_error(y, y_pred)


def test_bce():
    y = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([1, 1, 0, 1])
    assert cross_entropy(y, y_pred) == log_loss(y, y_pred)


def test_precision():
    y = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([1, 1, 0, 1])
    assert precision(y, y_pred) == precision_score(y, y_pred)


def test_recall():
    y = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([1, 1, 0, 1])
    assert recall(y, y_pred) == recall_score(y, y_pred)


def test_precision():
    y = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([1, 1, 0, 1])
    assert f1_score(y, y_pred) == ej1_f1_score(y, y_pred)
