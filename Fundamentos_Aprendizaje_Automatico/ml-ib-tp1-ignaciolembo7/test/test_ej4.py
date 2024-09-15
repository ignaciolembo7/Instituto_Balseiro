import pandas as pd
import os


def test_output_file():
    df = "../ej4/ej4_output.csv"
    assert df["1"].iloc[0] not in ("underfitting", "overfitting")
    assert df["2"].iloc[0] == "underfitting"
    assert df["3"].iloc[0] == "overfitting"
