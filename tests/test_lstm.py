"""Tests du LSTM sur données FACTICES (ignorés si PyTorch n'est pas installé)."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from elecfr.models.lstm import LSTMModel  # noqa: E402
from test_models import fake  # noqa: E402

PETIT = {"hidden": 8, "max_epochs": 3, "patience": 2, "batch_size": 16}


def split(df, day):
    d = pd.Timestamp(day)
    return df[df["date_local"] <= d - pd.Timedelta(days=2)], df[df["date_local"] == d]


def test_prediction_for_day_j_ignores_consumption_of_j_and_j_minus_1():
    base = fake(start="2024-01-01", end="2024-05-01")
    day = pd.Timestamp("2024-04-20")
    train, target = split(base, day)
    p1 = LSTMModel(base=base, **PETIT)(train, target)

    altered = base.copy()
    for d in (day, day - pd.Timedelta(days=1)):      # J et J-1 : inconnus (en tout ou partie) à l'émission
        altered.loc[altered["date_local"] == d, "consommation"] += 50_000
    train2, target2 = split(altered, day)
    p2 = LSTMModel(base=altered, **PETIT)(train2, target2)
    np.testing.assert_allclose(p1, p2)


def test_prediction_for_day_j_uses_day_j_minus_2():
    # Contre-épreuve : J-2 est connu, il DOIT pouvoir changer la prévision.
    base = fake(start="2024-01-01", end="2024-05-01")
    day = pd.Timestamp("2024-04-20")
    train, target = split(base, day)
    p1 = LSTMModel(base=base, **PETIT)(train, target)
    altered = base.copy()
    altered.loc[altered["date_local"] == day - pd.Timedelta(days=2), "consommation"] += 50_000
    p2 = LSTMModel(base=altered, **PETIT)(*split(altered, day))
    assert not np.allclose(p1, p2)


def test_one_finite_prediction_per_row_including_dst_days():
    base = fake(start="2024-01-01", end="2024-04-15")   # contient le passage à l'heure d'été (31/03)
    train = base[base["date_local"] < pd.Timestamp("2024-03-20")]
    target = base[base["date_local"] >= pd.Timestamp("2024-03-22")]
    m = LSTMModel(base=base, **PETIT)
    yhat = m(train, target)
    assert yhat.shape == (len(target),) and np.isfinite(yhat).all()
    assert m.dernier is not None and len(m.epoques) == 1
