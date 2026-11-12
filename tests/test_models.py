"""Tests des modèles sur données FACTICES : ils vérifient la mécanique
(colonnes stables, valeurs manquantes, interface walk-forward), pas la qualité."""
import numpy as np
import pandas as pd

from elecfr.evaluation import walk_forward
from elecfr.features import FEATURES
from elecfr.models.common import type_jour
from elecfr.models.gbm import LightGBMModel, importances
from elecfr.models.linear import LinearModel, coefficients, design_matrix


def fake(start="2024-01-01", end="2024-04-30", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start, tz="Europe/Paris"), pd.Timestamp(end, tz="Europe/Paris"),
                        freq="h", inclusive="left").tz_convert("UTC")
    loc = idx.tz_convert("Europe/Paris")
    df = pd.DataFrame(index=idx)
    df["heure"], df["jour_semaine"], df["mois"] = loc.hour, loc.day_of_week, loc.month
    df["date_local"] = loc.normalize().tz_localize(None)
    df["saison"] = "hiver"
    for c in FEATURES:
        if c not in df:
            df[c] = rng.normal(size=len(df))
    df["ferie"] = 0
    df["temp_prev"] = 10 + rng.normal(0, 4, len(df))
    df["dj_chauffage"] = (15 - df["temp_prev"]).clip(lower=0)
    df["consommation"] = 50000 + 1500 * df["dj_chauffage"] + 3000 * np.sin(df["heure"] / 24 * 2 * np.pi)
    return df


def test_type_jour_holiday_is_sunday():
    df = pd.DataFrame({"jour_semaine": [0, 2, 5], "ferie": [0, 1, 0]})
    assert type_jour(df).tolist() == [0, 6, 5]


def test_design_matrix_has_stable_columns():
    df = fake()
    full = design_matrix(df)
    one_hour = design_matrix(df[df["heure"] == 3])
    assert list(full.columns) == list(one_hour.columns)
    assert full.shape[1] == 10 + 24 * (7 + 4)


def test_models_handle_missing_values_and_return_one_prediction_per_row():
    df = fake()
    df.loc[df.index[::50], ["temp_prev", "conso_j7"]] = np.nan
    train, target = df[df["mois"] <= 3], df[df["mois"] == 4]
    for model in (LinearModel(alpha=10.0), LightGBMModel(n_estimators=50)):
        yhat = model(train, target)
        assert yhat.shape == (len(target),) and np.isfinite(yhat).all()
        assert model.dernier is not None and len(model.variables) > 0


def test_models_learn_the_heating_effect():
    df = fake(end="2024-06-30")
    train, target = df[df["mois"] <= 4], df[df["mois"] >= 5]
    for model in (LinearModel(alpha=1.0), LightGBMModel(n_estimators=200)):
        err = np.abs(model(train, target) - target["consommation"]).mean()
        assert err < 0.25 * target["consommation"].std()


def test_walk_forward_respects_train_start():
    df = fake(end="2024-06-30")
    seen = []

    class Spy(LinearModel):
        def __call__(self, train, target):
            seen.append(train["date_local"].min())
            return super().__call__(train, target)

    walk_forward(df, Spy(alpha=1.0), "2024-05-01", "2024-06-30", train_start="2024-02-01")
    assert all(d >= pd.Timestamp("2024-02-01") for d in seen)


def test_analysis_helpers():
    df = fake()
    lin, gbm_ = LinearModel(alpha=1.0), LightGBMModel(n_estimators=30)
    lin(df, df.head(5))
    gbm_(df, df.head(5))
    assert len(coefficients(lin.dernier, lin.variables)) == len(lin.variables)
    imp = importances(gbm_.dernier, gbm_.variables)
    assert abs(imp.sum() - 100) < 1e-6


def test_training_skips_hours_without_temperature():
    df = fake()
    df.loc[df["mois"] == 1, "temp_prev"] = np.nan
    seen = {}

    class Spy(LinearModel):
        def build(self):
            m = super().build()
            fit = m.fit
            m.fit = lambda X, y: (seen.setdefault("n", len(X)), fit(X, y))[1]
            return m

    Spy(alpha=1.0)(df, df.head(3))
    assert seen["n"] == int((df["mois"] != 1).sum())


def test_native_lightgbm_matches_sklearn_interface():
    # Le modèle final (interface native) doit reproduire le modèle du walk-forward.
    import lightgbm as lgb
    from elecfr.models.gbm import native_params
    df = fake()
    m = LightGBMModel(n_estimators=60, num_leaves=15)
    p_sk = m(df, df)
    native, n = native_params({"n_estimators": 60, "num_leaves": 15})
    X = m.features(df)
    b = lgb.train(native, lgb.Dataset(X, label=df["consommation"].to_numpy()), num_boost_round=n)
    np.testing.assert_allclose(b.predict(X), p_sk, rtol=1e-6, atol=1e-3)
