"""Tests sur DONNÉES FACTICES : ils vérifient que le code fait ce qu'il dit
(agrégation, absence de fuite, découpage), pas qu'un modèle est bon.
Aucun chiffre de ces tests ne doit apparaître dans le README.
"""
import numpy as np
import pandas as pd
import pytest

from elecfr.baselines import naive_seasonal
from elecfr.data import calendar, eco2mix, prices
from elecfr.evaluation import evaluate, mae, mape, walk_forward
from elecfr.features import FEATURES, FORBIDDEN, issue_time, same_local_hour_lag, temp_known_at_issue

TZ = "Europe/Paris"


def hourly_index(start, end):
    s = pd.Timestamp(start, tz=TZ).tz_convert("UTC")
    e = pd.Timestamp(end, tz=TZ).tz_convert("UTC")
    return pd.date_range(s, e, freq="h", inclusive="left")


# --- Agrégation -----------------------------------------------------------
def test_hourly_mean_not_sum():
    t = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({"date_heure": t, "consommation": [60, 62, 64, 66, 70, 70, np.nan, 70.0],
                       "prevision_j1": 1.0})
    h = eco2mix.to_hourly(df)
    assert h["consommation"].tolist() == [63.0, 70.0]
    assert h["n_cons"].tolist() == [4, 3]


def test_merge_prefers_consolidated():
    t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:30"], utc=True)
    d = pd.DataFrame({"date_heure": t, "consommation": [1.0, np.nan], "prevision_j1": 0.0})
    r = pd.DataFrame({"date_heure": t, "consommation": [9.0, 8.0], "prevision_j1": 0.0})
    m = eco2mix.merge_sources(d, r)
    assert m["consommation"].tolist() == [1.0, 8.0]


def test_prices_quarter_hour_to_hourly():
    t = pd.date_range("2025-10-01", periods=4, freq="15min", tz="UTC")
    h = prices.to_hourly(pd.Series([10, 20, 30, 40.0], index=t))
    assert h["prix"].iloc[0] == 25 and h["resolution_min"].iloc[0] == 15


# --- Absence de fuite -------------------------------------------------------
def test_issue_time_is_previous_day_10h_local():
    idx = hourly_index("2024-03-31", "2024-04-01")      # jour de passage à l'heure d'été
    it = issue_time(idx.tz_convert(TZ))
    assert (it.tz_convert(TZ) == pd.Timestamp("2024-03-30 10:00", tz=TZ)).all()


def test_temperature_uses_only_forecasts_issued_before_cutoff():
    idx = hourly_index("2024-01-10", "2024-01-11")
    d1 = pd.Series(1.0, index=idx)
    d2 = pd.Series(2.0, index=idx)
    t = temp_known_at_issue(d1, d2, lag_h=6)
    hours = idx.tz_convert(TZ).hour
    # Émission J-1 10 h, run d1 dispo 6 h après t-24 h : d1 seulement jusqu'à 4 h.
    assert (t[hours <= 4] == 1.0).all() and (t[hours > 4] == 2.0).all()


def test_forbidden_columns_not_in_features():
    assert not FORBIDDEN & set(FEATURES)
    # Aucun retard de consommation inférieur à 2 jours (J-1 n'est pas connu en entier).
    lags = [int(f.removeprefix("conso_j")) for f in FEATURES if f.startswith("conso_j")]
    assert min(lags) >= 2


def test_same_local_hour_lag_across_dst():
    idx = hourly_index("2024-03-24", "2024-04-02")
    s = pd.Series(idx.tz_convert(TZ).hour + 100 * idx.tz_convert(TZ).day, index=idx, dtype=float)
    lag = same_local_hour_lag(s, 7)
    loc = idx.tz_convert(TZ)
    i = np.where((loc.day == 1) & (loc.hour == 18))[0][0]   # 1er avril 18 h
    assert lag.iloc[i] == 25 * 100 + 18                     # 25 mars 18 h, malgré le changement d'heure


def test_walk_forward_never_trains_on_unknown_days():
    idx = hourly_index("2024-01-01", "2024-04-01")
    df = pd.DataFrame({"consommation": 1.0, "conso_j7": 1.0, "heure": 0, "saison": "hiver"}, index=idx)
    df["date_local"] = idx.tz_convert(TZ).normalize().tz_localize(None)
    seen = []

    def spy(train, target):
        seen.append((train["date_local"].max(), target["date_local"].min()))
        return np.zeros(len(target))

    walk_forward(df, spy, "2024-02-01", "2024-03-31")
    assert len(seen) == 2
    for last_train, first_target in seen:
        assert last_train <= first_target - pd.Timedelta(days=2)


# --- Métriques et références --------------------------------------------------
def test_metrics():
    assert mae([100, 200], [110, 190]) == 10
    assert mape([100, 200], [110, 190]) == pytest.approx(7.5)


def test_evaluate_reports_excluded_hours():
    idx = hourly_index("2024-01-08", "2024-01-09")
    df = pd.DataFrame({"consommation": 100.0, "heure": idx.tz_convert(TZ).hour, "saison": "hiver"}, index=idx)
    pred = pd.Series(110.0, index=idx)
    pred.iloc[0] = np.nan
    g = evaluate(df, pred)["global"].iloc[0]
    assert g["MAE_MW"] == 10 and g["n_exclues"] == 1 and g["n_heures"] == 23


def test_naive_is_lag_7():
    t = pd.DataFrame({"conso_j7": [5.0, 6.0]})
    assert naive_seasonal(None, t).tolist() == [5.0, 6.0]


# --- Calendrier ----------------------------------------------------------------
def test_school_zone_count_and_bridges():
    sched = pd.DataFrame({
        "description": ["Toussaint"] * 4,
        "zones": ["Zone A", "Zone B", "Zone C", "Zone C"],
        "start_date": ["2023-10-20T22:00:00+00:00"] * 4,
        "end_date": ["2023-11-05T23:00:00+00:00"] * 3 + ["2023-11-05T23:00:00+00:00"],
    })
    days = pd.date_range("2023-10-20", "2023-11-06")
    c = calendar.school_holiday_zone_count(sched, days)
    assert c[pd.Timestamp("2023-10-20")] == 0
    assert c[pd.Timestamp("2023-10-21")] == 3
    assert c[pd.Timestamp("2023-11-06")] == 0
    hols = calendar.public_holidays([2024])
    b = calendar.bridge_days(pd.date_range("2024-05-06", "2024-05-10"), hols)
    assert b[pd.Timestamp("2024-05-10")]          # vendredi après l'Ascension (jeudi 9 mai)
    assert not b[pd.Timestamp("2024-05-06")]


def test_rte_recalee_uses_only_past_bias():
    from elecfr.baselines import rte_day_ahead_recalee
    idx = hourly_index("2024-01-01", "2024-03-01")
    loc = idx.tz_convert(TZ)
    df = pd.DataFrame({"consommation": 1000.0, "heure": loc.hour,
                       "date_local": loc.normalize().tz_localize(None)}, index=idx)
    # RTE sous-estime de 50 MW à midi, et de 0 ailleurs, dans le passé...
    df["rte_j1"] = df["consommation"] - 50 * (df["heure"] == 12)
    train = df[df["date_local"] < pd.Timestamp("2024-02-01")]
    target = df[df["date_local"] >= pd.Timestamp("2024-02-01")].copy()
    # ... et le futur (inconnu à l'émission) est très différent : il ne doit pas compter.
    target["consommation"] = 9999.0
    pred = pd.Series(rte_day_ahead_recalee(train, target), index=target.index)
    assert (pred[target["heure"] == 12] == 1000.0).all()          # biais de -50 corrigé
    assert (pred[target["heure"] != 12] == target["rte_j1"][target["heure"] != 12]).all()
