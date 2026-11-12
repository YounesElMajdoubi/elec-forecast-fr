"""Lecteur Fabric testé sur une table Delta LOCALE qui imite le Lakehouse :
mêmes fichiers (Parquet + _delta_log), seul l'emplacement change."""
import numpy as np
import pandas as pd
import pytest
from deltalake import write_deltalake

from elecfr.data import eco2mix, fabric


def make_fab(tmp_path, df, fuseau="UTC", cols=None):
    write_deltalake(str(tmp_path / "silver_eco2mix"), df)
    return {"chemin_local": str(tmp_path), "eco2mix": {
        "table": "silver_eco2mix", "fuseau": fuseau,
        "colonnes": cols or {"date_heure": "ts", "consommation": "conso_mw", "prevision_j1": "prev_j1_mw"}}}


def test_column_mapping_and_hourly(tmp_path):
    ts = pd.date_range("2024-01-10", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({"ts": ts, "conso_mw": [60.0, 62, 64, 66, 70, 70, 70, 70], "prev_j1_mw": 1.0})
    out = fabric.read_eco2mix(make_fab(tmp_path, df), "2024-01-01", "2024-01-31")
    assert {"date_heure", "consommation", "prevision_j1"} <= set(out.columns)
    h = eco2mix.to_hourly(out)
    assert h["consommation"].tolist() == [63.0, 70.0]


def test_naive_local_time_across_autumn_dst(tmp_path):
    # 27 octobre 2024 : 2 h–3 h existe deux fois en heure de Paris.
    utc = pd.date_range("2024-10-26 22:00", "2024-10-27 04:00", freq="15min", tz="UTC", inclusive="left")
    naive_local = utc.tz_convert("Europe/Paris").tz_localize(None)
    df = pd.DataFrame({"ts": naive_local, "conso_mw": np.arange(len(utc), dtype=float), "prev_j1_mw": 0.0})
    out = fabric.read_eco2mix(make_fab(tmp_path, df, fuseau="Europe/Paris"), "2024-10-26", "2024-10-28")
    assert out["date_heure"].tolist() == list(utc)          # aucun quart d'heure perdu ni dupliqué


def test_duplicates_keep_row_with_consumption(tmp_path):
    ts = pd.to_datetime(["2024-01-10 00:00", "2024-01-10 00:00"], utc=True)
    df = pd.DataFrame({"ts": ts, "conso_mw": [np.nan, 5.0], "prev_j1_mw": 0.0})
    out = fabric.read_eco2mix(make_fab(tmp_path, df), "2024-01-01", "2024-01-31")
    assert out["consommation"].tolist() == [5.0]


def test_missing_required_column_is_explicit(tmp_path):
    df = pd.DataFrame({"ts": pd.date_range("2024-01-10", periods=2, freq="h", tz="UTC"), "conso_mw": 1.0})
    with pytest.raises(ValueError, match="prev_j1_mw"):
        fabric.read_eco2mix(make_fab(tmp_path, df), "2024-01-01", "2024-01-31")


def test_coverage_warning_when_history_starts_late():
    lo = pd.Timestamp("2021-10-01", tz="Europe/Paris").tz_convert("UTC")
    hi = pd.Timestamp("2026-11-01", tz="Europe/Paris").tz_convert("UTC")
    late = pd.Series(pd.date_range("2024-01-01", "2026-10-31 23:30", freq="30min", tz="UTC"))
    w = fabric.coverage_warnings(late, lo, hi)
    assert len(w) == 1 and "2024-01-01" in w[0] and "2021-10-01" in w[0]
    full = pd.Series(pd.date_range(lo, hi, freq="30min", inclusive="left"))
    assert fabric.coverage_warnings(full, lo, hi) == []


def test_mixed_30_and_15_min_steps():
    # Historique au pas de 30 min puis temps réel au pas de 15 min, comme le silver du projet A.
    ts = pd.date_range("2026-08-31 23:00", periods=2, freq="30min", tz="UTC").append(
        pd.date_range("2026-09-01 00:00", periods=4, freq="15min", tz="UTC"))
    df = pd.DataFrame({"date_heure": ts, "consommation": [50.0, 52, 60, 60, 64, 64], "prevision_j1": 0.0})
    h = eco2mix.to_hourly(df)
    assert h["consommation"].tolist() == [51.0, 62.0]
    assert h["n_cons"].tolist() == [2, 4]


def test_onelake_uri():
    fab = {"workspace": "ws", "lakehouse": "lh", "schema": "dbo"}
    uri = fabric.table_uri(fab, "silver_eco2mix")
    assert uri == "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/dbo/silver_eco2mix"
    assert fabric.storage_options(uri)["use_azure_cli"] == "true"
