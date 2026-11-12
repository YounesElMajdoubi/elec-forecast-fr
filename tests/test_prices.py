"""Prix ENTSO-E : passage à l'heure, cache par année, comparaison avec Ember.
Aucun appel réseau : le client ENTSO-E est remplacé par un faux client."""
import numpy as np
import pandas as pd

from elecfr.data import prices


class FakeClient:
    """Renvoie un prix par quart d'heure, égal au numéro de l'heure UTC."""
    def __init__(self):
        self.calls = []

    def query_day_ahead_prices(self, zone, start, end):
        self.calls.append((start, end))
        idx = pd.date_range(start, end, freq="15min").tz_convert("Europe/Paris")   # fin incluse, comme entsoe-py
        return pd.Series(idx.tz_convert("UTC").hour.astype(float), index=idx)


def test_to_hourly_averages_quarter_hours():
    idx = pd.date_range("2025-10-01", periods=8, freq="15min", tz="UTC")
    h = prices.to_hourly(pd.Series([10, 20, 30, 40, -4, -4, 0, 0], index=idx, dtype=float))
    assert h["prix"].tolist() == [25.0, -2.0]
    assert h["resolution_min"].tolist() == [15.0, 15.0]


def test_to_hourly_drops_incomplete_hour():
    idx = pd.date_range("2026-11-12 22:00", periods=5, freq="15min", tz="UTC")   # 22 h complète, 23 h : 1 quart d'heure
    h = prices.to_hourly(pd.Series([1, 2, 3, 4, 99], index=idx, dtype=float))
    assert h.index.tolist() == [pd.Timestamp("2026-11-12 22:00", tz="UTC")]
    assert h["prix"].tolist() == [2.5]


def test_to_hourly_keeps_hourly_series():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    h = prices.to_hourly(pd.Series([5.0, 6.0, 7.0], index=idx))
    assert h["prix"].tolist() == [5.0, 6.0, 7.0] and h["resolution_min"].tolist() == [60.0] * 3


def test_cache_refreshes_incomplete_year_only(tmp_path):
    fake = FakeClient()
    prices.download_prices("2024-12-30", "2025-01-02", tmp_path, client=fake)
    assert len(fake.calls) == 2                         # 2024 et 2025
    prices.download_prices("2024-12-30", "2025-01-02", tmp_path, client=fake)
    assert len(fake.calls) == 2                         # tout est en cache
    raw = prices.download_prices("2024-12-30", "2025-01-05", tmp_path, client=fake)
    assert len(fake.calls) == 3                         # seule 2025 est complétée
    assert raw.index.max() >= pd.Timestamp("2025-01-05 22:00", tz="UTC")
    assert str(raw.index.tz) == "UTC" and raw.index.is_unique
    assert raw.index.max() == pd.Timestamp("2025-01-05 22:45", tz="UTC")   # pas le quart d'heure de fin


class YearSplittingClient(FakeClient):
    """Imite entsoe-py : au-delà d'un an (marge d'un jour comprise), la demande est
    découpée en blocs annuels et le point de jonction est perdu."""
    def query_day_ahead_prices(self, zone, start, end):
        ser = super().query_day_ahead_prices(zone, start, end)
        pad = start - pd.Timedelta(days=1)
        if end + pd.Timedelta(days=1) - pad > pd.Timedelta(days=365):
            ser = ser.drop(pad + pd.DateOffset(years=1), errors="ignore")
        return ser


def test_monthly_requests_avoid_year_junction_loss(tmp_path):
    fake = YearSplittingClient()
    raw = prices.download_prices("2023-01-01", "2023-12-31", tmp_path, client=fake)
    assert len(fake.calls) == 12
    assert all(e - s <= pd.Timedelta(days=32) for s, e in fake.calls)
    junction = pd.Timestamp("2023-12-31 00:00", tz="Europe/Paris").tz_convert("UTC")
    assert junction in raw.index
    assert len(prices.to_hourly(raw)) == 8760                      # aucune heure perdue


def test_compare_sources_counts_and_differences():
    idx = pd.date_range("2025-12-31 20:00", periods=6, freq="h", tz="UTC")
    a = pd.Series([50.0, 51.0, np.nan, 53.0, 54.0, 55.0], index=idx)
    b = pd.Series([50.0, 51.004, 52.0, 63.0, 54.0, np.nan], index=idx)
    c = prices.compare_sources(a, b, tol=0.01)
    r = c["resume"]
    assert r["heures_communes"] == 4
    assert r["heures_seulement_ember"] == 1 and r["heures_seulement_entsoe"] == 1
    assert r["part_identiques"] == 0.75
    assert r["ecart_max"] == 10.0 and r["heure_ecart_max"] == idx[3]
    assert c["differences"].index.tolist() == [idx[3]]
    # 2025-12-31 20 h et 21 h UTC sont en 2025 à Paris ; 23 h et 0 h UTC en 2026
    assert c["par_annee"].loc[2025, "heures"] == 2 and c["par_annee"].loc[2026, "heures"] == 2
