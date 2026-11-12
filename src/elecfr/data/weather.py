"""Température : prévisions archivées (sans fuite) et observations (analyse).

Le piège classique : utiliser la température OBSERVÉE du jour J pour prévoir la
consommation du jour J. La veille, personne ne la connaît ; le modèle aurait
l'air meilleur qu'il ne le sera jamais en production.

On utilise donc l'API « Previous Runs » d'Open-Meteo, qui garde pour chaque
heure la valeur que le modèle météo prévoyait 1 jour avant
(`temperature_2m_previous_day1`) ou 2 jours avant (`..._previous_day2`).
L'API « Archive » (réanalyse ERA5, donc observée) sert seulement à mesurer
l'écart prévision/observation, jamais comme variable du modèle.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FC_VARS = ["temperature_2m_previous_day1", "temperature_2m_previous_day2"]


def _get(url: str, params: dict) -> dict:
    for attempt in range(4):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 429:  # trop de requêtes : on patiente
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def _hourly_frame(payload: dict, cols: list[str]) -> pd.DataFrame:
    h = payload["hourly"]
    idx = pd.to_datetime(h["time"]).tz_localize("UTC")  # timezone=GMT demandé
    return pd.DataFrame({c: h[c] for c in cols}, index=idx).rename_axis("heure_utc")


def fetch_city(name, lat, lon, start, end, out_dir: Path, model: str, kind: str) -> pd.DataFrame:
    """kind = 'forecast' (Previous Runs) ou 'observed' (Archive). Cache par an."""
    parts = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        s = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01")).date()
        e = min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31")).date()
        out = out_dir / f"meteo_{kind}_{name}_{y}.parquet"
        if not out.exists():
            params = {"latitude": lat, "longitude": lon, "start_date": str(s),
                      "end_date": str(e), "timezone": "GMT"}
            if kind == "forecast":
                params |= {"hourly": ",".join(FC_VARS), "models": model}
                df = _hourly_frame(_get(PREVIOUS_RUNS_URL, params), FC_VARS)
            else:
                params |= {"hourly": "temperature_2m"}
                df = _hourly_frame(_get(ARCHIVE_URL, params), ["temperature_2m"])
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out)
            time.sleep(1)  # politesse envers une API gratuite
        parts.append(pd.read_parquet(out))
    return pd.concat(parts).sort_index()


def weighted_mean(per_city: dict[str, pd.DataFrame], weights: dict[str, float], col: str) -> pd.Series:
    """Moyenne pondérée par la population ; si une ville manque à une heure,
    les poids des autres sont renormalisés."""
    vals = pd.DataFrame({c: df[col] for c, df in per_city.items()})
    w = pd.Series(weights)[vals.columns]
    num = vals.mul(w, axis=1).sum(axis=1, min_count=1)
    den = vals.notna().mul(w, axis=1).sum(axis=1)
    return num / den.where(den > 0)


def download_temperatures(cities: dict, start, end, out_dir: Path, model: str) -> pd.DataFrame:
    """Renvoie un DataFrame horaire UTC : t_fc_d1, t_fc_d2 (prévisions pondérées)
    et t_obs (observée pondérée, pour l'analyse uniquement)."""
    weights = {c: v[2] for c, v in cities.items()}
    fc = {c: fetch_city(c, la, lo, start, end, out_dir, model, "forecast") for c, (la, lo, _) in cities.items()}
    ob = {c: fetch_city(c, la, lo, start, end, out_dir, model, "observed") for c, (la, lo, _) in cities.items()}
    return pd.DataFrame({
        "t_fc_d1": weighted_mean(fc, weights, "temperature_2m_previous_day1"),
        "t_fc_d2": weighted_mean(fc, weights, "temperature_2m_previous_day2"),
        "t_obs": weighted_mean(ob, weights, "temperature_2m"),
    })
