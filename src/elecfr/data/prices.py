"""Prix day-ahead de la zone France (ENTSO-E Transparency Platform).

Jeton : variable d'environnement ENTSOE_TOKEN (fichier .env, jamais commité).

Attention : depuis le 1er octobre 2025 (jour de livraison), le marché
day-ahead européen fixe un prix par QUART D'HEURE (96 prix par jour) au lieu
d'un prix par heure. On ramène tout à l'heure par une moyenne simple des quatre
quarts d'heure, et on garde la résolution d'origine dans `resolution_min`.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from elecfr.config import TZ


def _client():
    from dotenv import load_dotenv
    from entsoe import EntsoePandasClient

    load_dotenv()
    token = os.environ.get("ENTSOE_TOKEN")
    if not token:
        raise RuntimeError("ENTSOE_TOKEN manquant : ajoute-le dans le fichier .env")
    return EntsoePandasClient(api_key=token)


def download_prices(start: str, end: str, out_dir: Path) -> pd.Series:
    """Télécharge par année (limite de l'API), met en cache, renvoie la série
    brute (index UTC, résolution d'origine)."""
    client = _client()
    parts = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        out = out_dir / f"prix_fr_{y}.parquet"
        if not out.exists():
            s = max(pd.Timestamp(start, tz=TZ), pd.Timestamp(f"{y}-01-01", tz=TZ))
            e = min(pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1), pd.Timestamp(f"{y + 1}-01-01", tz=TZ))
            ser = client.query_day_ahead_prices("FR", start=s, end=e)
            out_dir.mkdir(parents=True, exist_ok=True)
            ser.rename("prix").to_frame().to_parquet(out)
        parts.append(pd.read_parquet(out)["prix"])
    raw = pd.concat(parts).sort_index()
    raw = raw[~raw.index.duplicated()]
    raw.index = raw.index.tz_convert("UTC")
    return raw


def to_hourly(raw: pd.Series) -> pd.DataFrame:
    """Moyenne horaire des prix (€/MWh) + résolution d'origine en minutes."""
    step = raw.index.to_series().diff().dt.total_seconds().div(60)
    res = step.groupby(raw.index.floor("h")).min()
    out = raw.groupby(raw.index.floor("h")).mean().to_frame("prix")
    out["resolution_min"] = res
    out.index.name = "heure_utc"
    return out
