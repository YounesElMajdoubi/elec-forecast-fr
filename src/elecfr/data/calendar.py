"""Calendrier : jours fériés, ponts, vacances scolaires.

- Jours fériés : bibliothèque `holidays` (hors ligne).
- Vacances scolaires : la bibliothèque `holidays` ne les fournit pas pour la
  France. On les prend dans le jeu officiel du ministère de l'Éducation
  nationale (« fr-en-calendrier-scolaire », data.education.gouv.fr, sans clé).
  Variable produite : nombre de zones (A, B, C) en vacances ce jour-là, de 0 à 3,
  parce que l'effet sur la consommation croît avec la part du pays en congés.
"""
from __future__ import annotations

import io
from pathlib import Path

import holidays
import pandas as pd
import requests

SCHOOL_URL = ("https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
              "fr-en-calendrier-scolaire/exports/csv")
ZONES = ["Zone A", "Zone B", "Zone C"]


def public_holidays(years) -> set:
    return set(holidays.France(years=list(years)).keys())


def download_school_holidays(out_dir: Path) -> pd.DataFrame:
    out = out_dir / "calendrier_scolaire.parquet"
    if not out.exists():
        r = requests.get(SCHOOL_URL, params={"delimiter": ";"}, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep=";")
        need = {"description", "start_date", "end_date", "zones"}
        if not need <= set(df.columns):
            raise RuntimeError(f"Colonnes inattendues : {list(df.columns)}")
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    return pd.read_parquet(out)


def school_holiday_zone_count(sched: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """Pour chaque jour (naïf, local), nombre de zones A/B/C en vacances.

    Convention : un jour d est en vacances si start_date <= d < end_date, en
    dates locales (start_date = fin des cours, end_date = jour de reprise).
    Les lignes enseignants/élèves en double sont dédoublonnées par zone.
    """
    s = sched[sched["zones"].isin(ZONES)].copy()
    for c in ("start_date", "end_date"):
        s[c] = pd.to_datetime(s[c], utc=True).dt.tz_convert("Europe/Paris").dt.normalize().dt.tz_localize(None)
    s = s.drop_duplicates(["zones", "start_date", "end_date"])
    count = pd.Series(0, index=days, dtype="int64")
    for zone, g in s.groupby("zones"):
        on = pd.Series(False, index=days)
        for st, en in zip(g["start_date"], g["end_date"]):
            on |= (days >= st) & (days < en)
        count += on.astype("int64")
    return count


def bridge_days(days: pd.DatetimeIndex, hols: set) -> pd.Series:
    """Pont : lundi juste avant un mardi férié, ou vendredi juste après un jeudi férié."""
    d = pd.Series(days, index=days)
    nxt = (d + pd.Timedelta(days=1)).dt.date.isin(hols)
    prv = (d - pd.Timedelta(days=1)).dt.date.isin(hols)
    return ((d.dt.day_of_week == 0) & nxt) | ((d.dt.day_of_week == 4) & prv)
