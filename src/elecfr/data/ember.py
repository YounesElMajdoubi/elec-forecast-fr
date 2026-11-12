"""Prix day-ahead horaires de la France, depuis le fichier public d'Ember.

Pourquoi Ember : en attendant le jeton ENTSO-E, Ember republie les prix
day-ahead européens (source d'origine : ENTSO-E) dans un fichier ZIP gratuit,
sans compte, sous licence CC BY 4.0 (citer « Ember »).
Téléchargement : https://files.ember-energy.org/public-downloads/price/outputs/european_wholesale_electricity_price_data_hourly.zip
à placer dans data/raw/ember/ (voir README).

Le format exact du fichier n'étant pas documenté sur le site, ce module
reconnaît les colonnes par leur nom (pays, date-heure UTC, prix) et
s'arrête avec un message clair s'il n'y parvient pas. Sortie : la même série
que `prices.download_prices` (index UTC, prix en €/MWh), pour pouvoir changer
de source sans rien modifier d'autre.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

FRANCE = {"france", "fra", "fr"}


def _find(columns, *must, avoid=()):
    """Première colonne dont le nom contient tous les mots de `must` (sans casse)."""
    for c in columns:
        low = c.lower()
        if all(m in low for m in must) and not any(a in low for a in avoid):
            return c
    return None


def _csv_members(zf: zipfile.ZipFile) -> list[str]:
    return [n for n in zf.namelist() if n.lower().endswith(".csv")]


def pick_france_member(names: list[str]) -> str | None:
    """Fichier propre à la France dans le ZIP, s'il existe (sinon le fichier compilé)."""
    for n in names:
        stem = Path(n).stem.lower()
        if "france" in stem or stem.split("_")[0] in FRANCE or stem.endswith("_fr") or stem.endswith("_fra"):
            return n
    return None


def read_france(zip_path: Path) -> pd.Series:
    """Lit le ZIP d'Ember et renvoie les prix horaires de la France (index UTC)."""
    with zipfile.ZipFile(zip_path) as zf:
        members = _csv_members(zf)
        if not members:
            raise ValueError(f"Aucun fichier CSV dans {zip_path}")
        member = pick_france_member(members) or max(members, key=lambda n: zf.getinfo(n).file_size)
        df = pd.read_csv(io.BytesIO(zf.read(member)))

    cols = list(df.columns)
    t_col = _find(cols, "utc") or _find(cols, "date", avoid=("local",))
    p_col = _find(cols, "price") or _find(cols, "prix")
    c_col = _find(cols, "iso") or _find(cols, "country") or _find(cols, "area") or _find(cols, "zone")
    if t_col is None or p_col is None:
        raise ValueError(f"{member} : colonnes date (UTC) ou prix introuvables. Colonnes vues : {cols}")

    if c_col is not None:
        is_fr = df[c_col].astype(str).str.strip().str.lower().isin(FRANCE)
        if not is_fr.any():
            raise ValueError(f"{member} : aucune ligne pour la France dans « {c_col} ». "
                             f"Valeurs vues : {sorted(df[c_col].astype(str).unique())[:40]}")
        df = df[is_fr]

    ts = pd.to_datetime(df[t_col], utc=True)
    s = pd.Series(pd.to_numeric(df[p_col], errors="coerce").to_numpy(), index=ts, name="prix").sort_index()
    s = s[~s.index.duplicated()]
    s.attrs = {"fichier": member, "colonne_date": t_col, "colonne_prix": p_col, "colonne_pays": c_col}
    return s
