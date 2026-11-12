"""Téléchargement et agrégation horaire des données éCO2mix (RTE, via ODRÉ).

Deux jeux ODRÉ :
- « eco2mix-national-cons-def » : données consolidées / définitives (passé) ;
- « eco2mix-national-tr »       : temps réel (depuis fin mai 2022).
On télécharge les deux et on privilégie le consolidé là où il existe.

Les valeurs sont des PUISSANCES moyennes en MW sur des pas de 15 ou 30 min.
Pour passer à l'heure, on fait donc une MOYENNE (pas une somme) : une puissance
de 60 000 MW pendant quatre quarts d'heure reste 60 000 MW en moyenne horaire.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
DATASETS = {"def": "eco2mix-national-cons-def", "tr": "eco2mix-national-tr"}

# Champs indispensables : sans eux, le projet ne peut pas tourner.
REQUIRED = ["date_heure", "consommation", "prevision_j1"]
# Champs utiles (exploration, volet prix) : pris s'ils existent.
OPTIONAL = [
    "prevision_j", "nucleaire", "eolien", "solaire", "hydraulique", "gaz",
    "charbon", "fioul", "bioenergies", "ech_physiques",
]


def available_fields(dataset_id: str) -> list[str]:
    """Lit le schéma du jeu ODRÉ pour ne demander que des champs existants."""
    r = requests.get(f"{BASE}/{dataset_id}", timeout=60)
    r.raise_for_status()
    meta = r.json()
    fields = meta.get("fields") or meta.get("dataset", {}).get("fields", [])
    names = [f["name"] for f in fields]
    if not names:
        raise RuntimeError(f"Schéma vide pour {dataset_id} : réponse inattendue de l'API.")
    return names


def download_year(dataset_id: str, year: int, out_dir: Path, fields: list[str]) -> Path:
    """Télécharge une année (export CSV, horodatage en UTC). Met en cache."""
    out = out_dir / f"{dataset_id}_{year}.parquet"
    if out.exists():
        return out
    where = f"date_heure >= date'{year}-01-01' AND date_heure < date'{year + 1}-01-01'"
    params = {
        "select": ",".join(fields),
        "where": where,
        "timezone": "UTC",
        "delimiter": ";",
    }
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE}/{dataset_id}/exports/csv", params=params, timeout=600)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    df = pd.read_csv(io.StringIO(r.text), sep=";")
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def download_all(start: str, end: str, out_dir: Path) -> pd.DataFrame:
    """Télécharge les deux jeux sur [start, end] et les fusionne (sous-horaire)."""
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    frames = {}
    for key, ds in DATASETS.items():
        names = available_fields(ds)
        missing = [f for f in REQUIRED if f not in names]
        if missing:
            raise RuntimeError(f"{ds} : champs indispensables absents {missing}. Champs vus : {names}")
        fields = REQUIRED + [f for f in OPTIONAL if f in names]
        absent = [f for f in OPTIONAL if f not in names]
        if absent:
            print(f"[{ds}] champs optionnels absents (ignorés) : {absent}")
        parts = [pd.read_parquet(download_year(ds, y, out_dir, fields)) for y in years]
        frames[key] = pd.concat(parts, ignore_index=True)
        print(f"[{ds}] {len(frames[key]):,} lignes")
    return merge_sources(frames["def"], frames["tr"])


def merge_sources(df_def: pd.DataFrame, df_tr: pd.DataFrame) -> pd.DataFrame:
    """Priorité au consolidé : pour chaque horodatage, on garde la ligne « def »
    si sa consommation est renseignée, sinon la ligne « tr »."""
    a = df_def.assign(_src="def", _prio=0)
    b = df_tr.assign(_src="tr", _prio=1)
    both = pd.concat([a, b], ignore_index=True)
    both["_has_cons"] = both["consommation"].notna()
    both = both.sort_values(["date_heure", "_has_cons", "_prio"], ascending=[True, False, True])
    out = both.drop_duplicates("date_heure", keep="first").drop(columns=["_prio", "_has_cons"])
    return out.sort_values("date_heure").reset_index(drop=True)


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Moyenne horaire (UTC) de chaque colonne numérique.

    On travaille en UTC : chaque heure UTC dure exactement 60 min, alors qu'en
    heure locale certaines journées ont 23 ou 25 heures (changements d'heure).
    `n_cons` compte les valeurs de consommation ayant servi à la moyenne, pour
    repérer les heures incomplètes.
    """
    df = df.copy()
    df["heure_utc"] = df["date_heure"].dt.floor("h")
    num = df.select_dtypes("number").columns
    hourly = df.groupby("heure_utc")[list(num)].mean()
    hourly["n_cons"] = df.groupby("heure_utc")["consommation"].count()
    hourly.index.name = "heure_utc"
    return hourly
