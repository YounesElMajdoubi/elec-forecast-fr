"""Lecture des données éCO2mix depuis le Lakehouse Fabric (projet A).

Une table de Lakehouse est une table Delta : des fichiers Parquet plus un
journal de transactions (_delta_log). La bibliothèque `deltalake` sait la lire
directement dans OneLake, sans Spark et sans passer par Fabric.

Adresse d'une table :
    abfss://<espace de travail>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse/Tables/<table>
(avec `Tables/<schéma>/<table>` si les schémas sont activés sur le Lakehouse).

Authentification : la connexion de l'Azure CLI (`az login`, avec le compte
qui a accès à l'espace Fabric). Aucun secret n'est écrit dans le dépôt.

Les noms de tables et de colonnes du projet A sont dans `sources.toml` :
ce module renomme tes colonnes vers les noms « canoniques » attendus par la
suite du code (date_heure, consommation, prevision_j1...), puis la suite du
pipeline est identique à la source API.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd

from elecfr.data.eco2mix import OPTIONAL, REQUIRED

ONELAKE_HOST = "onelake.dfs.fabric.microsoft.com"


def load_sources(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : copie sources.example.toml en sources.toml et remplis-le.")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    fab = cfg.get("fabric", {})
    for key in ("workspace", "lakehouse"):
        if not fab.get(key) or "A_REMPLIR" in str(fab.get(key)):
            if not fab.get("chemin_local"):
                raise ValueError(f"sources.toml : [fabric].{key} n'est pas rempli.")
    return fab


def table_uri(fab: dict, table: str) -> str:
    """Construit l'adresse OneLake de la table (ou un chemin local pour les tests)."""
    if fab.get("chemin_local"):
        return str(Path(fab["chemin_local"]) / table)
    schema = f"{fab['schema']}/" if fab.get("schema") else ""
    return f"abfss://{fab['workspace']}@{ONELAKE_HOST}/{fab['lakehouse']}.Lakehouse/Tables/{schema}{table}"


def storage_options(uri: str) -> dict:
    if uri.startswith("abfss://"):
        return {"use_azure_cli": "true", "use_fabric_endpoint": "true"}
    return {}


def _to_utc(ts: pd.Series, tz_if_naive: str) -> pd.Series:
    """Horodatages → UTC.

    - Déjà munis d'un fuseau : simple conversion.
    - Naïfs en UTC : on déclare UTC.
    - Naïfs en heure de Paris : ambigus à l'automne (2 h–3 h existe deux fois).
      `ambiguous="infer"` déduit lequel est lequel grâce à l'ordre des lignes ;
      si c'est impossible (lignes en double, désordre), on s'arrête plutôt
      que de deviner.
    """
    ts = pd.to_datetime(ts)
    if ts.dt.tz is not None:
        return ts.dt.tz_convert("UTC")
    if tz_if_naive == "UTC":
        return ts.dt.tz_localize("UTC")
    try:
        return ts.dt.tz_localize(tz_if_naive, ambiguous="infer", nonexistent="raise").dt.tz_convert("UTC")
    except Exception as exc:  # noqa: BLE001 - on veut un message clair
        raise ValueError(
            "Horodatages naïfs en heure locale impossibles à convertir sans ambiguïté "
            f"(changement d'heure) : {exc}. Préfère une colonne en UTC dans le Lakehouse."
        ) from exc


def read_eco2mix(fab: dict, start: str, end: str) -> pd.DataFrame:
    """Lit la table éCO2mix du Lakehouse et renvoie le même format que
    `eco2mix.download_all` : une ligne par horodatage (UTC), colonnes canoniques."""
    from deltalake import DeltaTable

    spec = fab["eco2mix"]
    mapping: dict[str, str] = spec["colonnes"]           # canonique -> nom dans ta table
    missing_keys = [c for c in REQUIRED if c not in mapping]
    if missing_keys:
        raise ValueError(f"sources.toml : colonnes indispensables non déclarées {missing_keys}")

    uri = table_uri(fab, spec["table"])
    dt = DeltaTable(uri, storage_options=storage_options(uri))
    present = {f.name for f in dt.schema().fields}
    absent = [src for canon, src in mapping.items() if canon in REQUIRED and src not in present]
    if absent:
        raise ValueError(f"Colonnes absentes de {spec['table']} : {absent}. Colonnes vues : {sorted(present)}")
    wanted = {c: s for c, s in mapping.items() if s in present and c in REQUIRED + OPTIONAL}
    skipped = [c for c in mapping if c not in wanted]
    if skipped:
        print(f"[fabric] colonnes ignorées (absentes ou non utilisées) : {skipped}")

    df = dt.to_pandas(columns=list(wanted.values())).rename(columns={s: c for c, s in wanted.items()})
    df["date_heure"] = _to_utc(df["date_heure"], spec.get("fuseau", "UTC"))
    lo = pd.Timestamp(start, tz="Europe/Paris").tz_convert("UTC")
    hi = (pd.Timestamp(end, tz="Europe/Paris") + pd.Timedelta(days=1)).tz_convert("UTC")
    df = df[(df["date_heure"] >= lo) & (df["date_heure"] < hi)]

    # Doublons d'horodatage (ex. temps réel + consolidé empilés) : on garde la
    # ligne qui a une consommation, et on le signale.
    n_dup = int(df["date_heure"].duplicated().sum())
    if n_dup:
        print(f"[fabric] {n_dup} horodatages en double : on garde la ligne avec consommation")
        df = (df.assign(_has=df["consommation"].notna())
                .sort_values(["date_heure", "_has"], ascending=[True, False])
                .drop_duplicates("date_heure").drop(columns="_has"))
    df["_src"] = "fabric"
    print(f"[fabric] {spec['table']} : {len(df):,} lignes, {df['date_heure'].min()} → {df['date_heure'].max()}")
    for msg in coverage_warnings(df["date_heure"], lo, hi):
        print(msg)
    return df.sort_values("date_heure").reset_index(drop=True)


def coverage_warnings(ts: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp) -> list[str]:
    """Signale si la table commence trop tard ou s'arrête trop tôt par rapport
    à la fenêtre demandée. On prévient ici ; c'est `02_build_dataset.py` qui
    refuse ensuite de continuer si la couverture est insuffisante."""
    out = []
    paris = lambda t: t.tz_convert("Europe/Paris").strftime("%Y-%m-%d")  # noqa: E731 - dates lisibles, en heure locale
    if ts.empty:
        return [f"[fabric] ATTENTION : aucune ligne entre le {paris(lo)} et le {paris(hi)}."]
    if ts.min() > lo + pd.Timedelta(days=1):
        out.append(f"[fabric] ATTENTION : la table commence le {paris(ts.min())}, "
                   f"il manque l'historique depuis le {paris(lo)} (voir README, « Compléter l'historique »).")
    if ts.max() < hi - pd.Timedelta(days=2):
        out.append(f"[fabric] ATTENTION : la table s'arrête le {paris(ts.max())}, "
                   f"avant la fin demandée ({paris(hi - pd.Timedelta(hours=1))}).")
    return out
