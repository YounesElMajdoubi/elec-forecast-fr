"""Étape 0 : inspecte la table silver du Lakehouse AVANT de s'en servir.

Usage : python scripts/00_inspect_fabric.py
Prérequis : sources.toml rempli (au moins workspace) et `az login` fait.

Imprime : schéma (noms et types), période couverte, nombre de lignes par pas
de temps, taux de valeurs vides des colonnes déclarées dans sources.toml, et
la couverture par rapport à la fenêtre du projet. Colle la sortie dans la
conversation : elle suffit pour compléter sources.toml.
"""
import pandas as pd
from deltalake import DeltaTable

from elecfr import config
from elecfr.data import fabric

fab = fabric.load_sources(config.SOURCES_FILE)
spec = fab["eco2mix"]
uri = fabric.table_uri(fab, spec["table"])
print(f"Table : {uri}")
dt = DeltaTable(uri, storage_options=fabric.storage_options(uri))
print(f"Version Delta : {dt.version()}\n\n== Schéma")
fields = {f.name: str(f.type) for f in dt.schema().fields}
for name, typ in fields.items():
    print(f"  {name:<32} {typ}")

mapping = spec["colonnes"]
print("\n== Colonnes déclarées dans sources.toml")
for canon, src in mapping.items():
    print(f"  {canon:<14} → {src:<24} {'OK' if src in fields else 'ABSENTE'}")

date_col = mapping["date_heure"]
if date_col not in fields:
    raise SystemExit(f"\nLa colonne de date « {date_col} » n'existe pas : corrige sources.toml.")
cols = [c for c in {date_col, *mapping.values(), "pas_min", "priorite_nature"} if c in fields]
df = dt.to_pandas(columns=cols)
ts = fabric._to_utc(df[date_col], spec.get("fuseau", "UTC"))

print(f"\n== Période\n  {len(df):,} lignes, du {ts.min()} au {ts.max()} (UTC)")
print(f"  horodatages en double : {int(ts.duplicated().sum())}")
for c in ("pas_min", "priorite_nature"):
    if c in df:
        print(f"\n== Lignes par {c}\n" + df[c].value_counts(dropna=False).sort_index().to_string())

print("\n== Valeurs vides (%)")
present = [s for s in mapping.values() if s in df and s != date_col]
print(df[present].isna().mean().mul(100).round(2).to_string())

lo = pd.Timestamp(config.DOWNLOAD_START, tz=config.TZ).tz_convert("UTC")
hi = (pd.Timestamp(config.DOWNLOAD_END, tz=config.TZ) + pd.Timedelta(days=1)).tz_convert("UTC")
print(f"\n== Couverture de la fenêtre du projet ({config.DOWNLOAD_START} → {config.DOWNLOAD_END})")
warn = fabric.coverage_warnings(ts, lo, hi)
print("\n".join(warn) if warn else "  OK")
