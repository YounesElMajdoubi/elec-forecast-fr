"""Étape 11 (volet 2) : jeu de prix horaire → data/processed/prix_horaires.parquet

Source des prix : Ember (par défaut, fichier dans data/raw/ember/) ou
ENTSO-E (--source entsoe, après 01_download.py avec le jeton).
Ajoute le calendrier (heure locale, jour, férié) et, pour l'exploration,
consommation, éolien, solaire et nucléaire d'éCO2mix (depuis octobre 2021).

Usage : python scripts/11_build_prices.py [--source ember|entsoe]
"""
import argparse

import pandas as pd

from elecfr import config
from elecfr import simulation
from elecfr.data import eco2mix, ember, prices

p = argparse.ArgumentParser()
p.add_argument("--source", choices=["ember", "entsoe"], default="ember")
args = p.parse_args()

if args.source == "ember":
    zips = sorted((config.RAW_DIR / "ember").glob("*.zip"))
    if not zips:
        raise SystemExit("Aucun ZIP dans data/raw/ember/ (voir scripts/10_inspect_ember.py).")
    brut = ember.read_france(zips[-1])
else:
    brut = pd.read_parquet(config.RAW_DIR / "prix_brut.parquet")["prix"]
horaire = prices.to_hourly(brut)["prix"]

# Grille horaire complète en UTC : les heures manquantes restent visibles (NaN)
grid = pd.date_range(horaire.index.min(), horaire.index.max(), freq="h", name="heure_utc")
df = horaire.reindex(grid).to_frame()
loc = grid.tz_convert(config.TZ)
df = df.join(simulation.calendrier(grid, config.TZ))   # même calendrier que la simulation et l'API
df["mois"], df["annee"] = loc.month, loc.year

# Contexte physique (exploration seulement), disponible depuis octobre 2021
elec_file = config.RAW_DIR / "eco2mix_fusion.parquet"
if elec_file.exists():
    elec = eco2mix.to_hourly(pd.read_parquet(elec_file))
    for c in ("consommation", "eolien", "solaire", "nucleaire"):
        if c in elec:
            df[c] = elec[c].reindex(grid)

config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(config.PROCESSED_DIR / "prix_horaires.parquet")
print(f"Source : {args.source} ; {df['prix'].notna().sum():,} heures de prix, "
      f"du {grid.min()} au {grid.max()} (UTC) ; heures manquantes : {df['prix'].isna().sum()}")
print("Colonnes :", ", ".join(df.columns))
