"""Étape 10 (volet 2) : inspecte le fichier de prix d'Ember AVANT de s'en servir.

1. Télécharge à la main (navigateur) :
   https://files.ember-energy.org/public-downloads/price/outputs/european_wholesale_electricity_price_data_hourly.zip
2. Place-le dans data/raw/ember/ (sans le décompresser).
3. Lance : python scripts/10_inspect_ember.py   et colle la sortie.

Imprime : fichiers du ZIP, colonnes, colonnes reconnues, période couverte,
pas de temps, valeurs manquantes, prix négatifs, extrêmes, et la couverture
de la fenêtre du projet.
"""
import zipfile

import pandas as pd

from elecfr import config
from elecfr.data import ember

zips = sorted((config.RAW_DIR / "ember").glob("*.zip"))
if not zips:
    raise SystemExit("Aucun ZIP dans data/raw/ember/ : télécharge le fichier horaire d'Ember (voir l'en-tête du script).")
zp = zips[-1]
print(f"Fichier : {zp.name}")
with zipfile.ZipFile(zp) as zf:
    names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    print(f"{len(names)} fichiers CSV ; fichier France détecté : {ember.pick_france_member(names)}")
    for n in names[:5]:
        print("  ", n)

s = ember.read_france(zp)
a = s.attrs
print(f"\nLu : {a['fichier']}  (date : « {a['colonne_date']} », prix : « {a['colonne_prix']} », "
      f"pays : « {a['colonne_pays']} »)")
print(f"{len(s):,} valeurs, du {s.index.min()} au {s.index.max()} (UTC)")
pas = s.index.to_series().diff().dt.total_seconds().div(60).value_counts().head(3)
print("Pas de temps (minutes : nombre) :", pas.to_dict())
print(f"Valeurs manquantes : {s.isna().sum()} ; prix négatifs : {(s < 0).sum()} heures "
      f"({(s < 0).mean():.1%}) ; min {s.min():.1f} ; max {s.max():.1f} €/MWh")
print("\nPrix moyen par année (€/MWh) :")
print(s.groupby(s.index.year).agg(["mean", "min", "max", "count"]).round(1).to_string())

lo = pd.Timestamp(config.DOWNLOAD_START, tz=config.TZ).tz_convert("UTC")
hi = (pd.Timestamp(config.DOWNLOAD_END, tz=config.TZ) + pd.Timedelta(days=1)).tz_convert("UTC")
attendu = len(pd.date_range(lo, hi, freq="h", inclusive="left"))
dans = s[(s.index >= lo) & (s.index < hi)].notna().sum()
print(f"\nCouverture de la fenêtre du projet : {dans:,} / {attendu:,} heures ({dans / attendu:.1%})")
