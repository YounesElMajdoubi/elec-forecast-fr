"""Étape 1 : rassemble éCO2mix, températures, calendrier scolaire et prix.

Usage :  python scripts/01_download.py                 (éCO2mix lu dans le Lakehouse Fabric)
         python scripts/01_download.py --source api    (éCO2mix téléchargé depuis ODRÉ)
         python scripts/01_download.py --sans-prix     (si le jeton ENTSO-E n'est pas arrivé)
Les fichiers bruts vont dans data/raw/ et sont mis en cache : relancer ne
retélécharge que ce qui manque (la lecture Fabric, elle, est refaite à chaque fois).
"""
import argparse

from elecfr import config
from elecfr.data import calendar, eco2mix, fabric, prices, weather

p = argparse.ArgumentParser()
p.add_argument("--source", choices=["fabric", "api"], default=config.ECO2MIX_SOURCE)
p.add_argument("--sans-prix", action="store_true")
args = p.parse_args()

raw = config.RAW_DIR
raw.mkdir(parents=True, exist_ok=True)
s, e = config.DOWNLOAD_START, config.DOWNLOAD_END

print(f"== éCO2mix (source : {args.source})")
if args.source == "fabric":
    elec = fabric.read_eco2mix(fabric.load_sources(config.SOURCES_FILE), s, e)
else:
    elec = eco2mix.download_all(s, e, raw / "eco2mix")
elec.to_parquet(raw / "eco2mix_fusion.parquet", index=False)

print("== Températures (prévisions archivées + observées)")
temps = weather.download_temperatures(config.CITIES, s, e, raw / "meteo", config.WEATHER_MODEL)
temps.to_parquet(raw / "temperatures.parquet")

print("== Calendrier scolaire")
calendar.download_school_holidays(raw)

if not args.sans_prix:
    print("== Prix day-ahead (ENTSO-E)")
    px = prices.download_prices(s, e, raw / "prix")
    px.to_frame("prix").to_parquet(raw / "prix_brut.parquet")
print("Terminé.")
