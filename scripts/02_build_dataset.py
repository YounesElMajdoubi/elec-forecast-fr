"""Étape 2 : construit data/processed/dataset_horaire.parquet et imprime un
rapport de qualité. S'arrête si la couverture est insuffisante : mieux vaut
une erreur claire qu'un modèle entraîné sur des trous.
"""
import pandas as pd

from elecfr import config
from elecfr.data import calendar, eco2mix, prices
from elecfr.features import FEATURES, build_dataset
from elecfr.evaluation import period

raw = config.RAW_DIR
elec = eco2mix.to_hourly(pd.read_parquet(raw / "eco2mix_fusion.parquet"))
temps = pd.read_parquet(raw / "temperatures.parquet")
school = pd.read_parquet(raw / "calendrier_scolaire.parquet")
years = range(pd.Timestamp(config.DOWNLOAD_START).year, pd.Timestamp(config.DOWNLOAD_END).year + 2)

# Grille horaire complète en UTC : les heures absentes apparaissent en NaN
# au lieu de disparaître silencieusement.
start = pd.Timestamp(config.DOWNLOAD_START, tz=config.TZ).tz_convert("UTC")
end = (pd.Timestamp(config.DOWNLOAD_END, tz=config.TZ) + pd.Timedelta(days=1)).tz_convert("UTC")
grid = pd.date_range(start, end, freq="h", inclusive="left", name="heure_utc")
elec = elec.reindex(grid)

df = build_dataset(elec, temps, calendar.public_holidays(years), school)

price_file = raw / "prix_brut.parquet"
if price_file.exists():
    px = prices.to_hourly(pd.read_parquet(price_file)["prix"])
    df = df.join(px)

config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(config.PROCESSED_DIR / "dataset_horaire.parquet")

print("== Rapport de qualité")
print(f"Heures : {len(df):,}  du {df.index.min()} au {df.index.max()} (UTC)")
problems = []
for name, (a, b) in {"train": (config.TRAIN_START, config.TRAIN_END),
                     "validation": (config.VAL_START, config.VAL_END),
                     "test": (config.TEST_START, config.TEST_END)}.items():
    part = period(df, a, b)
    cov = part[["consommation", "rte_j1", "temp_prev", "conso_j7"] + (["prix"] if "prix" in df else [])].notna().mean()
    print(f"\n[{name}] {a} → {b} : {len(part):,} heures ; couverture :")
    print(cov.round(4).to_string())
    for col, v in cov.items():
        if v < config.MIN_COVERAGE:
            problems.append(f"{name}/{col} = {v:.1%}")

print("\nÉcart prévision météo / observation (°C), sur toute la période :")
err = (df["temp_prev"] - df["temp_obs"]).abs()
print(f"  erreur absolue moyenne : {err.mean():.2f} ; 95e centile : {err.quantile(.95):.2f}")

# Où manque la température « connue la veille » ? (mois avec au moins une heure vide)
miss = df["temp_prev"].isna().groupby(df["date_local"].dt.to_period("M")).sum()
miss = miss[miss > 0]
if len(miss):
    print("\nHeures sans température prévue, par mois :")
    print(miss.to_string())
    obs_ok = df.loc[df["temp_prev"].isna(), "temp_obs"].notna().mean()
    print(f"  (la température observée existe pour {obs_ok:.0%} de ces heures : trou de l'archive de prévisions)")
# Nombre de valeurs de consommation par heure : 4 (pas de 15 min), 2 (30 min),
# 1 (table déjà horaire). Une heure à 1 au milieu d'une période à 4 signale un trou.
print("\nValeurs de consommation agrégées par heure (nombre d'heures) :")
print(df["n_cons"].value_counts(dropna=False).sort_index().to_string())
assert not set(FEATURES) - set(df.columns), "variable manquante"
if problems:
    raise SystemExit("Couverture insuffisante : " + ", ".join(problems))
print("\nOK : jeu de données prêt.")
