"""Étape 14 (volet 2) : télécharge les prix ENTSO-E et les compare à Ember, heure par heure.

À lancer une fois le jeton ENTSO-E dans .env. Ce script :
  1. télécharge les prix day-ahead de la France depuis ENTSO-E (cache par année
     dans data/raw/prix/, l'année en cours est complétée à chaque lancement) ;
  2. écrit data/raw/prix_brut.parquet, lu par 11_build_prices.py --source entsoe ;
  3. compare les deux sources sur leurs heures communes et liste les écarts
     dans reports/comparaison_prix_ember_entsoe.csv.

Usage : python scripts/14_compare_prices.py [--debut 2021-10-01] [--fin AAAA-MM-JJ] [--tol 0.01]
Ensuite : python scripts/11_build_prices.py --source entsoe, puis 12 et 13.
"""
import argparse

import pandas as pd

from elecfr import config
from elecfr.data import ember, prices

p = argparse.ArgumentParser()
p.add_argument("--debut", default=config.DOWNLOAD_START)
p.add_argument("--fin", default=prices.today_paris(), help="dernier jour de livraison (défaut : aujourd'hui)")
p.add_argument("--tol", type=float, default=0.01, help="écart toléré, €/MWh")
args = p.parse_args()

print(f"== Téléchargement ENTSO-E, du {args.debut} au {args.fin}")
brut = prices.download_prices(args.debut, args.fin, config.RAW_DIR / "prix")
brut.to_frame("prix").to_parquet(config.RAW_DIR / "prix_brut.parquet")
res = brut.index.to_series().diff().dt.total_seconds().div(60).groupby(brut.index.tz_convert(config.TZ).year).median()
print(f"{len(brut):,} prix bruts, du {brut.index.min()} au {brut.index.max()} (UTC)")
print("Pas médian par année (min) :", ", ".join(f"{a} : {v:.0f}" for a, v in res.items()))
entsoe_h = prices.to_hourly(brut)["prix"]

zips = sorted((config.RAW_DIR / "ember").glob("*.zip"))
if not zips:
    raise SystemExit("Prix ENTSO-E enregistrés, mais aucun ZIP Ember dans data/raw/ember/ : comparaison impossible.")
ember_h = prices.to_hourly(ember.read_france(zips[-1]))["prix"]

# Comparaison limitée à la période téléchargée chez ENTSO-E
debut, fin = entsoe_h.index.min(), entsoe_h.index.max()
ember_h = ember_h[(ember_h.index >= debut) & (ember_h.index <= fin)]
cmp = prices.compare_sources(ember_h, entsoe_h, tol=args.tol)
r = cmp["resume"]

print(f"\n== Comparaison Ember / ENTSO-E (écart toléré : {args.tol} €/MWh)")
print(f"Heures communes : {r['heures_communes']:,}")
print(f"Heures seulement chez Ember : {r['heures_seulement_ember']} ; seulement chez ENTSO-E : {r['heures_seulement_entsoe']}")
print(f"Heures identiques : {r['part_identiques']:.2%}")
print(f"Écart absolu moyen : {r['ecart_moyen']:.3f} €/MWh ; écart moyen signé (ENTSO-E − Ember) : {r['ecart_moyen_signe']:+.3f}")
print(f"Écart maximal : {r['ecart_max']:.2f} €/MWh, le {r['heure_ecart_max']} (UTC)")

print("\nPar année (heure de Paris) :")
print(cmp["par_annee"].round(3).to_string())

quart = pd.Timestamp("2025-10-01", tz=config.TZ).tz_convert("UTC")
d = cmp["differences"]
print(f"\nHeures qui diffèrent : {len(d)} (avant le 01/10/2025 : {(d.index < quart).sum()} ; "
      f"depuis, prix au quart d'heure : {(d.index >= quart).sum()})")
if len(d):
    print("Les 10 plus grands écarts :")
    print(d.reindex(d["ecart"].abs().sort_values(ascending=False).index).head(10).round(2).to_string())

# Prix négatifs : la moyenne de quatre quarts d'heure peut être à peine sous zéro
# (par exemple −0,0025 €/MWh) ; arrondie au centime, elle devient 0.
communs = pd.concat([ember_h.rename("ember"), entsoe_h.rename("entsoe")], axis=1, sort=True).dropna()
neg_m, neg_e = (communs["ember"] < 0).sum(), (communs["entsoe"] < 0).sum()
presque = ((communs["entsoe"] < 0) & (communs["entsoe"] > -0.01) & (communs["ember"] >= 0)).sum()
print(f"\nHeures négatives (heures communes) : Ember {neg_m}, ENTSO-E {neg_e} ; "
      f"dont {presque} entre −0,01 et 0 €/MWh chez ENTSO-E et nulles chez Ember")

seul_m = cmp["seulement_ember"]
if len(seul_m):
    print("\nHeures absentes chez ENTSO-E mais présentes chez Ember (heure de Paris) :")
    print(", ".join(str(t.tz_convert(config.TZ)) for t in seul_m[:20]))

fin_ember = ember_h.index.max()
seul_e = cmp["seulement_entsoe"]
apres = (seul_e > fin_ember).sum()
print(f"\nHeures seulement chez ENTSO-E : {len(seul_e) - apres} pendant la période d'Ember, "
      f"{apres} après sa dernière heure ({fin_ember.tz_convert(config.TZ)})")
seul_e = seul_e[seul_e <= fin_ember]
if len(seul_e):
    print("Heures absentes chez Ember mais présentes chez ENTSO-E (heure de Paris) :")
    print(", ".join(str(t.tz_convert(config.TZ)) for t in seul_e[:20]))

config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
out = config.REPORTS_DIR / "comparaison_prix_ember_entsoe.csv"
d.to_csv(out)
print(f"\nDétail des écarts : {out}")
print("Suite : python scripts/11_build_prices.py --source entsoe, puis 12_explore_prices.py et 13_simulate_prices.py")
