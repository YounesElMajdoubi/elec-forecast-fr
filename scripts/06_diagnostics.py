"""Étape 6 : diagnostic des erreurs, pour EXPLIQUER le classement des modèles.

Question de départ : un modèle sur données publiques qui bat la prévision J-1
de RTE est un résultat surprenant. Avant de le croire, on vérifie :
1. le biais moyen (prévision - réel) : un modèle peut gagner simplement parce
   que RTE vise un niveau légèrement différent de notre « vérité » (données
   consolidées jusqu'au 31/08/2026, temps réel ensuite) ;
2. où se fait l'écart : par saison, par mois, par heure, par source de vérité ;
3. RTE corrigée de son biais mensuel : DIAGNOSTIC SEULEMENT (cette correction
   utilise le résultat du mois lui-même, impossible en conditions réelles).
   Si RTE corrigée repasse devant, l'avance du modèle tient surtout au niveau.

Lit les copies locales des prévisions (reports/previsions/), écrites par les
scripts 03 et 04. Sortie : tableaux imprimés + reports/diagnostics_*.csv.
"""
import pandas as pd

from elecfr import config
from elecfr.evaluation import mae

MODELES = ["naif_saisonnier", "rte_j1", "rte_j1_recalee", "regression_lineaire", "lightgbm", "lstm", "combinaison"]
BASCULE_TEMPS_REEL = pd.Timestamp("2026-09-01")  # projet A : temps réel à partir de cette date

df = pd.read_parquet(config.PROCESSED_DIR / "dataset_horaire.parquet")
pdir = config.REPORTS_DIR / "previsions"
out = config.REPORTS_DIR


def load(periode):
    frames = {}
    for m in MODELES:
        f = pdir / f"{m}_{periode}.parquet"
        if f.exists():
            frames[m] = pd.read_parquet(f)["prevision"]
    if not frames:
        raise SystemExit(f"Aucune prévision pour {periode} dans {pdir} : lance 03 et 04.")
    p = pd.DataFrame(frames)
    base = df.loc[p.index, ["consommation", "heure", "saison", "date_local"]]
    return base.join(p).dropna()


def mae_table(d, by):
    return d.groupby(by).apply(lambda g: pd.Series({m: mae(g["consommation"], g[m]) for m in MODELES if m in g})).round(0)


def bias_table(d, by):
    """Biais moyen (prévision - réel, MW) par groupe ; négatif = sous-estime."""
    return d.groupby(by).apply(lambda g: pd.Series({m: (g[m] - g["consommation"]).mean() for m in MODELES if m in g})).round(0)


pd.set_option("display.width", 160)
for periode in ("validation", "test"):
    d = load(periode)
    models = [m for m in MODELES if m in d]
    err = d[models].sub(d["consommation"], axis=0)

    print(f"\n######## {periode.upper()} ({len(d):,} heures communes à tous les modèles)")
    resume = pd.DataFrame({"MAE_MW": err.abs().mean(), "biais_MW": err.mean(),
                           "pct_heures_sous_estimees": (err < 0).mean() * 100}).round(1)
    print("\n== MAE et biais moyen (prévision - réel ; négatif = sous-estime)")
    print(resume.to_string())

    print("\n== MAE par saison (MW)")
    print(mae_table(d, "saison").to_string())

    mois = d["date_local"].dt.to_period("M").rename("mois")
    print("\n== MAE par mois (MW)")
    t_mois = mae_table(d.assign(mois=mois), "mois")
    print(t_mois.to_string())
    print("\n== BIAIS par mois (MW ; négatif = sous-estime)")
    b_mois = bias_table(d.assign(mois=mois), "mois")
    print(b_mois.to_string())

    if periode == "test":
        src = (d["date_local"] >= BASCULE_TEMPS_REEL).map({False: "consolidé", True: "temps réel"}).rename("verite")
        print("\n== MAE selon la source de la vérité (MW)")
        print(mae_table(d.assign(verite=src), "verite").to_string())
        print("\n== BIAIS selon la source de la vérité (MW)")
        print(bias_table(d.assign(verite=src), "verite").to_string())

    print("\n== MAE par heure (MW)")
    t_h = mae_table(d, "heure")[[m for m in ("rte_j1", "rte_j1_recalee", "lightgbm", "lstm", "combinaison") if m in models]]
    print(t_h.T.to_string())
    print("\n== BIAIS par heure (MW)")
    print(bias_table(d, "heure")[[m for m in ("rte_j1", "rte_j1_recalee", "lightgbm", "lstm", "combinaison") if m in models]].T.to_string())

    if "rte_j1" in models:
        bias_m = err["rte_j1"].groupby(mois).transform("mean")
        rte_corr = mae(d["consommation"], d["rte_j1"] - bias_m)
        print(f"\n== DIAGNOSTIC SEULEMENT : RTE corrigée de son biais de chaque mois → MAE {rte_corr:,.0f} MW")
        if "lightgbm" in models:
            print(f"   (LightGBM : {resume.loc['lightgbm', 'MAE_MW']:,.0f} MW ; "
                  f"heures où LightGBM est plus proche du réel que RTE : "
                  f"{(err['lightgbm'].abs() < err['rte_j1'].abs()).mean():.0%})")

    resume.to_csv(out / f"diagnostics_{periode}_resume.csv")
    t_mois.to_csv(out / f"diagnostics_{periode}_mois.csv")
    b_mois.to_csv(out / f"diagnostics_{periode}_biais_mois.csv")
