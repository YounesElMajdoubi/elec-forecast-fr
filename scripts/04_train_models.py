"""Étape 4 : régression linéaire et LightGBM.

Protocole, identique pour chaque modèle :
1. Réglage : chaque jeu d'hyperparamètres de la grille est évalué en
   walk-forward sur la VALIDATION (nov. 2024 → oct. 2025). Chaque essai est
   un run MLflow (selection = candidat).
2. Choix : le réglage de plus faible MAE sur la validation est retenu.
   Le test n'a joué aucun rôle dans ce choix.
3. Test : ce seul réglage est évalué en walk-forward sur le TEST
   (nov. 2025 → oct. 2026), réentraîné chaque mois sur tout le passé connu
   depuis le début de l'entraînement.

Usage : python scripts/04_train_models.py                 (les deux modèles)
        python scripts/04_train_models.py --modeles lineaire
"""
print("Chargement des bibliothèques (la première fois, ça peut prendre quelques minutes)...", flush=True)

import argparse
import time

import pandas as pd

from elecfr import config, tracking
from elecfr.evaluation import walk_forward
from elecfr.models import gbm, linear, lstm

df = pd.read_parquet(config.PROCESSED_DIR / "dataset_horaire.parquet")

MODELES = {
    "lineaire": (linear.LinearModel, linear.GRID, "regression_lineaire",
                 lambda m: {"coefficients": linear.coefficients(m.dernier, m.variables)}),
    "lightgbm": (gbm.LightGBMModel, gbm.GRID, "lightgbm",
                 lambda m: {"importances": gbm.importances(m.dernier, m.variables)}),
}

# Le LSTM a besoin de tout le jeu de données (pour lire l'historique J-8 … J-2
# de chaque jour prévu) : on le construit via une petite fabrique.
MODELES["lstm"] = (lambda **params: lstm.LSTMModel(base=df, **params), lstm.GRID, "lstm",
                   lambda m: {"epoques": pd.Series(m.epoques, name="epoques")})

p = argparse.ArgumentParser()
p.add_argument("--modeles", nargs="+", choices=list(MODELES), default=list(MODELES))
args = p.parse_args()

if "lstm" in args.modeles:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch n'est pas installé : le LSTM est ignoré (pip install torch).")
        args.modeles.remove("lstm")

tracking.setup()
protocole = {"train_debut": config.TRAIN_START, "refit": "MS"}

for key in args.modeles:
    cls, grid, nom, analyse = MODELES[key]
    print(f"\n=== {nom} : réglage sur la validation ({len(grid)} essais)")
    essais = []
    for params in grid:
        t0 = time.time()
        print(f"  essai {params}", flush=True)
        pred = walk_forward(df, cls(**params), config.VAL_START, config.VAL_END,
                            train_start=config.TRAIN_START, verbose=True)
        r = tracking.log_evaluation(df, pred, modele=nom, periode="validation", famille="ml",
                                    params={**params, **protocole}, selection="candidat")
        essais.append((r["MAE_MW"], params, r["run_id"], pred))
        print(f"  {params} → MAE {r['MAE_MW']:,.0f} MW, MAPE {r['MAPE_pct']:.2f} %  ({time.time() - t0:.0f} s)")

    mae_val, best, run_id, pred_val = min(essais, key=lambda e: e[0])
    tracking.mark_selected(run_id)
    tracking.save_local(df, pred_val, nom, "validation")
    print(f"  retenu : {best} (MAE validation {mae_val:,.0f} MW)")

    print(f"=== {nom} : test avec le réglage retenu")
    model = cls(**best)
    pred = walk_forward(df, model, config.TEST_START, config.TEST_END,
                        train_start=config.TRAIN_START, verbose=True)
    r = tracking.log_evaluation(df, pred, modele=nom, periode="test", famille="ml",
                                params={**best, **protocole}, model=model.dernier,
                                extra=analyse(model))
    print(f"  test : MAE {r['MAE_MW']:,.0f} MW, MAPE {r['MAPE_pct']:.2f} %")

print("\nTerminé. Tableau complet : python scripts/05_compare.py")
