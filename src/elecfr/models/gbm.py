"""LightGBM : boosting d'arbres de décision.

Principe : on empile des centaines de petits arbres. Chacun corrige l'erreur
laissée par les précédents (d'où « boosting »). Contrairement à la régression
linéaire, un arbre découvre seul les interactions (« s'il fait moins de 5 °C
ET qu'il est 19 h ET que c'est un jour ouvré... ») : on lui donne les variables
brutes, sans fabriquer de croisements.

Hyperparamètres réglés sur la validation :
- n_estimators : nombre d'arbres ;
- learning_rate : part de la correction de chaque arbre qu'on garde
  (petit = plus prudent, il faut plus d'arbres) ;
- num_leaves : taille de chaque arbre, donc complexité des interactions.
Les autres sont fixés à des valeurs prudentes : min_child_samples (au moins
50 heures par feuille, contre le sur-apprentissage), sous-échantillonnage des
lignes et des colonnes (chaque arbre voit 80 %), graine fixe pour la
reproductibilité.

Pas d'arrêt précoce (early stopping) : il faudrait un jeu de contrôle à
l'intérieur de chaque fenêtre d'entraînement ; le nombre d'arbres est choisi
sur la période de validation, comme les autres hyperparamètres.
"""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from elecfr.features import FEATURES
from elecfr.models.common import FitPredict, type_jour


class LightGBMModel(FitPredict):
    nom = "lightgbm"

    def features(self, df):
        X = df[FEATURES].astype(float).copy()
        X["type_jour"] = type_jour(df).astype(float)
        return X

    def build(self):
        p = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31}
        p.update(self.params)
        return lgb.LGBMRegressor(
            objective="regression_l1",   # minimise l'erreur absolue, donc la MAE
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1, **p)


def native_params(params: dict) -> tuple[dict, int]:
    """Mêmes réglages que `LightGBMModel.build`, pour l'interface native
    `lgb.train` (sans scikit-learn) : noms natifs + nombre d'arbres."""
    p = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31}
    p.update(params)
    native = {"objective": "regression_l1", "learning_rate": p["learning_rate"],
              "num_leaves": p["num_leaves"], "min_data_in_leaf": 50, "bagging_fraction": 0.8,
              "bagging_freq": 1, "feature_fraction": 0.8, "seed": 0, "num_threads": 0, "verbose": -1}
    return native, p["n_estimators"]


# Première exécution réelle (12/11/2026) : le meilleur réglage avait 1 000
# arbres, la plus grande valeur testée (au bord de la grille). On ajoute 2 000 ;
# le choix reste fait sur la validation seulement.
GRID = [
    {"n_estimators": n, "learning_rate": 0.05, "num_leaves": leaves}
    for n in (400, 1000, 2000) for leaves in (31, 63)
]


def importances(model, variables: list[str]) -> pd.Series:
    """Importance par gain : réduction totale de l'erreur apportée par les
    coupures sur chaque variable, en % du total."""
    g = pd.Series(model.booster_.feature_importance("gain"), index=variables)
    return (100 * g / g.sum()).sort_values(ascending=False)
