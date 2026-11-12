"""Régression linéaire (Ridge) avec variables construites à la main.

Un modèle linéaire ne sait faire qu'une chose : additionner des effets. Toute
non-linéarité doit donc être fabriquée sous forme de colonnes :

- Profil journalier par type de jour : une colonne indicatrice pour chaque
  couple (heure, type de jour), soit 24 x 7 = 168 colonnes. Le modèle apprend
  ainsi une courbe de charge propre au lundi, au samedi, au dimanche/férié...
  au lieu d'un seul profil moyen.
- Chauffage et climatisation par heure : degrés-jours x indicatrice d'heure.
  Un degré de froid en plus ne pèse pas pareil à 4 h et à 19 h.
- Saison x heure : sin/cos du jour de l'année croisés avec l'heure, pour
  l'éclairage (pic du soir plus tôt et plus fort en hiver).
- Retards de consommation J-2, J-7, J-14 et indicateurs calendaires.

Ridge = régression linéaire avec une pénalité sur la taille des coefficients
(alpha). Avec ~270 colonnes très corrélées, la pénalité stabilise les
coefficients ; alpha est choisi sur la période de validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from elecfr.models.common import FitPredict, type_jour

HOURS = range(24)
NUMERIC = ["conso_j2", "conso_j7", "conso_j14", "zones_vacances", "pont",
           "changement_heure", "annee_sin", "annee_cos", "dj_chauffage", "dj_clim"]


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matrice de variables, toujours avec les mêmes colonnes dans le même
    ordre, quelles que soient les lignes reçues (indispensable pour prévoir
    une période qui ne contient pas toutes les combinaisons)."""
    h = df["heure"].to_numpy()
    tj = type_jour(df).to_numpy()
    cols = {c: df[c].to_numpy(dtype=float) for c in NUMERIC}
    for hour in HOURS:
        is_h = (h == hour).astype(float)
        for t in range(7):
            cols[f"profil_h{hour:02d}_j{t}"] = is_h * (tj == t)
        cols[f"chauffage_h{hour:02d}"] = is_h * df["dj_chauffage"].to_numpy()
        cols[f"clim_h{hour:02d}"] = is_h * df["dj_clim"].to_numpy()
        cols[f"saison_sin_h{hour:02d}"] = is_h * df["annee_sin"].to_numpy()
        cols[f"saison_cos_h{hour:02d}"] = is_h * df["annee_cos"].to_numpy()
    return pd.DataFrame(cols, index=df.index)


class LinearModel(FitPredict):
    nom = "regression_lineaire"

    def features(self, df):
        return design_matrix(df)

    def build(self):
        # Imputation par la médiane : quelques heures n'ont pas de température
        # ou de retard (changements d'heure, trous de l'archive météo).
        # Standardisation : la pénalité Ridge doit traiter toutes les colonnes
        # à la même échelle.
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             Ridge(alpha=self.params.get("alpha", 1.0)))


# Grille sur plusieurs ordres de grandeur. Première exécution réelle (12/11/2026) :
# le meilleur alpha était 1, la plus petite valeur testée (au bord de la grille),
# l'erreur de validation montant avec alpha. On a donc ajouté 0,01 et 0,1 :
# le choix reste fait sur la validation seulement.
GRID = [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)]


def coefficients(model, variables: list[str]) -> pd.Series:
    """Coefficients en MW par écart-type de chaque variable (pour l'analyse)."""
    return pd.Series(model[-1].coef_, index=variables).sort_values(key=np.abs, ascending=False)
