"""Briques communes aux modèles."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "consommation"


def type_jour(df: pd.DataFrame) -> pd.Series:
    """Jour de la semaine (0 = lundi … 6 = dimanche), un jour férié étant
    traité comme un dimanche : sa courbe de charge lui ressemble bien plus
    qu'à celle du jour de semaine qu'il remplace."""
    t = df["jour_semaine"].copy()
    t[df["ferie"] == 1] = 6
    return t.astype(int)


class FitPredict:
    """Enveloppe commune : entraîne sur `train`, prévoit `target`.

    Les lignes d'entraînement sans valeur cible sont retirées. Le modèle du
    dernier appel est gardé dans `self.dernier` (pour MLflow) avec la liste
    des variables utilisées.
    """

    nom = "modele"

    def __init__(self, **params):
        self.params = params
        self.dernier = None
        self.variables: list[str] = []

    def features(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - abstrait
        raise NotImplementedError

    def build(self):  # pragma: no cover - abstrait
        raise NotImplementedError

    def __call__(self, train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
        # On n'apprend que sur des heures complètes : consommation connue ET
        # température prévue disponible. Remplacer une température manquante
        # de janvier par la médiane (~13 °C) apprendrait au modèle qu'il fait
        # doux en plein hiver (DECISIONS.md, D18).
        train = train[train[TARGET].notna() & train["temp_prev"].notna()]
        X_train = self.features(train)
        self.variables = list(X_train.columns)
        model = self.build()
        model.fit(X_train, train[TARGET].to_numpy())
        self.dernier = model
        return model.predict(self.features(target)[self.variables])
