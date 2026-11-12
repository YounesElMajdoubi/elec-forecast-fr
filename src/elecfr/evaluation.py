"""Découpage chronologique, walk-forward et métriques.

Interface commune à TOUS les modèles, références comprises :
    fit_predict(train: DataFrame, target: DataFrame) -> np.ndarray
`train` ne contient que des lignes dont la consommation était connue avant
l'émission de la première prévision du bloc ; `target` contient les lignes à
prévoir (leurs variables sont déjà construites sans fuite).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Lignes dont le jour local est dans [start, end]."""
    d = df["date_local"]
    return df[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def mape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat) / np.abs(y)) * 100)


def walk_forward(df: pd.DataFrame, fit_predict: FitPredict, start: str, end: str,
                 refit: str = "MS", train_days: int | None = None,
                 train_start: str | None = None, verbose: bool = False) -> pd.Series:
    """Prévoit [start, end] par blocs successifs (mensuels par défaut).

    Pour un bloc commençant le jour B, la toute première prévision est émise
    la veille B-1 à 10 h : seuls les jours <= B-2 ont une consommation
    entièrement connue. Le modèle est donc réentraîné sur date_local <= B-2,
    puis il prévoit tous les jours du bloc sans être mis à jour (c'est plus
    prudent que la réalité, jamais plus optimiste).

    train_days=None : fenêtre qui s'agrandit (tout le passé disponible).
    train_days=1095 : fenêtre glissante de longueur fixe (les 3 dernières années).
    train_start : première journée utilisable pour l'entraînement (les jours
    téléchargés avant ne servent qu'à calculer les retards).
    """
    starts = pd.date_range(start, end, freq=refit)
    if len(starts) == 0 or starts[0] != pd.Timestamp(start):
        starts = starts.insert(0, pd.Timestamp(start))
    bounds = list(starts) + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    preds = []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        last = b0 - pd.Timedelta(days=2)
        train = df[df["date_local"] <= last]
        if train_days is not None:
            train = train[train["date_local"] > last - pd.Timedelta(days=train_days)]
        if train_start is not None:
            train = train[train["date_local"] >= pd.Timestamp(train_start)]
        train = train[train["consommation"].notna()]
        target = df[(df["date_local"] >= b0) & (df["date_local"] < b1)]
        if target.empty:
            continue
        t0 = time.time()
        yhat = np.asarray(fit_predict(train, target), dtype=float)
        preds.append(pd.Series(yhat, index=target.index))
        if verbose:
            print(f"    {b0:%Y-%m} : entraîné sur {len(train):,} h, {time.time() - t0:.1f} s", flush=True)
    return pd.concat(preds).rename("prevision")


def evaluate(df: pd.DataFrame, pred: pd.Series) -> dict[str, pd.DataFrame]:
    """MAE (MW) et MAPE (%) : globales, par heure locale, par saison.

    On n'évalue que les heures où la vérité ET la prévision existent, et on
    rapporte combien d'heures ont été exclues (pour ne rien cacher).
    """
    d = df.loc[pred.index, ["consommation", "heure", "saison"]].assign(prevision=pred)
    n_total = len(d)
    d = d.dropna(subset=["consommation", "prevision"])

    def agg(g):
        return pd.Series({"MAE_MW": mae(g["consommation"], g["prevision"]),
                          "MAPE_pct": mape(g["consommation"], g["prevision"]),
                          "n_heures": len(g)})

    glob = agg(d).to_frame().T
    glob["n_exclues"] = n_total - len(d)
    return {
        "global": glob,
        "par_heure": d.groupby("heure")[["consommation", "prevision"]].apply(agg),
        "par_saison": d.groupby("saison")[["consommation", "prevision"]].apply(agg),
    }
