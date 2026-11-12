"""Modèle final et prévision d'un jour, partagés par le script d'entraînement
final et par l'API.

Choix du modèle servi : LightGBM (DECISIONS.md, D25). Le LSTM est 8 % plus
précis sur le test, mais il imposerait PyTorch dans l'image Docker (plusieurs
centaines de Mo) ; LightGBM tient dans une image légère et prédit en
quelques millisecondes.

Point clé : les variables sont calculées par LightGBMModel.features, la MÊME
fonction qu'à l'entraînement. Pas de seconde implémentation qui pourrait
diverger (« training/serving skew »).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from elecfr import config
from elecfr.models.common import TARGET
from elecfr.models.gbm import LightGBMModel, native_params

MODEL_FILE, META_FILE = "lightgbm_final.txt", "lightgbm_final.json"


def train_final(df: pd.DataFrame, params: dict) -> tuple[lgb.Booster, dict]:
    """Entraîne LightGBM sur toutes les heures complètes depuis TRAIN_START.

    Interface native `lgb.train` (pas scikit-learn) : mêmes réglages et même
    filtrage des lignes que le walk-forward (`native_params`, `FitPredict`),
    sans dépendre de scikit-learn ni au service ni à l'entraînement final."""
    train = df[df["date_local"] >= pd.Timestamp(config.TRAIN_START)]
    train = train[train[TARGET].notna() & train["temp_prev"].notna()]
    X = LightGBMModel().features(train)
    native, n_trees = native_params(params)
    booster = lgb.train(native, lgb.Dataset(X, label=train[TARGET].to_numpy()), num_boost_round=n_trees)
    meta = {
        "modele": "lightgbm",
        "params": params,
        "variables": list(X.columns),
        "entraine_du": str(train["date_local"].min().date()),
        "entraine_au": str(train.loc[train["consommation"].notna(), "date_local"].max().date()),
        "cree_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return booster, meta


def save(booster: lgb.Booster, meta: dict, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(directory / MODEL_FILE))
    (directory / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load(directory: Path) -> tuple[lgb.Booster, dict]:
    booster = lgb.Booster(model_file=str(directory / MODEL_FILE))
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    return booster, meta


class ForecastError(Exception):
    """Erreur prévue (date absente, données incomplètes), avec son code HTTP."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def forecast_day(df: pd.DataFrame, booster: lgb.Booster, meta: dict, day: str) -> dict:
    """Prévision des heures du jour local `day` (AAAA-MM-JJ) à partir des
    variables déjà construites dans le jeu de données (sans fuite, D1)."""
    try:
        d = pd.Timestamp(day)
    except ValueError as exc:
        raise ForecastError(422, f"Date invalide : {day!r} (format attendu AAAA-MM-JJ)") from exc
    rows = df[df["date_local"] == d]
    if rows.empty:
        lo, hi = df["date_local"].min().date(), df["date_local"].max().date()
        raise ForecastError(404, f"Pas de données pour le {d.date()} : jeu disponible du {lo} au {hi}.")
    if rows["temp_prev"].isna().any():
        raise ForecastError(422, f"Température prévue manquante pour le {d.date()} : prévision impossible.")

    X = LightGBMModel().features(rows)[meta["variables"]]
    pred = booster.predict(X)
    loc = rows.index.tz_convert(config.TZ)

    def num(v):
        return None if pd.isna(v) else round(float(v), 1)

    heures = [{"heure_locale": t.isoformat(), "prevision_mw": num(p), "reel_mw": num(r), "rte_j1_mw": num(q)}
              for t, p, r, q in zip(loc, pred, rows["consommation"], rows["rte_j1"])]
    reel = rows["consommation"].to_numpy(dtype=float)
    ok = ~np.isnan(reel)
    in_train = d <= pd.Timestamp(meta["entraine_au"])
    return {
        "date": str(d.date()),
        "modele": meta["modele"],
        "unite": "MW",
        "entraine_au": meta["entraine_au"],
        "dans_la_periode_d_entrainement": bool(in_train),
        "avertissement": ("Jour vu à l'entraînement : l'erreur affichée est optimiste. "
                          "Les performances honnêtes sont celles du walk-forward (README).") if in_train else None,
        "mae_jour_mw": num(np.abs(pred[ok] - reel[ok]).mean()) if ok.any() else None,
        "heures": heures,
    }
