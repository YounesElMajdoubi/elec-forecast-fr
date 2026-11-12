"""Enregistrement des exécutions dans MLflow.

Une exécution (run) = un modèle, un réglage, une période. On y range :
- paramètres : réglage du modèle et protocole (dates, fréquence de réentraînement) ;
- métriques  : MAE et MAPE globales, nombre d'heures évaluées et exclues ;
- tags       : modele, periode, famille, selection (candidat / retenu) ;
- artefacts  : métriques par heure et par saison, prévisions heure par heure,
  et pour les modèles appris le dernier modèle entraîné.
Le tableau comparatif final (scripts/05_compare.py) est relu depuis MLflow.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from elecfr import config
from elecfr.evaluation import evaluate


def setup():
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)


def log_evaluation(df: pd.DataFrame, pred: pd.Series, *, modele: str, periode: str,
                   famille: str, params: dict, selection: str = "retenu",
                   model=None, extra: dict[str, pd.Series] | None = None) -> dict:
    """Évalue `pred`, enregistre tout dans un run MLflow, renvoie la ligne globale."""
    res = evaluate(df, pred)
    g = res["global"].iloc[0]
    with mlflow.start_run(run_name=f"{modele}_{periode}") as run, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mlflow.set_tags({"modele": modele, "periode": periode, "famille": famille, "selection": selection})
        mlflow.log_params(params)
        mlflow.log_metrics({"MAE_MW": g["MAE_MW"], "MAPE_pct": g["MAPE_pct"],
                            "n_heures": g["n_heures"], "n_exclues": g["n_exclues"]})
        for k in ("par_heure", "par_saison"):
            res[k].to_csv(tmp / f"{k}.csv")
            mlflow.log_artifact(str(tmp / f"{k}.csv"), artifact_path="metriques")
        preds = df.loc[pred.index, ["consommation"]].assign(prevision=pred)
        preds.to_parquet(tmp / "previsions.parquet")
        mlflow.log_artifact(str(tmp / "previsions.parquet"))
        for name, ser in (extra or {}).items():
            ser.to_csv(tmp / f"{name}.csv", header=["valeur"])
            mlflow.log_artifact(str(tmp / f"{name}.csv"), artifact_path="analyse")
        if model is not None:
            _log_model(model)
        out = {"run_id": run.info.run_id, "modele": modele, "periode": periode, **g.to_dict()}
    if selection == "retenu":
        save_local(df, pred, modele, periode)
    return out


def save_local(df: pd.DataFrame, pred: pd.Series, modele: str, periode: str):
    """Copie locale des prévisions RETENUES (diagnostics, graphiques du README).
    Les essais candidats ne sont gardés que dans MLflow, pour ne pas écraser
    le réglage retenu par le dernier essai de la grille."""
    pdir = config.REPORTS_DIR / "previsions"
    pdir.mkdir(parents=True, exist_ok=True)
    df.loc[pred.index, ["consommation"]].assign(prevision=pred).to_parquet(pdir / f"{modele}_{periode}.parquet")


def _log_model(model):
    """Enregistre le dernier modèle entraîné.

    - LightGBM : format natif de LightGBM (fichier texte des arbres).
    - scikit-learn : MLflow 3 sérialise avec skops, un format qui refuse de
      recharger des types qu'on ne lui a pas déclarés sûrs (protection contre
      l'exécution de code à l'ouverture, risque connu du format pickle). Notre
      pipeline contient un `numpy.dtype`, qu'on déclare explicitement.
    La signature de log_model a changé avec MLflow 3 (name au lieu de
    artifact_path) : on essaie la nouvelle, puis l'ancienne.
    """
    import lightgbm as lgb

    if hasattr(model, "state_dict"):
        # Réseau PyTorch : on enregistre ses poids (state_dict), format simple
        # et sûr ; l'architecture est décrite dans elecfr/models/lstm.py.
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lstm_poids.pt"
            torch.save(model.state_dict(), path)
            mlflow.log_artifact(str(path), artifact_path="modele")
        return
    if isinstance(model, lgb.LGBMModel):
        log, extra = mlflow.lightgbm.log_model, {}
    else:
        log, extra = mlflow.sklearn.log_model, {"skops_trusted_types": ["numpy.dtype"]}
    for kwargs in ({"name": "modele", **extra}, {"name": "modele"}, {"artifact_path": "modele"}):
        try:
            log(model, **kwargs)
            return
        except TypeError:
            continue
    raise RuntimeError("Impossible d'enregistrer le modèle dans MLflow (version incompatible ?)")


def mark_selected(run_id: str):
    """Marque le run de validation dont le réglage a été retenu."""
    mlflow.MlflowClient().set_tag(run_id, "selection", "retenu")
