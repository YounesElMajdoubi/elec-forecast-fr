"""Étape 9 : entraîne le modèle servi par l'API et l'enregistre dans models/.

Réglage : celui retenu pour LightGBM sur la validation, relu dans MLflow
(aucun nouveau réglage ici). Données : toutes les heures complètes depuis le
début de l'entraînement, test compris : pour servir, on utilise tout le passé.
Les performances à annoncer restent celles du walk-forward (README).
"""
import pandas as pd

from elecfr import config, serving  # LightGBM importé avant MLflow (voir DECISIONS.md, D28)
import mlflow  # noqa: E402
from elecfr import tracking  # noqa: E402

tracking.setup()
runs = mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT],
                          filter_string="tags.modele = 'lightgbm' and tags.periode = 'test' and tags.selection = 'retenu'",
                          order_by=["attributes.start_time DESC"], max_results=1)
if runs.empty:
    raise SystemExit("Aucun run LightGBM retenu dans MLflow : lance d'abord 04_train_models.py.")
r = runs.iloc[0]
params = {"n_estimators": int(r["params.n_estimators"]), "learning_rate": float(r["params.learning_rate"]),
          "num_leaves": int(r["params.num_leaves"])}

df = pd.read_parquet(config.PROCESSED_DIR / "dataset_horaire.parquet")
booster, meta = serving.train_final(df, params)
meta["mae_test_walk_forward_mw"] = round(float(r["metrics.MAE_MW"]), 1)
meta["run_mlflow_reglage"] = r["run_id"]
serving.save(booster, meta, config.ROOT / "models")

with mlflow.start_run(run_name="lightgbm_final"):
    mlflow.set_tags({"modele": "lightgbm", "periode": "final", "famille": "ml", "selection": "servi"})
    mlflow.log_params({**params, "entraine_du": meta["entraine_du"], "entraine_au": meta["entraine_au"]})
    mlflow.log_artifacts(str(config.ROOT / "models"), artifact_path="modele")

print(f"Modèle final enregistré dans models/ : {params}, entraîné du {meta['entraine_du']} au {meta['entraine_au']}")
print("Lancer l'API : uvicorn elecfr.api:app --reload   puis http://127.0.0.1:8000/docs")
