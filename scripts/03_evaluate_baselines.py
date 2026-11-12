"""Étape 3 : évalue les deux références en walk-forward, sur la validation et
sur le test, et enregistre chaque exécution dans MLflow.

Les chiffres imprimés ici viennent de tes données : copie-les dans le README tels quels.
"""
import pandas as pd

from elecfr import config, tracking
from elecfr.baselines import BASELINES
from elecfr.evaluation import walk_forward

df = pd.read_parquet(config.PROCESSED_DIR / "dataset_horaire.parquet")
tracking.setup()

rows = []
for periode, (a, b) in {"validation": (config.VAL_START, config.VAL_END),
                        "test": (config.TEST_START, config.TEST_END)}.items():
    for name, fp in BASELINES.items():
        pred = walk_forward(df, fp, a, b)
        rows.append(tracking.log_evaluation(
            df, pred, modele=name, periode=periode, famille="reference",
            params={"debut": a, "fin": b, "refit": "MS"}))

table = pd.DataFrame(rows).drop(columns="run_id")
config.REPORTS_DIR.mkdir(exist_ok=True)
table.to_csv(config.REPORTS_DIR / "references.csv", index=False)
print(table.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
