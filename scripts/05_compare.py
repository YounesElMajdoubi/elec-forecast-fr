"""Étape 5 : tableau comparatif final, relu depuis MLflow (et non recalculé).

Pour chaque modèle et chaque période, on prend le dernier run marqué
« retenu ». On ajoute deux colonnes de lecture :
- gain_vs_naif_pct : de combien la MAE est réduite par rapport au naïf saisonnier ;
- ecart_vs_rte_pct : de combien la MAE dépasse celle de la prévision J-1 de RTE
  (négatif = meilleur que RTE).
Sorties : reports/comparaison.csv et reports/comparaison.md (à coller dans le README).
"""
import mlflow
import pandas as pd

from elecfr import config, tracking

ORDRE = ["naif_saisonnier", "rte_j1", "rte_j1_recalee", "regression_lineaire", "lightgbm", "lstm", "combinaison"]

tracking.setup()
runs = mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT],
                          filter_string="tags.selection = 'retenu'",
                          order_by=["attributes.start_time DESC"])
if runs.empty:
    raise SystemExit("Aucun run « retenu » dans MLflow : lance d'abord les scripts 03 et 04.")

runs = runs.drop_duplicates(["tags.modele", "tags.periode"], keep="first")
t = runs.rename(columns={"tags.modele": "modele", "tags.periode": "periode",
                         "metrics.MAE_MW": "MAE_MW", "metrics.MAPE_pct": "MAPE_pct",
                         "metrics.n_heures": "n_heures"})[["modele", "periode", "MAE_MW", "MAPE_pct", "n_heures"]]

out = []
for periode, g in t.groupby("periode"):
    g = g.set_index("modele")
    naif = g["MAE_MW"].get("naif_saisonnier")
    rte = g["MAE_MW"].get("rte_j1")
    g["gain_vs_naif_pct"] = 100 * (1 - g["MAE_MW"] / naif) if naif else float("nan")
    g["ecart_vs_rte_pct"] = 100 * (g["MAE_MW"] / rte - 1) if rte else float("nan")
    out.append(g.assign(periode=periode))
table = pd.concat(out).reset_index()
table["_p"] = table["periode"].map({"validation": 0, "test": 1})
table["_m"] = table["modele"].map({m: i for i, m in enumerate(ORDRE)}).fillna(99)
table = table.sort_values(["_p", "_m"]).drop(columns=["_p", "_m"])

config.REPORTS_DIR.mkdir(exist_ok=True)
table.to_csv(config.REPORTS_DIR / "comparaison.csv", index=False)

fmt = table.copy()
fmt["MAE (MW)"] = fmt["MAE_MW"].map("{:,.0f}".format).str.replace(",", " ")
fmt["MAPE (%)"] = fmt["MAPE_pct"].map("{:.2f}".format)
fmt["Gain vs naïf"] = fmt["gain_vs_naif_pct"].map("{:+.1f} %".format)
fmt["Écart vs RTE J-1"] = fmt["ecart_vs_rte_pct"].map("{:+.1f} %".format)
md = fmt[["periode", "modele", "MAE (MW)", "MAPE (%)", "Gain vs naïf", "Écart vs RTE J-1"]].to_markdown(index=False)
(config.REPORTS_DIR / "comparaison.md").write_text(md, encoding="utf-8")
print(md)
