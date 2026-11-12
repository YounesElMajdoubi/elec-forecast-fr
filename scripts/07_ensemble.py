"""Étape 7 : combinaison des modèles appris.

Idée : les erreurs du LSTM (fêtes, août) et de LightGBM (vagues de froid) ne
tombent pas aux mêmes moments. Une moyenne pondérée peut donc faire mieux que
chacun pris seul.

Protocole, comme pour les hyperparamètres :
1. Les poids (w_lineaire, w_lightgbm, w_lstm), positifs et de somme 1, sont
   cherchés sur une grille au pas de 0,1 (66 combinaisons) en minimisant la
   MAE sur la VALIDATION. Les prévisions de validation utilisées sont
   elles-mêmes hors échantillon (walk-forward) : pas de fuite.
2. Les poids retenus sont appliqués une seule fois au TEST.
Tout est enregistré dans MLflow sous le modèle « combinaison ».

Lit les prévisions retenues dans reports/previsions/ (écrites par 04).
"""
import itertools

import numpy as np
import pandas as pd

from elecfr import config, tracking
from elecfr.evaluation import mae

MEMBRES = ["regression_lineaire", "lightgbm", "lstm"]
PAS = 0.1

df = pd.read_parquet(config.PROCESSED_DIR / "dataset_horaire.parquet")
pdir = config.REPORTS_DIR / "previsions"


def load(periode):
    frames = {}
    for m in MEMBRES:
        f = pdir / f"{m}_{periode}.parquet"
        if not f.exists():
            raise SystemExit(f"Prévisions manquantes : {f}. Lance d'abord 04_train_models.py.")
        frames[m] = pd.read_parquet(f)["prevision"]
    return pd.DataFrame(frames)


val, test = load("validation"), load("test")
y_val = df.loc[val.index, "consommation"]
ok = y_val.notna() & val.notna().all(axis=1)

n = round(1 / PAS)
grille = [np.array(c) / n for c in itertools.product(range(n + 1), repeat=len(MEMBRES)) if sum(c) == n]
scores = []
for w in grille:
    scores.append((mae(y_val[ok], val[ok].to_numpy() @ w), w))
scores.sort(key=lambda s: s[0])

print("== Les 5 meilleures combinaisons sur la validation")
for s, w in scores[:5]:
    print("  " + ", ".join(f"{m} {x:.1f}" for m, x in zip(MEMBRES, w)) + f" → MAE {s:,.0f} MW")
seuls = {m: mae(y_val[ok], val.loc[ok, m]) for m in MEMBRES}
print("  (seuls : " + ", ".join(f"{m} {v:,.0f}" for m, v in seuls.items()) + ")")

mae_val, w = scores[0]
params = {f"poids_{m}": round(float(x), 2) for m, x in zip(MEMBRES, w)} | {"pas_grille": PAS}

tracking.setup()
r_val = tracking.log_evaluation(df, pd.Series(val.to_numpy() @ w, index=val.index), modele="combinaison",
                                periode="validation", famille="ml", params=params)
r_test = tracking.log_evaluation(df, pd.Series(test.to_numpy() @ w, index=test.index), modele="combinaison",
                                 periode="test", famille="ml", params=params)
print(f"\nRetenu : {params}")
print(f"Validation : MAE {r_val['MAE_MW']:,.0f} MW ; test : MAE {r_test['MAE_MW']:,.0f} MW, "
      f"MAPE {r_test['MAPE_pct']:.2f} %")
print("Tableau complet : python scripts/05_compare.py")
