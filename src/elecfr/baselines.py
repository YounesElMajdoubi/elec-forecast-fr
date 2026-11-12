"""Les références, écrites avec la même interface que les modèles.

1. Naïf saisonnier : la consommation de la même heure locale, 7 jours plus tôt.
   Aucun apprentissage. C'est le plancher : un modèle qui ne le bat pas
   n'apporte rien.
2. Prévision J-1 de RTE, telle que publiée.
3. Prévision J-1 de RTE RECALÉE : on lui retire, heure par heure, son biais
   moyen des 28 derniers jours connus à l'émission (jusqu'à J-2).

Pourquoi la troisième (DECISIONS.md, D20) : la prévision publiée de RTE est
en moyenne ~950 MW SOUS notre vérité (données consolidées). Une partie de
l'avance apparente de nos modèles tient donc à ce décalage de niveau, pas à
une meilleure prévision. La version recalée corrige ce décalage avec la seule
information disponible la veille : c'est la référence honnête pour juger nos
modèles.
"""
import numpy as np
import pandas as pd

JOURS_RECALAGE = 28


def naive_seasonal(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    return target["conso_j7"].to_numpy()


def rte_day_ahead(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    return target["rte_j1"].to_numpy()


def rte_day_ahead_recalee(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """RTE J-1 moins son biais moyen par heure sur les 28 derniers jours de
    `train`. `train` s'arrête à J-2 du premier jour prévu (walk-forward) :
    aucune information du futur. Le biais n'est mis à jour qu'une fois par
    bloc (mois), ce qui est plus prudent qu'un recalage quotidien."""
    recent = train[train["date_local"] > train["date_local"].max() - pd.Timedelta(days=JOURS_RECALAGE)]
    biais = (recent["rte_j1"] - recent["consommation"]).groupby(recent["heure"]).mean()
    return (target["rte_j1"] - target["heure"].map(biais).fillna(0.0)).to_numpy()


BASELINES = {
    "naif_saisonnier": naive_seasonal,
    "rte_j1": rte_day_ahead,
    "rte_j1_recalee": rte_day_ahead_recalee,
}
