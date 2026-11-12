"""API de prévision (FastAPI).

Lancer en local :  uvicorn elecfr.api:app --reload    puis http://127.0.0.1:8000/docs
Variables d'environnement (facultatives) :
  ELECFR_DATA   chemin du jeu horaire     (défaut : data/processed/dataset_horaire.parquet)
  ELECFR_MODELS dossier du modèle final   (défaut : models/)

ELECFR_PRIX   jeu de prix horaire (défaut : data/processed/prix_horaires.parquet)

Mode actuel : « rejeu ». L'API prévoit tout jour présent dans le jeu de
données, avec les variables connues la veille. Prévoir le lendemain en
direct demanderait de construire la ligne de demain (prévision météo du
jour, consommation jusqu'à J-2 depuis le Lakehouse) : étape suivante, voir README.

Les réponses sont décrites par des modèles Pydantic : la documentation
(/docs) montre chaque champ, son type et sa signification, et FastAPI
vérifie que les réponses respectent ce contrat.
"""
from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from elecfr import config, serving, simulation

STATE: dict = {}


class Sante(BaseModel):
    statut: str = Field(examples=["ok"])
    modele: str = Field(examples=["lightgbm"])
    entraine_au: str = Field(description="Dernier jour de l'entraînement du modèle servi", examples=["2026-10-31"])
    donnees_du: str = Field(examples=["2021-10-01"])
    donnees_au: str = Field(examples=["2026-10-31"])


class Heure(BaseModel):
    heure_locale: str = Field(description="Début de l'heure, heure de Paris (ISO 8601)",
                              examples=["2026-01-12T18:00:00+01:00"])
    prevision_mw: float | None = Field(description="Consommation prévue (MW)", examples=[71850.2])
    reel_mw: float | None = Field(description="Consommation réelle, si connue (MW)", examples=[70214.0])
    rte_j1_mw: float | None = Field(description="Prévision J-1 publiée par RTE (MW)", examples=[69120.5])


class Prevision(BaseModel):
    date: str = Field(examples=["2026-01-12"])
    modele: str = Field(examples=["lightgbm"])
    unite: str = Field(examples=["MW"])
    entraine_au: str = Field(examples=["2026-10-31"])
    dans_la_periode_d_entrainement: bool = Field(
        description="Vrai si le jour a servi à l'entraînement : l'erreur affichée est alors optimiste")
    avertissement: str | None = None
    mae_jour_mw: float | None = Field(description="Erreur absolue moyenne du jour, si le réel est connu")
    heures: list[Heure] = Field(description="23, 24 ou 25 heures selon les changements d'heure")


class QuantilesHeure(BaseModel):
    heure_locale: str = Field(examples=["2026-10-15T19:00:00+02:00"])
    q05: float
    q25: float
    q50: float = Field(description="Médiane des trajectoires (€/MWh)")
    q75: float
    q95: float
    reel: float | None = Field(description="Prix réel, s'il est déjà connu (€/MWh)")


class Simulation(BaseModel):
    mois: str = Field(examples=["2026-10"])
    variante: str = Field(description="Modèle retenu sur la validation", examples=["ou_sauts_ancrage"])
    n_trajectoires: int = Field(examples=[1000])
    unite: str = Field(examples=["€/MWh"])
    estime_du: str = Field(examples=["2024-09-30 22:00:00+00:00"])
    estime_au: str = Field(description="Dernière heure de prix utilisée (avant le 1er du mois)",
                           examples=["2026-09-30 21:00:00+00:00"])
    phi: float = Field(description="Coefficient de retour à la moyenne par heure (AR(1))", examples=[0.9471])
    demi_vie_heures: float = Field(description="Temps pour qu'un écart au prix « normal » soit divisé par deux",
                                   examples=[12.76])
    probabilite_saut_par_heure: float = Field(examples=[0.0211])
    prix_moyen_du_mois: dict[str, float] = Field(description="Quantiles 5 %, 50 %, 95 % du prix moyen simulé du mois",
                                                 examples=[{"q05": 123.7, "q50": 139.3, "q95": 155.9}])
    couverture_90_reel: float | None = Field(description="Part des heures réelles dans l'intervalle 5 %–95 %, si connues",
                                             examples=[0.591])
    heures: list[QuantilesHeure]


def _paths():
    data = Path(os.environ.get("ELECFR_DATA", config.PROCESSED_DIR / "dataset_horaire.parquet"))
    models = Path(os.environ.get("ELECFR_MODELS", config.ROOT / "models"))
    prix = Path(os.environ.get("ELECFR_PRIX", config.PROCESSED_DIR / "prix_horaires.parquet"))
    return data, models, prix


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement unique, au démarrage, des jeux de données et des modèles.
    Les prix (volet 2) sont facultatifs : sans eux, /simulate répond 503."""
    data, models, prix = _paths()
    STATE["df"] = pd.read_parquet(data)
    STATE["booster"], STATE["meta"] = serving.load(models)
    if prix.exists():
        STATE["prix"] = pd.read_parquet(prix)
        cfg = models / "simulation_prix.json"
        STATE["sim"] = (json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists()
                        else {"variante": "ou_sauts", "params": simulation.VARIANTES["ou_sauts"],
                              "fenetre_mois": simulation.FENETRE_MOIS})
    yield
    STATE.clear()


app = FastAPI(
    title="Prévision de la consommation électrique française",
    description="Prévision horaire J+1 de la consommation (LightGBM) et simulation Monte-Carlo des prix day-ahead.",
    version="0.1.0",
    lifespan=lifespan,
)

ERREURS = {404: {"description": "Jour absent du jeu de données"},
           422: {"description": "Date invalide ou données du jour incomplètes"}}


@app.get("/health", response_model=Sante)
def health():
    df, meta = STATE["df"], STATE["meta"]
    return {"statut": "ok", "modele": meta["modele"], "entraine_au": meta["entraine_au"],
            "donnees_du": str(df["date_local"].min().date()), "donnees_au": str(df["date_local"].max().date())}


@app.get("/forecast", response_model=Prevision, responses=ERREURS)
def forecast(date: str = Query(..., description="Jour à prévoir, AAAA-MM-JJ (heure de Paris)",
                               examples=["2026-01-12"])):
    try:
        return serving.forecast_day(STATE["df"], STATE["booster"], STATE["meta"], date)
    except serving.ForecastError as e:
        raise HTTPException(status_code=e.status, detail=e.message) from e


@app.get("/simulate", response_model=Simulation,
         responses={422: {"description": "Mois invalide ou prix insuffisants avant ce mois"},
                    503: {"description": "Données de prix non chargées"}})
def simulate(month: str = Query(..., description="Mois à simuler, AAAA-MM", examples=["2026-10"]),
             n: int = Query(1000, ge=100, le=5000, description="Nombre de trajectoires")):
    if "prix" not in STATE:
        raise HTTPException(status_code=503, detail="Données de prix absentes (scripts/11_build_prices.py).")
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise HTTPException(status_code=422, detail=f"Mois invalide : {month!r} (format attendu AAAA-MM)")
    cfg = STATE["sim"]
    try:
        model, sims, cal, reel = simulation.simulate_month(STATE["prix"], month, cfg["params"], n=n,
                                                           fenetre_mois=cfg["fenetre_mois"])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    q = np.quantile(sims, simulation.QUANTILES, axis=0)
    moy = sims.mean(axis=1)
    known = ~np.isnan(reel)
    loc = cal.index.tz_convert(config.TZ)
    heures = [{"heure_locale": t.isoformat(), "q05": round(float(a), 2), "q25": round(float(b), 2),
               "q50": round(float(c), 2), "q75": round(float(d), 2), "q95": round(float(e), 2),
               "reel": None if np.isnan(r) else round(float(r), 2)}
              for t, a, b, c, d, e, r in zip(loc, *q, reel)]
    return {
        "mois": month, "variante": cfg["variante"], "n_trajectoires": n, "unite": "€/MWh",
        "estime_du": model.infos["entraine_du"], "estime_au": model.infos["entraine_au"],
        "phi": round(model.phi, 4), "demi_vie_heures": round(model.demi_vie_h, 2),
        "probabilite_saut_par_heure": round(model.p_saut, 5),
        "prix_moyen_du_mois": {k: round(float(np.quantile(moy, v)), 2) for k, v in (("q05", .05), ("q50", .5), ("q95", .95))},
        "couverture_90_reel": (round(float(((reel >= q[0]) & (reel <= q[4]))[known].mean()), 3) if known.any() else None),
        "heures": heures,
    }
