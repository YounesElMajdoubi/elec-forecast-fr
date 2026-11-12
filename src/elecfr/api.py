"""API de prévision (FastAPI).

Lancer en local :  uvicorn elecfr.api:app --reload    puis http://127.0.0.1:8000/docs
Variables d'environnement (facultatives) :
  ELECFR_DATA   chemin du jeu horaire     (défaut : data/processed/dataset_horaire.parquet)
  ELECFR_MODELS dossier du modèle final   (défaut : models/)

Mode actuel : « rejeu ». L'API prévoit tout jour présent dans le jeu de
données, avec les variables connues la veille. Prévoir le lendemain en
direct demanderait de construire la ligne de demain (prévision météo du
jour, consommation jusqu'à J-2 depuis le Lakehouse) : étape suivante, voir README.

Les réponses sont décrites par des modèles Pydantic : la documentation
(/docs) montre chaque champ, son type et sa signification, et FastAPI
vérifie que les réponses respectent ce contrat.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from elecfr import config, serving

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


def _paths():
    data = Path(os.environ.get("ELECFR_DATA", config.PROCESSED_DIR / "dataset_horaire.parquet"))
    models = Path(os.environ.get("ELECFR_MODELS", config.ROOT / "models"))
    return data, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement unique, au démarrage, du jeu de données et du modèle."""
    data, models = _paths()
    STATE["df"] = pd.read_parquet(data)
    STATE["booster"], STATE["meta"] = serving.load(models)
    yield
    STATE.clear()


app = FastAPI(
    title="Prévision de la consommation électrique française",
    description="Prévision horaire J+1 (LightGBM) ; simulation des prix à venir (volet 2).",
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


@app.get("/simulate", responses={501: {"description": "Volet 2 pas encore disponible"}})
def simulate(month: str = Query(..., description="Mois à simuler, AAAA-MM", examples=["2026-11"])):
    raise HTTPException(status_code=501, detail="Simulation des prix (volet 2) pas encore disponible : "
                                                "en attente des données de prix ENTSO-E.")
