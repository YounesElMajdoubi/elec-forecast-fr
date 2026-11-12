"""Tests de l'API sur un petit jeu FACTICE et un modèle LightGBM minuscule."""
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from elecfr import serving  # noqa: E402
from test_models import fake  # noqa: E402
from test_simulation import fake_prices  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    df = fake(start="2024-01-01", end="2024-04-15")
    df["rte_j1"] = df["consommation"] + 100
    df.loc[df["date_local"] >= pd.Timestamp("2024-04-10"), "consommation"] = float("nan")  # « futur » sans réel
    df.to_parquet(tmp / "dataset.parquet")
    booster, meta = serving.train_final(df, {"n_estimators": 20, "num_leaves": 7})
    serving.save(booster, meta, tmp / "models")

    import os
    os.environ["ELECFR_DATA"] = str(tmp / "dataset.parquet")
    os.environ["ELECFR_MODELS"] = str(tmp / "models")
    fake_prices(start="2023-01-01", end="2025-03-10").to_parquet(tmp / "prix.parquet")
    os.environ["ELECFR_PRIX"] = str(tmp / "prix.parquet")
    from elecfr.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["statut"] == "ok"


def test_forecast_returns_one_value_per_hour(client):
    r = client.get("/forecast", params={"date": "2024-03-05"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["heures"]) == 24 and body["unite"] == "MW"
    assert all(h["prevision_mw"] is not None for h in body["heures"])
    assert body["dans_la_periode_d_entrainement"] is True and body["avertissement"]


def test_forecast_dst_day_has_23_hours(client):
    assert len(client.get("/forecast", params={"date": "2024-03-31"}).json()["heures"]) == 23


def test_forecast_day_without_truth(client):
    body = client.get("/forecast", params={"date": "2024-04-12"}).json()
    assert body["mae_jour_mw"] is None and body["heures"][0]["reel_mw"] is None
    assert body["dans_la_periode_d_entrainement"] is False


def test_errors(client):
    assert client.get("/forecast", params={"date": "2030-01-01"}).status_code == 404
    assert client.get("/forecast", params={"date": "pas-une-date"}).status_code == 422
    assert client.get("/forecast").status_code == 422
    assert client.get("/simulate", params={"month": "2026-11"}).status_code == 422   # pas de prix avant
    assert client.get("/simulate", params={"month": "mars"}).status_code == 422


def test_simulate_returns_quantiles_per_hour(client):
    r = client.get("/simulate", params={"month": "2025-03", "n": 200})
    assert r.status_code == 200
    b = r.json()
    assert len(b["heures"]) == 31 * 24 - 1 and b["n_trajectoires"] == 200
    h = b["heures"][100]
    assert h["q05"] <= h["q25"] <= h["q50"] <= h["q75"] <= h["q95"]
    assert h["reel"] is not None and b["couverture_90_reel"] is not None   # 1er au 9 mars connus
    assert b["heures"][-1]["reel"] is None                                 # fin de mois inconnue


def test_openapi_documents_response_fields(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert {"Prevision", "Heure", "Sante", "Simulation", "QuantilesHeure"} <= set(schemas)
    assert "prevision_mw" in schemas["Heure"]["properties"]


def test_dockerfile_sets_every_path_variable_read_by_the_api():
    """Dans l'image, le code est installé dans site-packages : chaque variable
    ELECFR_* lue par l'API doit être fixée dans le Dockerfile, sinon le fichier
    monté n'est pas trouvé (bug vu le 12/11/2026 : /simulate répondait 503)."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    lues = set(re.findall(r'environ\.get\("(ELECFR_[A-Z_]+)"', (root / "src/elecfr/api.py").read_text(encoding="utf-8")))
    docker = (root / "Dockerfile").read_text(encoding="utf-8")
    assert lues and all(v + "=" in docker for v in lues), sorted(v for v in lues if v + "=" not in docker)
