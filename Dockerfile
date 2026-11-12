# Image de l'API de prévision (LightGBM). Le jeu de données et le modèle ne
# sont PAS dans l'image : ils sont montés au lancement (ils changent plus
# souvent que le code, et data/ et models/ ne sont pas dans Git).
#
#   docker build -t elecfr-api .
#   docker run --rm -p 8000:8000 -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" elecfr-api
#   → http://127.0.0.1:8000/docs
FROM python:3.12-slim

# LightGBM a besoin de la bibliothèque OpenMP (calcul parallèle)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances de l'API seulement (pas MLflow, deltalake, PyTorch...) : image légère.
# Installées avant le code pour profiter du cache Docker.
RUN pip install --no-cache-dir "pandas>=2.2" "numpy>=1.26" "pyarrow>=15" "lightgbm>=4.5" \
    "holidays>=0.50" "requests>=2.31" "fastapi>=0.110" "uvicorn>=0.29"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

ENV ELECFR_DATA=/app/data/processed/dataset_horaire.parquet \
    ELECFR_MODELS=/app/models

# Utilisateur sans privilèges
RUN useradd --create-home api
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "elecfr.api:app", "--host", "0.0.0.0", "--port", "8000"]
