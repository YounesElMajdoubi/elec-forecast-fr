"""Paramètres centraux du projet.

Tout ce qui est un *choix* (fenêtres de dates, villes, heure d'émission de la
prévision) est ici, pour qu'on puisse le retrouver et le justifier dans
DECISIONS.md. Les dates sont des jours LOCAUX (Europe/Paris), bornes incluses.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"

TZ = "Europe/Paris"

# --- Fenêtres temporelles -------------------------------------------------
# On télécharge un mois avant le début de l'entraînement pour pouvoir calculer
# les retards (J-7, J-14) dès le premier jour d'entraînement.
DOWNLOAD_START = "2021-10-01"
DOWNLOAD_END = "2026-10-31"

# Validation et test couvrent chacun une année complète (novembre → octobre),
# pour que chaque saison, hiver compris, soit vue au réglage ET au test.
TRAIN_START, TRAIN_END = "2021-11-01", "2024-10-31"   # 3 ans
VAL_START, VAL_END = "2024-11-01", "2025-10-31"       # 12 mois
TEST_START, TEST_END = "2025-11-01", "2026-10-31"     # 12 mois

# --- Suivi des expériences ------------------------------------------------
# Base MLflow explicite (SQLite à la racine du dépôt) : tous les scripts et
# `mlflow ui` lisent la même, quel que soit le dossier d'où on les lance.
MLFLOW_URI = "sqlite:///" + (ROOT / "mlflow.db").as_posix()
MLFLOW_EXPERIMENT = "prevision_conso_j1"

# --- Source de la consommation -------------------------------------------
# "fabric" : tables du Lakehouse du projet A (paramètres dans sources.toml)
# "api"    : téléchargement direct depuis ODRÉ
ECO2MIX_SOURCE = "fabric"
SOURCES_FILE = ROOT / "sources.toml"

# --- Moment où la prévision est faite ------------------------------------
# On se place la veille (J-1) à 10h, heure de Paris, avant la fermeture du
# marché day-ahead à midi. Toute information postérieure est interdite.
FORECAST_ISSUE_HOUR_LOCAL = 10

# Délai (heures) entre l'heure nominale d'un run météo et sa disponibilité.
# Valeur prudente : voir DECISIONS.md (D4).
WEATHER_AVAILABILITY_LAG_H = 6

# --- Météo -----------------------------------------------------------------
# Modèle de l'API « Previous Runs » d'Open-Meteo. GFS est archivé depuis
# mars 2021, ce qui couvre toute notre fenêtre ; la plupart des autres modèles
# seulement depuis janvier 2024.
WEATHER_MODEL = "gfs_seamless"

# Villes et poids ≈ population des aires d'attraction (ordre de grandeur,
# millions d'habitants). Ce n'est pas la « température France » de RTE ;
# c'est une approximation documentée.
CITIES = {
    "paris": (48.8566, 2.3522, 13.1),
    "lyon": (45.7640, 4.8357, 2.3),
    "marseille": (43.2965, 5.3698, 1.9),
    "lille": (50.6292, 3.0573, 1.5),
    "toulouse": (43.6047, 1.4442, 1.45),
    "bordeaux": (44.8378, -0.5792, 1.4),
    "nantes": (47.2184, -1.5536, 1.0),
    "strasbourg": (48.5734, 7.7521, 0.85),
}

# Seuils des degrés-jours (°C) : chauffage sous 15 °C, climatisation au-dessus
# de 22 °C. Choix usuel, à vérifier sur le nuage consommation/température.
HDD_BASE = 15.0
CDD_BASE = 22.0

# Couverture minimale exigée des données avant de continuer (sinon erreur).
MIN_COVERAGE = 0.95
