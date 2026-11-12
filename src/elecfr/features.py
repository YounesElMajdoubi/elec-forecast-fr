"""Construction du jeu horaire et des variables, sans fuite d'information.

Principe : chaque ligne est une heure cible t d'un jour local J. Toutes ses
variables doivent être connues au moment d'émission de la prévision,
I(J) = J-1 à FORECAST_ISSUE_HOUR_LOCAL heure de Paris.

- Calendrier : connu à l'avance, aucun risque.
- Température : prévision météo émise avant I(J) (voir `temp_known_at_issue`).
- Consommation passée : seulement des jours entièrement écoulés à I(J),
  c'est-à-dire J-2 et avant (J-1 n'est connu que jusqu'à 10 h).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from elecfr import config
from elecfr.data.calendar import bridge_days, school_holiday_zone_count

SEASONS = {12: "hiver", 1: "hiver", 2: "hiver", 3: "printemps", 4: "printemps",
           5: "printemps", 6: "été", 7: "été", 8: "été", 9: "automne",
           10: "automne", 11: "automne"}


def local_index(idx_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx_utc.tz_convert(config.TZ)


def issue_time(target_local: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Instant d'émission (UTC) de la prévision pour chaque heure cible."""
    day = target_local.normalize().tz_localize(None)            # minuit local de J, naïf
    issue_naive = day - pd.Timedelta(days=1) + pd.Timedelta(hours=config.FORECAST_ISSUE_HOUR_LOCAL)
    return issue_naive.tz_localize(config.TZ).tz_convert("UTC")


def temp_known_at_issue(t_fc_d1: pd.Series, t_fc_d2: pd.Series,
                        lag_h: int = config.WEATHER_AVAILABILITY_LAG_H) -> pd.Series:
    """Choisit, heure par heure, la prévision la plus récente déjà disponible.

    La valeur « previous_day1 » pour l'heure t vient d'un run lancé environ
    24 h avant t et publié environ `lag_h` heures plus tard. Elle est donc
    utilisable si t - 24 h + lag_h <= I. Sinon on prend « previous_day2 »
    (run de t - 48 h), toujours disponible pour les heures de J.
    """
    idx = t_fc_d1.index
    issue = issue_time(local_index(idx))
    usable_d1 = (idx - pd.Timedelta(hours=24) + pd.Timedelta(hours=lag_h)) <= issue
    out = t_fc_d2.copy()
    out[usable_d1] = t_fc_d1[usable_d1]
    return out.rename("temp_prev")


def same_local_hour_lag(series: pd.Series, days: int) -> pd.Series:
    """Valeur à la même heure LOCALE, `days` jours plus tôt.

    On compare des heures « murales » (heure affichée à Paris) et non des écarts
    de 24 h en UTC, pour que le naïf saisonnier compare bien 18 h à 18 h même
    quand un changement d'heure tombe dans la semaine. L'heure doublée d'octobre
    est moyennée ; l'heure qui n'existe pas en mars donne NaN.
    """
    wall = local_index(series.index).tz_localize(None)
    by_wall = series.groupby(wall).mean()
    looked_up = by_wall.reindex(wall - pd.Timedelta(days=days))
    return pd.Series(looked_up.to_numpy(), index=series.index)


def build_dataset(hourly_elec: pd.DataFrame, temps: pd.DataFrame, hols: set,
                  school: pd.DataFrame | None) -> pd.DataFrame:
    idx = hourly_elec.index
    loc = local_index(idx)
    df = pd.DataFrame(index=idx)
    df["consommation"] = hourly_elec["consommation"]
    df["rte_j1"] = hourly_elec["prevision_j1"]
    df["n_cons"] = hourly_elec["n_cons"]

    # Calendrier (heure locale)
    day = loc.normalize().tz_localize(None)
    df["date_local"] = day
    df["heure"] = loc.hour
    df["jour_semaine"] = loc.day_of_week
    df["mois"] = loc.month
    df["jour_annee"] = loc.dayofyear
    df["saison"] = pd.Series(loc.month, index=idx).map(SEASONS)
    df["week_end"] = (loc.day_of_week >= 5).astype(int)
    df["ferie"] = pd.Series(day.date, index=idx).isin(hols).astype(int)
    df["pont"] = bridge_days(day, hols).to_numpy().astype(int)
    if school is not None:
        uniq = pd.DatetimeIndex(day.unique())
        counts = school_holiday_zone_count(school, uniq)
        df["zones_vacances"] = counts.reindex(day).to_numpy()
    hours_in_day = pd.Series(1, index=idx).groupby(day).transform("sum")
    df["changement_heure"] = (hours_in_day != 24).astype(int)
    # Encodage cyclique : 23 h et 0 h sont voisines, ce qu'un entier ne dit pas.
    df["heure_sin"] = np.sin(2 * np.pi * df["heure"] / 24)
    df["heure_cos"] = np.cos(2 * np.pi * df["heure"] / 24)
    df["annee_sin"] = np.sin(2 * np.pi * df["jour_annee"] / 365.25)
    df["annee_cos"] = np.cos(2 * np.pi * df["jour_annee"] / 365.25)

    # Température connue la veille + non-linéarité chauffage / climatisation
    t = temps.reindex(idx)
    df["temp_prev"] = temp_known_at_issue(t["t_fc_d1"], t["t_fc_d2"])
    df["dj_chauffage"] = (config.HDD_BASE - df["temp_prev"]).clip(lower=0)
    df["dj_clim"] = (df["temp_prev"] - config.CDD_BASE).clip(lower=0)
    df["temp_obs"] = t["t_obs"]  # ANALYSE SEULEMENT : interdit comme variable

    # Retards de consommation : J-2, J-7, J-14 (même heure locale)
    for d in (2, 7, 14):
        df[f"conso_j{d}"] = same_local_hour_lag(df["consommation"], d)
    return df


# Variables autorisées pour les modèles. `temp_obs`, `rte_j1` et la cible en
# sont exclues ; un test vérifie que cette liste ne contient pas de fuite.
FEATURES = [
    "heure", "jour_semaine", "mois", "week_end", "ferie", "pont", "zones_vacances",
    "changement_heure", "heure_sin", "heure_cos", "annee_sin", "annee_cos",
    "temp_prev", "dj_chauffage", "dj_clim", "conso_j2", "conso_j7", "conso_j14",
]
FORBIDDEN = {"consommation", "temp_obs", "rte_j1", "n_cons"}
