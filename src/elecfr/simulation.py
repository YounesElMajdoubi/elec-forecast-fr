"""Simulation de trajectoires du prix day-ahead (volet 2).

Modèle : prix(t) = partie déterministe(t) + résidu(t)

1. Partie déterministe (moindres carrés ordinaires) :
   - profil journalier selon le type de jour : 24 heures × 3 types
     (semaine, samedi, dimanche/férié) ;
   - saisonnalité annuelle : sin/cos du jour de l'année (2 harmoniques) ;
   - tendance linéaire sur la fenêtre d'estimation.
2. Résidu à retour à la moyenne, AR(1) en temps discret :
       x(t) = φ·x(t-1) + ε(t)
   C'est la version horaire d'un processus d'Ornstein-Uhlenbeck : un écart
   au prix « normal » se résorbe à la vitesse κ = -ln φ par heure (demi-vie
   ln 2 / κ). φ est estimé par régression de x(t) sur x(t-1).
3. Amélioration « sauts » : les chocs ε trop grands pour une loi normale
   (|ε| > SEUIL_SAUT × écart-type robuste) sont traités à part. À chaque
   heure, un saut survient avec la probabilité observée p, et sa taille est
   tirée parmi les sauts réellement observés (pics vers le haut ET prix
   négatifs). Le reste des chocs est gaussien. Les sauts se résorbent
   ensuite par le même retour à la moyenne.
4. Option « ancrage » : la partie déterministe est recalée pour que sa
   moyenne sur les N derniers jours connus égale le prix moyen observé
   sur ces jours. Utile quand le niveau des prix change (prix du gaz, du CO2).

Tout est en NumPy pur (pas de statsmodels ici) : la même fonction sert au
script de validation et à l'API, sans seconde implémentation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEUIL_SAUT = 4.0          # en écarts-types robustes
TYPES_JOUR = 3            # semaine, samedi, dimanche/férié
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def type_jour3(cal: pd.DataFrame) -> np.ndarray:
    """0 = lundi-vendredi, 1 = samedi, 2 = dimanche ou jour férié."""
    t = np.where(cal["jour_semaine"].to_numpy() == 5, 1, 0)
    t = np.where((cal["jour_semaine"].to_numpy() == 6) | (cal["ferie"].to_numpy() == 1), 2, t)
    return t


def design(cal: pd.DataFrame, t0: pd.Timestamp) -> np.ndarray:
    """Matrice de la partie déterministe. `cal` : heure, jour_semaine, ferie,
    jour_annee (heure locale) ; `t0` : origine de la tendance (UTC)."""
    h, tj = cal["heure"].to_numpy(), type_jour3(cal)
    cols = []
    for k in range(TYPES_JOUR):
        for hour in range(24):
            cols.append(((h == hour) & (tj == k)).astype(float))   # 72 indicatrices (pas d'intercept séparé)
    doy = cal["jour_annee"].to_numpy()
    for harm in (1, 2):
        cols += [np.sin(2 * np.pi * harm * doy / 365.25), np.cos(2 * np.pi * harm * doy / 365.25)]
    years = (cal.index - t0).total_seconds().to_numpy() / (365.25 * 86400)
    cols.append(years)
    return np.column_stack(cols)


@dataclass
class PriceModel:
    beta: np.ndarray
    t0: pd.Timestamp
    phi: float
    sigma: float                     # écart-type des chocs gaussiens
    p_saut: float = 0.0
    sauts: np.ndarray = field(default_factory=lambda: np.array([]))
    decalage: float = 0.0            # recalage du niveau (ancrage)
    dernier_residu: float = 0.0      # écart au « normal » à la dernière heure connue
    infos: dict = field(default_factory=dict)

    @property
    def demi_vie_h(self) -> float:
        return float(np.log(2) / -np.log(self.phi)) if 0 < self.phi < 1 else float("inf")


def fit(prix: pd.Series, cal: pd.DataFrame, sauts: bool = True, ancrage_jours: int | None = None) -> PriceModel:
    """Estime le modèle sur une fenêtre d'heures consécutives (index UTC)."""
    ok = prix.notna().to_numpy()
    t0 = prix.index[0]
    X = design(cal, t0)
    beta, *_ = np.linalg.lstsq(X[ok], prix.to_numpy()[ok], rcond=None)
    resid = pd.Series(prix.to_numpy() - X @ beta, index=prix.index)

    # AR(1) sur les paires d'heures consécutives toutes deux connues
    r, r1 = resid.to_numpy()[1:], resid.to_numpy()[:-1]
    pair = ~np.isnan(r) & ~np.isnan(r1)
    phi = float(np.dot(r1[pair], r[pair]) / np.dot(r1[pair], r1[pair]))
    eps = r[pair] - phi * r1[pair]

    p_saut, tailles = 0.0, np.array([])
    if sauts:
        med = np.median(eps)
        s_rob = 1.4826 * np.median(np.abs(eps - med))       # écart-type robuste (MAD)
        is_jump = np.abs(eps - med) > SEUIL_SAUT * s_rob
        p_saut, tailles = float(is_jump.mean()), eps[is_jump]
        sigma = float(np.std(eps[~is_jump]))
    else:
        sigma = float(np.std(eps))

    decalage = 0.0
    if ancrage_jours:
        recent = prix.index >= prix.index[-1] - pd.Timedelta(days=ancrage_jours)
        decalage = float(np.nanmean(resid.to_numpy()[recent]))

    last = resid.dropna()
    return PriceModel(beta=beta, t0=t0, phi=phi, sigma=sigma, p_saut=p_saut, sauts=tailles,
                      decalage=decalage, dernier_residu=float(last.iloc[-1]) - decalage,
                      infos={"n_heures": int(ok.sum()), "sauts": bool(sauts), "ancrage_jours": ancrage_jours or 0})


def simulate(model: PriceModel, cal_future: pd.DataFrame, n: int = 1000, seed: int = 0) -> np.ndarray:
    """`n` trajectoires horaires sur les heures de `cal_future` (index UTC,
    à la suite de la fenêtre d'estimation). Renvoie un tableau [n, heures]."""
    rng = np.random.default_rng(seed)
    det = design(cal_future, model.t0) @ model.beta + model.decalage
    H = len(det)
    x = np.full(n, model.dernier_residu)
    out = np.empty((n, H))
    for t in range(H):
        shock = rng.normal(0.0, model.sigma, n)
        if model.p_saut > 0 and len(model.sauts):
            jump = rng.random(n) < model.p_saut
            shock[jump] = rng.choice(model.sauts, size=int(jump.sum()))
        x = model.phi * x + shock
        out[:, t] = det[t] + x
    return out


# --- Un mois complet : utilisé par la validation (script 13) ET par l'API ------
FENETRE_MOIS = 24
VARIANTES = {"ou": {"sauts": False}, "ou_sauts": {"sauts": True},
             "ou_sauts_ancrage": {"sauts": True, "ancrage_jours": 28}}


def calendrier(index_utc: pd.DatetimeIndex, tz: str = "Europe/Paris") -> pd.DataFrame:
    """Variables de calendrier (heure locale) pour des heures UTC quelconques,
    y compris futures : heure, jour_semaine, jour_annee, ferie, date_local."""
    import holidays

    loc = index_utc.tz_convert(tz)
    cal = pd.DataFrame(index=index_utc)
    cal["date_local"] = loc.normalize().tz_localize(None)
    cal["heure"], cal["jour_semaine"], cal["jour_annee"] = loc.hour, loc.day_of_week, loc.dayofyear
    hols = holidays.France(years=list(range(loc.year.min(), loc.year.max() + 1)))
    cal["ferie"] = cal["date_local"].dt.date.isin(set(hols.keys())).astype(int)
    return cal


def bornes_mois(mois: str | pd.Timestamp, tz: str = "Europe/Paris") -> tuple[pd.Timestamp, pd.Timestamp]:
    """Début et fin (exclue) d'un mois local, en UTC."""
    m0 = pd.Timestamp(mois).to_period("M").to_timestamp()
    start = pd.Timestamp(m0, tz=tz).tz_convert("UTC")
    end = pd.Timestamp(m0 + pd.offsets.MonthBegin(1), tz=tz).tz_convert("UTC")
    return start, end


def simulate_month(prix_df: pd.DataFrame, mois, params: dict, n: int = 1000,
                   fenetre_mois: int = FENETRE_MOIS):
    """Estime le modèle sur les `fenetre_mois` mois qui précèdent `mois` (tout
    ce qui est connu avant le 1er du mois), puis simule `n` trajectoires de
    chaque heure du mois. Renvoie (modèle, trajectoires [n, heures],
    calendrier du mois, prix réels du mois (NaN si inconnus))."""
    start, end = bornes_mois(mois)
    train = prix_df[(prix_df.index >= start - pd.DateOffset(months=fenetre_mois)) & (prix_df.index < start)]
    if train.empty or train.index[-1] < start - pd.Timedelta(hours=2):
        raise ValueError(f"Prix inconnus juste avant le {start.tz_convert('Europe/Paris'):%d/%m/%Y} : "
                         f"dernière heure connue {prix_df['prix'].dropna().index.max()}")
    model = fit(train["prix"], train, **params)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    cal = calendrier(hours)
    sims = simulate(model, cal, n=n, seed=int(pd.Timestamp(start).tz_convert("Europe/Paris").strftime("%Y%m")))
    reel = prix_df["prix"].reindex(hours).to_numpy()
    model.infos.update({"entraine_du": str(train.index[0]), "entraine_au": str(train.index[-1])})
    return model, sims, cal, reel


# --- Évaluation probabiliste -------------------------------------------------
def crps(sims: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS d'un ensemble, heure par heure (€/MWh ; plus petit = meilleur).
    CRPS = E|X - y| - ½ E|X - X'|, calculé en O(n log n) grâce au tri :
    Σ_i Σ_j |x_i - x_j| = 2 Σ_i (2i - n - 1) x_(i)."""
    n = sims.shape[0]
    xs = np.sort(sims, axis=0)
    term1 = np.abs(sims - y).mean(axis=0)
    w = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    term2 = (w * xs).sum(axis=0) / (n * n)
    return term1 - term2


def evaluate(sims: np.ndarray, y: np.ndarray) -> dict:
    """Couverture des intervalles, largeur, CRPS, erreur de la médiane, PIT."""
    ok = ~np.isnan(y)
    s, y = sims[:, ok], y[ok]
    q = np.quantile(s, QUANTILES, axis=0)
    pit = (s < y).mean(axis=0)                                    # rang du réel parmi les trajectoires
    return {
        "couverture_90": float(((y >= q[0]) & (y <= q[4])).mean()),
        "couverture_50": float(((y >= q[1]) & (y <= q[3])).mean()),
        "largeur_90": float((q[4] - q[0]).mean()),
        "crps": float(crps(s, y).mean()),
        "mae_mediane": float(np.abs(q[2] - y).mean()),
        "part_negatifs_reel": float((y < 0).mean()),
        "part_negatifs_simule": float((s < 0).mean()),
        "moyenne_reelle": float(y.mean()),
        "moyenne_simulee_q05": float(np.quantile(s.mean(axis=1), 0.05)),
        "moyenne_simulee_q95": float(np.quantile(s.mean(axis=1), 0.95)),
        "pit": pit,
    }
