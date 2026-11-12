"""Prix day-ahead de la zone France (ENTSO-E Transparency Platform).

Jeton : variable d'environnement ENTSOE_TOKEN (fichier .env, jamais commité).

Attention : depuis le 1er octobre 2025 (jour de livraison), le marché
day-ahead européen fixe un prix par QUART D'HEURE (96 prix par jour) au lieu
d'un prix par heure. On ramène tout à l'heure par une moyenne simple des quatre
quarts d'heure, et on garde la résolution d'origine dans `resolution_min`.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from elecfr.config import TZ


def _client():
    from dotenv import load_dotenv
    from entsoe import EntsoePandasClient

    load_dotenv()
    token = os.environ.get("ENTSOE_TOKEN", "").strip()
    if not token or token == "colle-ton-jeton-ici":
        raise RuntimeError("ENTSOE_TOKEN manquant : copie .env.example en .env et colles-y ton jeton")
    return EntsoePandasClient(api_key=token)


def today_paris() -> str:
    """Date du jour à Paris (AAAA-MM-JJ) : les prix de demain sont publiés vers 13 h."""
    return str(pd.Timestamp.now(tz=TZ).date())


def _query_by_month(client, s: pd.Timestamp, e: pd.Timestamp) -> pd.Series:
    """Interroge ENTSO-E mois par mois, sur [s, e[.

    Pourquoi pas une année d'un coup : entsoe-py élargit chaque demande d'un
    jour de chaque côté, puis découpe ce qui dépasse un an en blocs annuels ; il
    perd alors le point situé à la jonction des blocs (le 31 décembre à 0 h,
    heure de Paris, chaque année). Une demande d'un mois n'est jamais découpée."""
    parts = []
    debut = s
    while debut < e:
        fin = min(debut + pd.DateOffset(months=1), e)
        ser = client.query_day_ahead_prices("FR", start=debut, end=fin)
        parts.append(ser[(ser.index >= debut) & (ser.index < fin)])   # entsoe-py inclut aussi l'instant de fin
        debut = fin
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated()]


def download_prices(start: str, end: str, out_dir: Path, client=None) -> pd.Series:
    """Télécharge par année (une demande par mois), met en cache, renvoie la
    série brute (index UTC, résolution d'origine).

    Une année en cache est retéléchargée si elle ne va pas jusqu'à la fin
    demandée : c'est le cas de l'année en cours, qui s'allonge chaque jour.
    Les fichiers de cache portent le suffixe « _mensuel » : ceux d'avant la
    correction du téléchargement (D39) sont ignorés et refaits."""
    client = client or _client()
    parts = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        out = out_dir / f"prix_fr_{y}_mensuel.parquet"
        s = max(pd.Timestamp(start, tz=TZ), pd.Timestamp(f"{y}-01-01", tz=TZ))
        e = min(pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1), pd.Timestamp(f"{y + 1}-01-01", tz=TZ))
        complet = False
        if out.exists():
            cache = pd.read_parquet(out)["prix"]
            # dernier prix attendu : la dernière heure avant e (un quart d'heure suffit)
            complet = len(cache) > 0 and cache.index.max() >= e - pd.Timedelta(hours=1)
        if not complet:
            ser = _query_by_month(client, s, e)
            out_dir.mkdir(parents=True, exist_ok=True)
            ser.rename("prix").to_frame().to_parquet(out)
        parts.append(pd.read_parquet(out)["prix"])
    raw = pd.concat(parts).sort_index()
    raw = raw[~raw.index.duplicated()]
    raw.index = raw.index.tz_convert("UTC")
    return raw


def to_hourly(raw: pd.Series) -> pd.DataFrame:
    """Moyenne horaire des prix (€/MWh) + résolution d'origine en minutes.

    Une heure incomplète (par exemple un seul quart d'heure sur quatre, en bord
    de période) est retirée : sa « moyenne » ne serait que le prix d'un quart d'heure."""
    raw = raw.dropna()
    heure = raw.index.floor("h")
    step = raw.index.to_series().diff().dt.total_seconds().div(60)
    step = step.where(step > 0).fillna(step.median() if step.notna().any() else 60)
    res = step.groupby(heure).min().clip(upper=60)
    n = raw.groupby(heure).size()
    out = raw.groupby(heure).mean().to_frame("prix")
    out["resolution_min"] = res
    out = out[n * res >= 60]
    out.index.name = "heure_utc"
    return out


def compare_sources(a: pd.Series, b: pd.Series, tol: float = 0.01,
                    noms: tuple[str, str] = ("ember", "entsoe")) -> dict:
    """Compare deux séries de prix horaires (index UTC) heure par heure.

    Renvoie un résumé (heures communes, heures propres à chaque source, part
    des heures identiques à `tol` €/MWh près, écart absolu moyen et maximal),
    le détail par année (en heure de Paris) et la table des heures qui diffèrent."""
    na, nb = noms
    df = pd.concat([a.rename(na), b.rename(nb)], axis=1, sort=True)
    both = df.dropna()
    diff = both[nb] - both[na]
    ecart = diff.abs()
    seulement_a = df.index[df[na].notna() & df[nb].isna()]
    seulement_b = df.index[df[nb].notna() & df[na].isna()]
    resume = {
        "heures_communes": int(len(both)),
        f"heures_seulement_{na}": int(len(seulement_a)),
        f"heures_seulement_{nb}": int(len(seulement_b)),
        "part_identiques": float((ecart <= tol).mean()) if len(both) else np.nan,
        "ecart_moyen": float(ecart.mean()) if len(both) else np.nan,
        "ecart_moyen_signe": float(diff.mean()) if len(both) else np.nan,
        "ecart_max": float(ecart.max()) if len(both) else np.nan,
        "heure_ecart_max": ecart.idxmax() if len(both) else None,
    }
    annee = both.index.tz_convert(TZ).year
    par_annee = pd.DataFrame({
        "heures": ecart.groupby(annee).size(),
        "part_identiques": (ecart <= tol).groupby(annee).mean(),
        "ecart_moyen": ecart.groupby(annee).mean(),
        "ecart_max": ecart.groupby(annee).max(),
    })
    par_annee.index.name = "annee"
    differences = both[ecart > tol].assign(ecart=diff[ecart > tol])
    return {"resume": resume, "par_annee": par_annee, "differences": differences,
            f"seulement_{na}": seulement_a, f"seulement_{nb}": seulement_b}
