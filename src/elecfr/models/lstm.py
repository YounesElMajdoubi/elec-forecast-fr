"""LSTM encodeur-décodeur pour la prévision des 24 heures du lendemain.

Idée. Un réseau récurrent (LSTM) lit une séquence pas à pas et garde une
« mémoire » (état caché) de ce qu'il a vu. Deux LSTM s'enchaînent :

1. ENCODEUR : lit la consommation horaire de la dernière semaine CONNUE à
   l'émission, du jour J-8 au jour J-2 (168 heures), avec l'heure et un
   indicateur de jour chômé. Il résume cette semaine dans son état final :
   niveau récent, forme de la courbe, tendance.
2. DÉCODEUR : part de ce résumé et avance heure par heure sur les 24 heures du
   jour J, en recevant à chaque pas ce qu'on sait déjà de cette heure :
   température prévue, degrés-jours, calendrier, type de jour, et les retards
   J-2, J-7, J-14 de la même heure. Une couche linéaire transforme chaque état
   en une consommation.

Pourquoi J-2 et pas J-1 : la prévision est émise la veille à 10 h, quand le
jour J-1 n'est connu qu'en partie (DECISIONS.md, D1 et D6). Un test vérifie
que modifier la consommation de J-1 ou de J ne change pas la prévision de J.

Données : une ligne = un jour local, 24 heures « murales » (0 h … 23 h). Les
jours de changement d'heure (23 ou 25 h) sont ramenés à 24 heures : l'heure
doublée d'octobre est moyennée, l'heure absente de mars est comblée par la
valeur précédente (jamais par une valeur future).

Entraînement : perte absolue moyenne (comme la MAE), optimiseur Adam.
- Phase 1 : on met de côté les 56 derniers jours de la fenêtre d'entraînement
  et on arrête quand l'erreur sur ces jours ne baisse plus (arrêt précoce),
  ce qui donne le bon nombre d'époques.
- Phase 2 : on réentraîne sur TOUTE la fenêtre pendant ce nombre d'époques,
  pour ne pas priver le modèle des 8 semaines les plus récentes.
Graine fixe pour la reproductibilité.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from elecfr.models.common import TARGET, type_jour

HIST_DAYS = 7      # longueur de l'historique lu par l'encodeur
GAP_DAYS = 2       # dernier jour d'historique = J-2
HOLDOUT_DAYS = 56  # jours mis de côté pour l'arrêt précoce
CONSO_LAGS = ["conso_j2", "conso_j7", "conso_j14"]
DEC_NUM = ["temp_prev", "dj_chauffage", "dj_clim", "zones_vacances", "pont",
           "changement_heure", "annee_sin", "annee_cos"]


def _grid(df: pd.DataFrame, col: str, days: pd.DatetimeIndex) -> np.ndarray:
    """Tableau [jour, heure murale 0..23] d'une colonne (moyenne si heure doublée)."""
    t = df.pivot_table(index="date_local", columns="heure", values=col, aggfunc="mean")
    return t.reindex(index=days, columns=range(24)).to_numpy(dtype=float)


class DayArrays:
    """Toutes les données du jeu complet rangées par jour × heure, une seule fois."""

    def __init__(self, base: pd.DataFrame):
        days = pd.date_range(base["date_local"].min(), base["date_local"].max(), freq="D")
        self.days = days
        self.pos = pd.Series(np.arange(len(days)), index=days)
        self.conso = _grid(base, TARGET, days)
        b = base.assign(type_jour=type_jour(base),
                        jour_chome=((type_jour(base) >= 5) | (base["pont"] == 1)).astype(float))
        self.dec_num = np.stack([_grid(b, c, days) for c in DEC_NUM], axis=-1)
        self.lags = np.stack([_grid(b, c, days) for c in CONSO_LAGS], axis=-1)
        tj = np.nan_to_num(_grid(b, "type_jour", days), nan=0).astype(int)
        self.type_jour = np.eye(7)[tj]                                   # [jour, 24, 7]
        self.chome = np.nan_to_num(_grid(b, "jour_chome", days))
        h = np.arange(24)
        self.hsin = np.broadcast_to(np.sin(2 * np.pi * h / 24), (len(days), 24))
        self.hcos = np.broadcast_to(np.cos(2 * np.pi * h / 24), (len(days), 24))
        # Historique de consommation : on comble les trous avec la dernière
        # valeur CONNUE (remplissage vers l'avant uniquement).
        self.conso_hist = pd.Series(self.conso.ravel()).ffill().to_numpy().reshape(self.conso.shape)


class LSTMModel:
    nom = "lstm"

    def __init__(self, base: pd.DataFrame | None = None, hidden: int = 64,
                 max_epochs: int = 80, patience: int = 8, batch_size: int = 32,
                 lr: float = 1e-3, seed: int = 0):
        self.base = base
        self.params = {"hidden": hidden, "max_epochs": max_epochs, "patience": patience,
                       "batch_size": batch_size, "lr": lr, "seed": seed}
        self.dernier = None
        self.variables = ["encodeur: conso, heure_sin, heure_cos, jour_chome",
                          "decodeur: " + ", ".join(DEC_NUM + CONSO_LAGS) + ", heure_sin, heure_cos, type_jour(7)"]
        self.epoques: list[int] = []
        self._arrays = None

    # ------------------------------------------------------------------ données
    def _arrays_for(self, df):
        if self._arrays is None:
            self._arrays = DayArrays(self.base if self.base is not None else df)
        return self._arrays

    def _inputs(self, A: DayArrays, idx: np.ndarray, scal: dict):
        """Entrées de l'encodeur [n, 168, 4] et du décodeur [n, 24, 20] pour les
        jours d'indices `idx`. L'historique couvre les jours idx-8 … idx-2."""
        hist = np.stack([np.arange(i - GAP_DAYS - HIST_DAYS + 1, i - GAP_DAYS + 1) for i in idx])  # [n, 7]
        c = (A.conso_hist[hist] - scal["mu"]) / scal["sd"]                                         # [n, 7, 24]
        enc = np.stack([c, A.hsin[hist], A.hcos[hist], A.chome[hist]], axis=-1).reshape(len(idx), HIST_DAYS * 24, 4)
        num = (A.dec_num[idx] - scal["num_mu"]) / scal["num_sd"]
        lag = (A.lags[idx] - scal["mu"]) / scal["sd"]
        dec = np.concatenate([num, lag, A.hsin[idx][..., None], A.hcos[idx][..., None], A.type_jour[idx]], axis=-1)
        return np.nan_to_num(enc).astype(np.float32), np.nan_to_num(dec).astype(np.float32)

    # ------------------------------------------------------------- entraînement
    def __call__(self, train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
        import torch

        A = self._arrays_for(train)
        p = self.params
        train_days = pd.DatetimeIndex(sorted(train.loc[train[TARGET].notna(), "date_local"].unique()))
        idx = A.pos.reindex(train_days).dropna().astype(int).to_numpy()
        idx = idx[idx >= GAP_DAYS + HIST_DAYS - 1]                          # historique complet disponible
        idx = idx[~np.isnan(A.dec_num[idx, :, 0]).any(axis=1)]              # température prévue complète (D18)

        y_all = A.conso[idx]
        scal = {"mu": np.nanmean(y_all), "sd": np.nanstd(y_all)}
        num = A.dec_num[idx].reshape(-1, len(DEC_NUM))
        scal["num_mu"], scal["num_sd"] = np.nanmean(num, axis=0), np.nanstd(num, axis=0) + 1e-6
        enc, dec = self._inputs(A, idx, scal)
        y = ((y_all - scal["mu"]) / scal["sd"]).astype(np.float32)

        cut = len(idx) - HOLDOUT_DAYS
        _, best_epoch = self._fit(enc[:cut], dec[:cut], y[:cut], (enc[cut:], dec[cut:], y[cut:]), p["max_epochs"])
        net, _ = self._fit(enc, dec, y, None, best_epoch)                   # phase 2 : toute la fenêtre
        self.epoques.append(best_epoch)
        self.dernier = net

        # Prévision des jours du bloc, puis retour aux lignes (heures UTC).
        tdays = pd.DatetimeIndex(sorted(target["date_local"].unique()))
        tidx = A.pos.reindex(tdays).astype(int).to_numpy()
        tenc, tdec = self._inputs(A, tidx, scal)
        net.eval()
        with torch.no_grad():
            out = net(torch.from_numpy(tenc), torch.from_numpy(tdec)).numpy() * scal["sd"] + scal["mu"]
        day_row = pd.Series(np.arange(len(tdays)), index=tdays)
        r = day_row.reindex(target["date_local"]).to_numpy()
        return out[r, target["heure"].to_numpy()]

    def _fit(self, enc, dec, y, holdout, epochs):
        """Entraîne un réseau neuf. Avec `holdout`, arrêt précoce et renvoi de la
        meilleure époque ; sans, entraînement pendant exactement `epochs` époques."""
        import torch
        from torch import nn

        p = self.params
        torch.manual_seed(p["seed"])
        rng = np.random.default_rng(p["seed"])
        net = Seq2Seq(enc.shape[-1], dec.shape[-1], p["hidden"])
        opt = torch.optim.Adam(net.parameters(), lr=p["lr"])
        E, D, Y = torch.from_numpy(enc), torch.from_numpy(dec), torch.from_numpy(y)
        mask = ~torch.isnan(Y)
        Y = torch.nan_to_num(Y)
        best, best_epoch, wait, best_state = np.inf, max(epochs, 1), 0, None
        for epoch in range(1, epochs + 1):
            net.train()
            for b in np.array_split(rng.permutation(len(E)), max(1, len(E) // p["batch_size"])):
                opt.zero_grad()
                out = net(E[b], D[b])
                loss = (out - Y[b]).abs()[mask[b]].mean()        # perte absolue, heures connues seulement
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # évite les pas trop grands
                opt.step()
            if holdout is not None:
                he, hd, hy = (torch.from_numpy(a) for a in holdout)
                hm = ~torch.isnan(hy)
                net.eval()
                with torch.no_grad():
                    v = (net(he, hd) - torch.nan_to_num(hy)).abs()[hm].mean().item()
                if v < best - 1e-4:
                    best, best_epoch, wait = v, epoch, 0
                    best_state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= p["patience"]:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_epoch


def _seq2seq_class():
    from torch import nn

    class _Seq2Seq(nn.Module):
        def __init__(self, n_enc, n_dec, hidden):
            super().__init__()
            self.enc = nn.LSTM(n_enc, hidden, batch_first=True)
            self.dec = nn.LSTM(n_dec, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x_enc, x_dec):
            _, state = self.enc(x_enc)            # résumé de la semaine passée
            out, _ = self.dec(x_dec, state)       # 24 pas, un par heure du jour J
            return self.head(out).squeeze(-1)     # [lot, 24]

    return _Seq2Seq


def Seq2Seq(n_enc, n_dec, hidden):  # noqa: N802 - fabrique, torch importé à la demande
    return _seq2seq_class()(n_enc, n_dec, hidden)


GRID = [{"hidden": h} for h in (32, 64)]
