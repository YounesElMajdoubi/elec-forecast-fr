"""Étape 12 (volet 2) : analyse exploratoire des prix day-ahead France.

Imprime les chiffres clés et produit reports/figures/prix_*.png :
1. prix_historique.png     : prix moyen journalier depuis 2015 (crise de 2022 incluse) ;
2. prix_profils.png        : profil horaire selon le type de jour, saisonnalité mensuelle,
                             fréquence des prix négatifs par heure (période récente) ;
3. prix_vs_conso_enr.png   : prix face à la consommation et à la part éolien + solaire.
Puis des diagnostics statsmodels sur le résidu du modèle de base, estimé sur
les 24 mois qui précèdent la validation : test de stationnarité (ADF),
AR(1) estimé par statsmodels (contre-vérification de simulation.fit), et
autocorrélations restantes aux décalages 1 h, 24 h et 168 h.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.ar_model import AutoReg  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller  # noqa: E402

from elecfr import config, simulation  # noqa: E402

RECENT = ("2024-01-01", "2026-10-31")      # période « après la crise » pour les profils
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C3 = ["#2a78d6", "#eb6834", "#1baf7a"]      # semaine, samedi, dimanche/férié
SEQ = "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 1.5, "legend.frameon": False,
})

df = pd.read_parquet(config.PROCESSED_DIR / "prix_horaires.parquet")
out = config.REPORTS_DIR / "figures"
out.mkdir(parents=True, exist_ok=True)
rec = df[(df["date_local"] >= RECENT[0]) & (df["date_local"] <= RECENT[1])].dropna(subset=["prix"])

# --- Chiffres clés -------------------------------------------------------------
print("== Prix par année (€/MWh)")
g = df.groupby("annee")["prix"]
print(pd.DataFrame({"moyenne": g.mean(), "ecart_type": g.std(), "min": g.min(), "max": g.max(),
                    "heures_negatives": g.apply(lambda s: int((s < 0).sum())),
                    "heures_>200": g.apply(lambda s: int((s > 200).sum()))}).round(1).to_string())

tj = simulation.type_jour3(rec)
prof = rec.groupby([tj, rec["heure"]])["prix"].mean().unstack(0)
print(f"\n== Période {RECENT[0]} → {RECENT[1]} : amplitude du profil journalier (max - min de l'heure moyenne)")
for k, nom in enumerate(("semaine", "samedi", "dimanche/férié")):
    print(f"  {nom:<15} {prof[k].max() - prof[k].min():6.1f} €/MWh ; heure la moins chère : {prof[k].idxmin()} h ; "
          f"la plus chère : {prof[k].idxmax()} h")
neg_h = rec.groupby("heure")["prix"].apply(lambda s: (s < 0).mean() * 100)
print(f"  Prix négatifs : {(rec['prix'] < 0).mean():.1%} des heures ; heure la plus touchée : {neg_h.idxmax()} h "
      f"({neg_h.max():.1f} % des cas)")

phys = ["consommation", "eolien", "solaire"]
ctx = rec.dropna(subset=phys) if set(phys) <= set(rec.columns) else rec.iloc[0:0]   # éCO2mix facultatif
if not len(ctx):
    print("\n(éCO2mix absent : pas d'analyse du lien avec la consommation et les renouvelables)")
else:
    enr = (ctx["eolien"] + ctx["solaire"]) / ctx["consommation"] * 100
    print("\n== Corrélations de rang (Spearman) avec le prix, période récente")
    print(f"  consommation : {ctx['prix'].corr(ctx['consommation'], method='spearman'):+.2f}")
    print(f"  part éolien + solaire : {ctx['prix'].corr(enr, method='spearman'):+.2f}")
    print(f"  prix négatifs : part ENR moyenne {enr[ctx['prix'] < 0].mean():.0f} % contre {enr[ctx['prix'] >= 0].mean():.0f} % sinon")

# --- 1. Historique ----------------------------------------------------------------
daily = df.set_index("date_local")["prix"].groupby(level=0).mean()
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(daily.index, daily.values, color=INK, lw=0.8)
ax.set_title("Prix day-ahead France : moyenne journalière depuis 2015", pad=8)
ax.set_ylabel("€/MWh")
pic = daily.idxmax()
ax.annotate(f"Crise de 2022 (pic journalier {daily.max():,.0f} €/MWh)".replace(",", " "), (pic, daily.max()),
            xytext=(12, -4), textcoords="offset points", fontsize=8, color=INK2, va="top")
fig.tight_layout()
fig.savefig(out / "prix_historique.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# --- 2. Profils -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
ax = axes[0]
for k, (nom, col) in enumerate(zip(("Semaine", "Samedi", "Dimanche / férié"), C3)):
    ax.plot(prof.index, prof[k], color=col, label=nom, marker="o", ms=3, mec=SURFACE, mew=0.8)
ax.set_title("Profil horaire moyen", pad=22)
ax.set_xlabel("Heure (Paris)")
ax.set_ylabel("€/MWh")
ax.set_xticks(range(0, 24, 4))
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7.5, borderaxespad=0.2)
ax = axes[1]
mois = rec.groupby("mois")["prix"].mean()
ax.plot(mois.index, mois.values, color=INK, marker="o", ms=3.5, mec=SURFACE, mew=0.8)
ax.set_title("Saisonnalité mensuelle", pad=22)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(list("JFMAMJJASOND"))
ax.set_ylabel("€/MWh")
ax = axes[2]
ax.bar(neg_h.index, neg_h.values, width=0.7, color=SEQ)
ax.set_title("Heures à prix négatif", pad=22)
ax.set_xlabel("Heure (Paris)")
ax.set_ylabel("% des jours")
ax.set_xticks(range(0, 24, 4))
ax.grid(axis="x", visible=False)
fig.suptitle(f"Saisonnalités des prix, {RECENT[0][:4]}–{RECENT[1][:4]} (après la crise)", x=0.01, ha="left",
             fontsize=11.5, fontweight="bold", y=0.99)
fig.tight_layout()
fig.savefig(out / "prix_profils.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# --- 3. Prix, consommation, renouvelables -------------------------------------------------
if len(ctx):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    ylim = np.nanpercentile(ctx["prix"], [0.5, 99.5])
    for ax, x, lab, titre in ((axes[0], ctx["consommation"] / 1000, "Consommation (GW)", "Prix et consommation"),
                              (axes[1], enr, "Éolien + solaire (% de la consommation)", "Prix et renouvelables")):
        hb = ax.hexbin(x, ctx["prix"], gridsize=45, cmap="Blues", mincnt=1, bins="log", linewidths=0.2,
                       extent=(np.nanmin(x), np.nanmax(x), ylim[0], ylim[1]))
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel(lab)
        ax.set_title(titre, pad=6)
        ax.grid(False)
    axes[0].set_ylabel("Prix (€/MWh)")
    cb = fig.colorbar(hb, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("Nombre d'heures (échelle log)")
    fig.suptitle(f"Heures de {RECENT[0][:4]} à {RECENT[1][:4]}", x=0.01, ha="left", fontsize=11.5,
                 fontweight="bold", y=1.03)
    fig.savefig(out / "prix_vs_conso_enr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

# --- Diagnostics statsmodels du modèle de base ------------------------------------------------
fin = pd.Timestamp(config.VAL_START, tz=config.TZ).tz_convert("UTC")
w = df[(df.index >= fin - pd.DateOffset(months=24)) & (df.index < fin)]
m = simulation.fit(w["prix"], w, sauts=False)
resid = (w["prix"] - simulation.design(w, m.t0) @ m.beta).interpolate(limit=3).dropna()
ar = AutoReg(resid.to_numpy(), lags=1, trend="n").fit()
try:   # statsmodels récent : forme du résultat à préciser explicitement
    adf = adfuller(resid.to_numpy()[-8760:], maxlag=48, autolag=None, result_object=False)
except TypeError:
    adf = adfuller(resid.to_numpy()[-8760:], maxlag=48, autolag=None)
r_acf = acf(resid.to_numpy(), nlags=168, fft=True)
print(f"\n== Diagnostics du résidu (24 mois avant {config.VAL_START}, {len(resid):,} heures)")
print(f"  φ : simulation.fit {m.phi:.4f} ; statsmodels AutoReg {ar.params[0]:.4f} → demi-vie {m.demi_vie_h:.1f} h")
print(f"  ADF (dernière année) : statistique {adf[0]:.2f}, p-valeur {adf[1]:.4f} "
      f"({'stationnaire : un retour à la moyenne est justifié' if adf[1] < 0.05 else 'NON stationnaire'})")
print(f"  autocorrélation du résidu : 1 h {r_acf[1]:.2f} ; 24 h {r_acf[24]:.2f} ; 168 h {r_acf[168]:.2f}")
print("  (AR(1) pur : l'autocorrélation à 24 h devrait valoir φ^24 = "
      f"{m.phi ** 24:.2f} ; un écart signale une mémoire journalière que le modèle ignore)")
print(f"\nGraphiques écrits dans {out}")
