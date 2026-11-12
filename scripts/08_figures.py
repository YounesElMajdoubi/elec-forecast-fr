"""Étape 8 : graphiques du volet 1, pour le README (reports/figures/*.png).

1. prevision_semaines.png : consommation réelle et prévisions sur une semaine
   d'hiver et une semaine d'été du test (deux panneaux, mêmes couleurs).
2. mae_par_mois.png       : erreur moyenne de chaque modèle, mois par mois, sur le test.
3. biais_rte_par_heure.png : biais de la prévision RTE publiée et recalée selon
   l'heure, sur le test (le résultat central du diagnostic).

Couleurs : une couleur par modèle, la même dans tous les graphiques ; au plus
trois modèles en couleur par graphique (au-delà, deux teintes deviennent
difficiles à distinguer, vérifié avec un outil de contrôle des daltonismes).
La consommation réelle est en noir, la prévision RTE publiée en gris.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

from elecfr import config  # noqa: E402
from elecfr.evaluation import mae  # noqa: E402

SEMAINE_HIVER = ("2026-01-12", "2026-01-18")
SEMAINE_ETE = ("2026-06-15", "2026-06-21")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
STYLE = {  # modèle : (libellé, couleur)
    "reel": ("Consommation réelle", INK),
    "rte_j1": ("RTE J-1 publiée", "#9a9993"),
    "lstm": ("LSTM", "#2a78d6"),
    "lightgbm": ("LightGBM", "#eb6834"),
    "rte_j1_recalee": ("RTE J-1 recalée", "#1baf7a"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "legend.frameon": False,
})

df = pd.read_parquet(config.PROCESSED_DIR / "dataset_horaire.parquet")
pdir, out = config.REPORTS_DIR / "previsions", config.REPORTS_DIR / "figures"
out.mkdir(parents=True, exist_ok=True)


def test_preds(models):
    p = {m: pd.read_parquet(pdir / f"{m}_test.parquet")["prevision"] for m in models if (pdir / f"{m}_test.parquet").exists()}
    d = pd.DataFrame(p)
    # On retire les heures sans valeur réelle (heure absente au passage à l'heure
    # d'hiver) : une seule valeur vide rendrait la MAE du mois incalculable.
    return df.loc[d.index, ["consommation", "date_local", "heure"]].join(d).dropna()


JOURS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
fmt_jour = matplotlib.ticker.FuncFormatter(lambda v, _: (lambda d: f"{JOURS[d.weekday()]} {d.day}")(matplotlib.dates.num2date(v)))
fmt_mois = matplotlib.ticker.FuncFormatter(lambda v, _: (lambda d: f"{MOIS[d.month - 1]} {d.year}")(matplotlib.dates.num2date(v)))


def legend_top(ax, ncol):
    """Légende sous le titre, au-dessus du graphique (jamais sur les courbes)."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, fontsize=8, borderaxespad=0.3)


def label_end(ax, x, y, text, color):
    """Libellé direct en bout de courbe : texte à l'encre, petit trait de la couleur du modèle."""
    ax.annotate(f"— {text}", (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK2)
    ax.plot([x], [y], "o", ms=4, color=color, mec=SURFACE, mew=1, zorder=5)


# 1. Deux semaines --------------------------------------------------------------
models = ["rte_j1_recalee", "lightgbm", "lstm"]
d = test_preds(models)
fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharey=False)
for ax, (a, b), titre in zip(axes, (SEMAINE_HIVER, SEMAINE_ETE), ("Semaine d'hiver", "Semaine d'été")):
    w = d[(d["date_local"] >= a) & (d["date_local"] <= b)]
    t = w.index.tz_convert(config.TZ).tz_localize(None)
    for m in ["reel"] + [m for m in models if m in w]:
        y = w["consommation"] if m == "reel" else w[m]
        ax.plot(t, y / 1000, color=STYLE[m][1], lw=2.0 if m == "reel" else 1.5,
                label=STYLE[m][0], zorder=4 if m == "reel" else 3)
    erreurs = ", ".join(f"{STYLE[m][0]} {mae(w['consommation'], w[m]):,.0f}".replace(",", " ") for m in models if m in w)
    ax.set_title(f"{titre} ({a} → {b})", pad=16)
    ax.text(0, 1.015, f"MAE de la semaine (MW) : {erreurs}", transform=ax.transAxes, fontsize=8,
            color=INK2, va="bottom")
    ax.set_ylabel("GW")
    ax.xaxis.set_major_formatter(fmt_jour)
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.16), ncol=4, fontsize=8)
fig.suptitle("Consommation électrique française : réel et prévisions de la veille (test)", x=0.06, ha="left",
             fontsize=12, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(out / "prevision_semaines.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# 2. MAE par mois -----------------------------------------------------------------
models = ["rte_j1", "rte_j1_recalee", "lightgbm", "lstm"]
d = test_preds(models)
mois = d["date_local"].dt.to_period("M")
fig, ax = plt.subplots(figsize=(10, 4.2))
for m in [m for m in models if m in d]:
    s = d.groupby(mois).apply(lambda g, m=m: mae(g["consommation"], g[m]))
    x = s.index.to_timestamp()
    ax.plot(x, s.values, color=STYLE[m][1], marker="o", ms=4, mec=SURFACE, mew=1, label=STYLE[m][0])
ax.set_title("Erreur moyenne (MAE) par mois sur la période de test", pad=24)
ax.set_ylabel("MAE (MW)")
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(fmt_mois)
legend_top(ax, 4)
fig.tight_layout()
fig.savefig(out / "mae_par_mois.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# 3. Biais horaire de RTE -----------------------------------------------------------
models = ["rte_j1", "rte_j1_recalee"]
d = test_preds(models)
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.axhline(0, color=INK2, lw=0.8, zorder=2)
for m in [m for m in models if m in d]:
    b = (d[m] - d["consommation"]).groupby(d["heure"]).mean()
    ax.plot(b.index, b.values, color=STYLE[m][1], marker="o", ms=4, mec=SURFACE, mew=1, lw=2.0, label=STYLE[m][0])
    label_end(ax, b.index[-1], b.values[-1], STYLE[m][0], STYLE[m][1])
ax.set_title("Biais de la prévision RTE selon l'heure (prévision − réel, test)", pad=24)
ax.set_ylabel("Biais moyen (MW)")
ax.set_xlabel("Heure (Paris)")
ax.set_xticks(range(0, 24, 2))
ax.set_xlim(-0.5, 26)
legend_top(ax, 2)
fig.tight_layout()
fig.savefig(out / "biais_rte_par_heure.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Graphiques écrits dans {out}")
