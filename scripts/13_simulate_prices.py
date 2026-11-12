"""Étape 13 (volet 2) : 1 000 trajectoires de prix par mois, validées sur les prix réels.

Protocole, calqué sur le volet 1 :
- Pour chaque mois M, le modèle est estimé sur les 24 mois qui précèdent
  (tout ce qui est connu la veille du 1er du mois : les prix d'un jour sont
  publiés la veille), puis 1 000 trajectoires horaires de M sont simulées.
- Cinq variantes : « ou » (déterministe + retour à la moyenne),
  « ou_sauts » (+ sauts), « ou_sauts_ancrage » (+ niveau recalé sur les
  28 derniers jours), et le modèle à deux facteurs (niveau journalier
  persistant + écart horaire) sans et avec ancrage.
- Choix de la variante sur la VALIDATION (nov. 2024 → oct. 2025) par le CRPS
  moyen, la règle de score qui juge à la fois la justesse et la largeur des
  intervalles ; test une seule fois (nov. 2025 → oct. 2026).
Métriques : couverture des intervalles 90 % et 50 % (cibles 90 % et 50 %),
largeur, CRPS, erreur de la médiane, part des prix négatifs simulés et réels,
histogramme PIT (rang du prix réel parmi les trajectoires : plat si le modèle
est bien calibré).
Sorties : MLflow (expérience « simulation_prix »), reports/simulation_*.md,
reports/figures/simulation_*.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from elecfr import config, simulation  # noqa: E402

N_TRAJ = 1000
FENETRE_MOIS, VARIANTES = simulation.FENETRE_MOIS, simulation.VARIANTES

df = pd.read_parquet(config.PROCESSED_DIR / "prix_horaires.parquet")
out_fig = config.REPORTS_DIR / "figures"
out_fig.mkdir(parents=True, exist_ok=True)


def months(a, b):
    return pd.date_range(a, b, freq="MS")


def run_month(m0: pd.Timestamp, params: dict):
    """Même fonction que l'API (simulation.simulate_month) : aucune divergence possible."""
    model, sims, cal, reel = simulation.simulate_month(df, m0, params, n=N_TRAJ)
    return model, sims, cal.assign(prix=reel)


def run_period(a, b, params, keep=False):
    rows, pits, kept = [], [], []
    for m0 in months(a, b):
        model, sims, fut = run_month(m0, params)
        ev = simulation.evaluate(sims, fut["prix"].to_numpy())
        pits.append(ev.pop("pit"))
        rows.append({"mois": m0.strftime("%Y-%m"), "heures": int(fut["prix"].notna().sum()), **ev,
                     "phi": model.phi, "demi_vie_h": model.demi_vie_h, "p_saut": model.p_saut,
                     "phi_niveau": model.phi_n if model.deux_facteurs else np.nan})
        if keep:
            q = np.quantile(sims, simulation.QUANTILES, axis=0)
            kept.append(pd.DataFrame(q.T, index=fut.index, columns=[f"q{int(x * 100):02d}" for x in simulation.QUANTILES])
                        .assign(reel=fut["prix"]))
    t = pd.DataFrame(rows)
    w = t["heures"] / t["heures"].sum()
    agg = {k: float((t[k] * w).sum()) for k in ("couverture_90", "couverture_50", "largeur_90", "crps", "mae_mediane",
                                                "part_negatifs_reel", "part_negatifs_simule")}
    agg["couverture_moyenne_mensuelle_90"] = float(((t["moyenne_reelle"] >= t["moyenne_simulee_q05"]) &
                                                    (t["moyenne_reelle"] <= t["moyenne_simulee_q95"])).mean())
    return t, agg, np.concatenate(pits), (pd.concat(kept) if keep else None)


mlflow.set_tracking_uri(config.MLFLOW_URI)
mlflow.set_experiment("simulation_prix")
resultats, details, pits_test = [], {}, {}

for nom, params in VARIANTES.items():
    t, agg, pit, _ = run_period(config.VAL_START, config.VAL_END, params)
    details[(nom, "validation")] = t
    resultats.append({"variante": nom, "periode": "validation", **agg})
    with mlflow.start_run(run_name=f"{nom}_validation"):
        mlflow.set_tags({"variante": nom, "periode": "validation"})
        mlflow.log_params({**{k: str(v) for k, v in params.items()}, "fenetre_mois": FENETRE_MOIS, "n_traj": N_TRAJ})
        mlflow.log_metrics(agg)
    print(f"[validation] {nom:<22} CRPS {agg['crps']:6.2f} €/MWh ; couverture 90 % : {agg['couverture_90']:.1%} ; "
          f"50 % : {agg['couverture_50']:.1%} ; largeur 90 % : {agg['largeur_90']:.0f} €/MWh")

val = pd.DataFrame(resultats)
retenue = val.loc[val["crps"].idxmin(), "variante"]
print(f"\nVariante retenue sur la validation (CRPS le plus bas) : {retenue}")

for nom, params in VARIANTES.items():          # le test de toutes les variantes, pour la discussion
    t, agg, pit, kept = run_period(config.TEST_START, config.TEST_END, params, keep=(nom == retenue))
    details[(nom, "test")] = t
    pits_test[nom] = pit
    resultats.append({"variante": nom, "periode": "test", **agg})
    with mlflow.start_run(run_name=f"{nom}_test"):
        mlflow.set_tags({"variante": nom, "periode": "test", "selection": "retenue" if nom == retenue else "comparaison"})
        mlflow.log_params({**{k: str(v) for k, v in params.items()}, "fenetre_mois": FENETRE_MOIS, "n_traj": N_TRAJ})
        mlflow.log_metrics(agg)
    if kept is not None:
        quant_test = kept
    print(f"[test]       {nom:<22} CRPS {agg['crps']:6.2f} €/MWh ; couverture 90 % : {agg['couverture_90']:.1%} ; "
          f"50 % : {agg['couverture_50']:.1%} ; négatifs réels {agg['part_negatifs_reel']:.1%} / simulés {agg['part_negatifs_simule']:.1%}")

import json  # noqa: E402

(config.ROOT / "models").mkdir(exist_ok=True)
(config.ROOT / "models" / "simulation_prix.json").write_text(json.dumps(
    {"variante": retenue, "params": VARIANTES[retenue], "fenetre_mois": FENETRE_MOIS,
     "choisie_sur": f"validation {config.VAL_START} → {config.VAL_END} (CRPS)"}, indent=2, ensure_ascii=False),
    encoding="utf-8")
res = pd.DataFrame(resultats)
cols = ["periode", "variante", "crps", "couverture_90", "couverture_50", "largeur_90", "mae_mediane",
        "couverture_moyenne_mensuelle_90", "part_negatifs_reel", "part_negatifs_simule"]
md = res[cols].round(3).to_markdown(index=False)
(config.REPORTS_DIR / "simulation_resultats.md").write_text(md, encoding="utf-8")
detail = details[(retenue, "test")]
(config.REPORTS_DIR / "simulation_test_par_mois.md").write_text(
    detail[["mois", "crps", "couverture_90", "couverture_50", "moyenne_reelle", "moyenne_simulee_q05",
            "moyenne_simulee_q95", "phi", "demi_vie_h", "p_saut", "phi_niveau"]].round(3).to_markdown(index=False), encoding="utf-8")
print("\n" + md)
print(f"\nDétail mensuel du test ({retenue}) :\n" + detail[["mois", "crps", "couverture_90", "moyenne_reelle",
      "moyenne_simulee_q05", "moyenne_simulee_q95"]].round(2).to_string(index=False))

# --- Figures ----------------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "legend.frameon": False})

# Éventail du dernier mois de test
last = quant_test[quant_test.index >= pd.Timestamp(pd.Timestamp(config.TEST_END).strftime("%Y-%m-01"), tz=config.TZ)]
t = last.index.tz_convert(config.TZ).tz_localize(None)
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.fill_between(t, last["q05"], last["q95"], color="#cde2fb", lw=0, label="Intervalle 5 %–95 %")
ax.fill_between(t, last["q25"], last["q75"], color="#86b6ef", lw=0, label="Intervalle 25 %–75 %")
ax.plot(t, last["q50"], color="#1c5cab", lw=1.4, label="Médiane des 1 000 trajectoires")
ax.plot(t, last["reel"], color=INK, lw=0.9, label="Prix réel")
ax.axhline(0, color=INK2, lw=0.6)
inside = ((last["reel"] >= last["q05"]) & (last["reel"] <= last["q95"])).mean()
ax.set_title(f"Prix simulés et réels, {t[0]:%m/%Y} ({retenue}) : {inside:.0%} des heures dans l'intervalle 90 %", pad=24)
ax.set_ylabel("€/MWh")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8, borderaxespad=0.3)
fig.tight_layout()
fig.savefig(out_fig / "simulation_eventail.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# Au plus 3 variantes en couleur (lisibilité, daltonisme) : la base, la meilleure à un
# facteur et la meilleure à deux facteurs, selon la validation.
v = val.set_index("variante")["crps"]
un = v[[k for k in v.index if not k.startswith("deux")]]
deux = v[[k for k in v.index if k.startswith("deux")]]
montrees = list(dict.fromkeys(["ou", un.idxmin(), deux.idxmin() if len(deux) else retenue]))[:3]
COUL = dict(zip(montrees, ("#2a78d6", "#eb6834", "#1baf7a")))

# Calibration : couverture par mois (test) et PIT de la variante retenue
fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
ax = axes[0]
for nom, col in COUL.items():
    d = details[(nom, "test")]
    ax.plot(pd.to_datetime(d["mois"]), d["couverture_90"] * 100, color=col, marker="o", ms=3.5, mec=SURFACE,
            mew=0.8, label=nom)
ax.axhline(90, color=INK2, lw=0.8)
ax.annotate("cible 90 %", (pd.to_datetime(details[(retenue, "test")]["mois"].iloc[-1]), 90), xytext=(8, -3),
            textcoords="offset points", fontsize=8, color=INK2, annotation_clip=False)
ax.set_ylim(0, 100)
ax.set_ylabel("% des heures dans l'intervalle 90 %")
ax.set_title("Couverture de l'intervalle 90 %, mois par mois (test)", pad=22)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8, borderaxespad=0.2)
ax = axes[1]
ax.hist(pits_test[retenue], bins=20, range=(0, 1), color="#2a78d6", edgecolor=SURFACE, linewidth=1.5,
        weights=np.full(len(pits_test[retenue]), 20 / len(pits_test[retenue])))
ax.axhline(1, color=INK2, lw=0.8)
ax.set_title(f"Histogramme PIT ({retenue}, test) : plat si bien calibré", pad=22)
ax.set_xlabel("Rang du prix réel parmi les trajectoires (0 = sous toutes, 1 = au-dessus)")
ax.set_ylabel("Densité")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(out_fig / "simulation_calibration.png", dpi=160, bbox_inches="tight")
plt.close(fig)
# Moyenne mensuelle : intervalle 90 % simulé contre prix moyen réel (le défaut visé par les deux facteurs)
fig, ax = plt.subplots(figsize=(11, 4))
decal = {nom: k for nom, k in zip(montrees, (-8, 0, 8))}
for nom, col in COUL.items():
    d = details[(nom, "test")]
    x = pd.to_datetime(d["mois"]) + pd.to_timedelta(decal[nom], unit="D")
    lo, hi = d["moyenne_simulee_q05"], d["moyenne_simulee_q95"]
    ax.vlines(x, lo, hi, color=col, lw=4, alpha=0.85, label=f"{nom} : intervalle 90 %")
d0 = details[(retenue, "test")]
ax.plot(pd.to_datetime(d0["mois"]), d0["moyenne_reelle"], "o", color=INK, ms=6, mec=SURFACE, mew=1, label="Prix moyen réel")
ax.set_ylabel("€/MWh")
ax.set_title("Prix moyen du mois : intervalle 90 % simulé et réel (test)", pad=24)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8, borderaxespad=0.3)
fig.tight_layout()
fig.savefig(out_fig / "simulation_moyenne_mensuelle.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Graphiques : {out_fig / 'simulation_eventail.png'}, {out_fig / 'simulation_calibration.png'}, "
      f"{out_fig / 'simulation_moyenne_mensuelle.png'}")
