"""Tests du modèle de prix sur des séries FACTICES à structure connue."""
import numpy as np
import pandas as pd
import pytest

from elecfr import simulation


def fake_prices(phi=0.9, sigma=8.0, p_jump=0.004, start="2023-01-01", end="2025-03-01", seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start, tz="Europe/Paris").tz_convert("UTC"),
                        pd.Timestamp(end, tz="Europe/Paris").tz_convert("UTC"), freq="h", inclusive="left")
    cal = simulation.calendrier(idx)
    det = 60 + 15 * np.sin(2 * np.pi * (cal["heure"].to_numpy() - 5) / 24) - 8 * (cal["jour_semaine"].to_numpy() >= 5)
    x, xs = 0.0, np.empty(len(idx))
    for t in range(len(idx)):
        e = rng.normal(0, sigma) + (rng.choice([-150.0, 200.0]) if rng.random() < p_jump else 0.0)
        x = phi * x + e
        xs[t] = x
    return cal.assign(prix=det + xs)


def test_fit_recovers_mean_reversion_and_jumps():
    df = fake_prices()
    m = simulation.fit(df["prix"], df, sauts=True)
    assert abs(m.phi - 0.9) < 0.02
    assert abs(m.sigma - 8.0) < 1.0              # chocs gaussiens hors sauts
    assert 0.002 < m.p_saut < 0.008              # fréquence des sauts retrouvée
    assert abs(m.demi_vie_h - np.log(2) / -np.log(0.9)) < 0.5


def test_crps_matches_closed_form_for_a_normal():
    # CRPS d'une loi normale N(0, 1) en y = 0 : 2φ(0) - 1/√π ≈ 0.2337
    sims = np.random.default_rng(0).normal(0, 1, (20000, 1))
    assert abs(simulation.crps(sims, np.array([0.0]))[0] - (2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))) < 0.01


def test_calibrated_simulation_covers_90_percent():
    rng = np.random.default_rng(3)
    sims, y = rng.normal(0, 1, (1000, 5000)), rng.normal(0, 1, 5000)
    ev = simulation.evaluate(sims, y)
    assert abs(ev["couverture_90"] - 0.90) < 0.02 and abs(ev["couverture_50"] - 0.50) < 0.03


def test_simulate_month_uses_only_prices_before_the_month():
    df = fake_prices(end="2025-04-01")
    a = simulation.simulate_month(df, "2025-03", {"sauts": True}, n=200)
    altered = df.copy()
    altered.loc[altered.index >= simulation.bornes_mois("2025-03")[0], "prix"] += 1000   # le « futur »
    b = simulation.simulate_month(altered, "2025-03", {"sauts": True}, n=200)
    np.testing.assert_allclose(a[1], b[1])                       # trajectoires identiques
    assert np.nanmean(b[3]) > np.nanmean(a[3]) + 900             # seule la vérité a changé


def test_month_hours_follow_daylight_saving():
    df = fake_prices(end="2025-04-01")
    _, sims, cal, _ = simulation.simulate_month(df, "2025-03", {"sauts": False}, n=50)
    assert sims.shape == (50, 31 * 24 - 1)                       # mars : une heure de moins


def test_month_without_previous_prices_is_refused():
    df = fake_prices(end="2025-03-01")
    with pytest.raises(ValueError, match="Prix inconnus"):
        simulation.simulate_month(df, "2025-06", {"sauts": True}, n=50)


def fake_two_factor(phi_n=0.9, sd_n=15.0, phi=0.8, sigma=5.0, start="2023-01-01", end="2025-04-01", seed=2):
    """Prix factices avec un VRAI niveau journalier persistant + un écart horaire rapide."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start, tz="Europe/Paris").tz_convert("UTC"),
                        pd.Timestamp(end, tz="Europe/Paris").tz_convert("UTC"), freq="h", inclusive="left")
    cal = simulation.calendrier(idx)
    det = 60 + 15 * np.sin(2 * np.pi * (cal["heure"].to_numpy() - 5) / 24)
    days = cal["date_local"].to_numpy()
    level, x, out = 0.0, 0.0, np.empty(len(idx))
    for t in range(len(idx)):
        if t == 0 or days[t] != days[t - 1]:
            level = phi_n * level + rng.normal(0, sd_n)
        x = phi * x + rng.normal(0, sigma)
        out[t] = det[t] + level + x
    return cal.assign(prix=out)


def test_two_factor_fit_recovers_daily_persistence():
    df = fake_two_factor()
    m = simulation.fit(df["prix"], df, sauts=False, deux_facteurs=True)
    assert abs(m.phi_n - 0.9) < 0.05                     # persistance du niveau journalier retrouvée
    # φ horaire : estimé APRÈS retrait de la moyenne de chaque jour, il ne mesure
    # que la persistance à l'intérieur d'une journée (biais vers le bas attendu).
    assert 0.6 < m.phi < 0.8
    assert m.demi_vie_niveau_j > 4


def test_two_factor_gives_realistic_uncertainty_on_the_monthly_mean():
    # Sur des données à niveau persistant, le modèle à un facteur sous-estime
    # fortement la dispersion de la moyenne du mois ; le modèle à deux facteurs non.
    df = fake_two_factor(end="2025-04-01")
    one = simulation.simulate_month(df, "2025-03", {"sauts": False}, n=400)[1].mean(axis=1).std()
    two = simulation.simulate_month(df, "2025-03", {"sauts": False, "deux_facteurs": True}, n=400)[1].mean(axis=1).std()
    # Écart-type théorique de la moyenne sur n = 31 jours d'un niveau AR(1) (φ = 0,9 ; σ = 15)
    # dont la valeur de départ est connue : σ/n · √Σ_i ((1 - φ^(n-i+1)) / (1 - φ))²
    phi, sd, n = 0.9, 15.0, 31
    theorie = sd / n * np.sqrt(sum(((1 - phi ** (n - i + 1)) / (1 - phi)) ** 2 for i in range(1, n + 1)))
    assert abs(two - theorie) / theorie < 0.2
    assert two > one


def test_two_factor_month_uses_only_past_prices():
    df = fake_two_factor(end="2025-04-01")
    p = {"sauts": True, "deux_facteurs": True, "ancrage_jours": 28}
    a = simulation.simulate_month(df, "2025-03", p, n=100)[1]
    altered = df.copy()
    altered.loc[altered.index >= simulation.bornes_mois("2025-03")[0], "prix"] -= 500
    np.testing.assert_allclose(a, simulation.simulate_month(altered, "2025-03", p, n=100)[1])
