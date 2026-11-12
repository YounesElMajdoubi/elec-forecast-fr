# Prévision de la consommation et simulation des prix de l'électricité en France

Prévoir la consommation française du lendemain heure par heure, comparer honnêtement plusieurs familles de modèles, puis simuler des trajectoires de prix day-ahead.

**État :** volet 1 (prévision J+1), volet 2 (simulation des prix, prix ENTSO-E depuis le 12/11/2026) et industrialisation (MLflow, API, Docker, tests) terminés. La prévision en direct dans l'API reste à faire.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -e ".[dev,api]"
pip install -e ".[lstm]"                               # PyTorch, pour le LSTM (≈ 130 Mo sous Windows)
# Si Windows bloque torch.dll (« Une stratégie de contrôle d'application a bloqué ce fichier ») :
#   pip install "torch==2.10.*"     (version plus ancienne, voir DECISIONS.md D23)
cp .env.example .env                                    # puis colle ton jeton ENTSO-E
cp sources.example.toml sources.toml                    # puis remplis le nom de l'espace de travail
az login --allow-no-subscriptions                       # même compte que celui qui ouvre Fabric
pytest                                                  # tests sur données factices
python scripts/00_inspect_fabric.py                     # vérifie l'accès et les colonnes du silver
```

La consommation vient de la table `export_ml_eco2mix` du Lakehouse `lh_eco2mix` (projet A), lue dans OneLake au format Delta. `--allow-no-subscriptions` est nécessaire si ton compte Fabric n'a pas d'abonnement Azure. Sans accès au Lakehouse, `--source api` télécharge la consommation depuis ODRÉ.

### Table d'export côté projet A

Le silver est écrit par un `MERGE`, qui produit des vecteurs de suppression que la bibliothèque `deltalake` ne sait pas lire. Une dernière cellule du notebook `02_silver_qualite` recrée donc une copie lisible à chaque exécution (voir DECISIONS.md, D13) :

```python
spark.sql("""
CREATE OR REPLACE TABLE export_ml_eco2mix
TBLPROPERTIES ('delta.enableDeletionVectors' = 'false')
AS SELECT ts_utc, pas_min, nature, priorite_nature,
          consommation, prevision_j, prevision_j1, taux_co2,
          nucleaire, eolien, solaire, hydraulique, gaz, charbon, fioul, bioenergies,
          ech_physiques, pompage, stockage_batterie, destockage_batterie
FROM silver_eco2mix
""")
```

Vérification : `DESCRIBE DETAIL export_ml_eco2mix` doit indiquer `minReaderVersion = 1`.

### Compléter l'historique du Lakehouse

Le projet A charge les données depuis le 01/01/2024 ; le découpage du projet C a besoin des données depuis le 01/10/2021. Dans Fabric, sans toucher au code du projet A :

1. Exécuter `01_bronze_ingestion` avec `dataset = "eco2mix-national-cons-def"` et `date_debut = "2021-10-01"`.
2. Exécuter `02_silver_qualite` avec `fenetre_jours = 1870` le 11/11/2026 (ajouter 1 par jour de retard). Le filtre compare des instants (`current_timestamp() - INTERVAL n DAYS`), pas des dates : deux jours de marge évitent de perdre les premiers relevés du 01/10/2021, datés du 30/09 à 22 h UTC.
3. Remettre les paramètres par défaut : `dataset = "eco2mix-national-tr"`, `jours_glissants = 3`, `date_debut = ""` dans le notebook 01 ; `fenetre_jours = 5` dans le notebook 02.
4. Relancer `python scripts/00_inspect_fabric.py` : la ligne « Couverture » doit afficher OK.

Le `MERGE` sur `ts_utc` insère les nouvelles lignes sans toucher aux existantes.

**Fait le 11/11/2026.** Silver du 01/10/2021 (heure de Paris) au 11/11/2026, aucune valeur vide pour `consommation` ni `prevision_j1`. Les relevés absents correspondent exactement aux défauts connus de la source : une heure à chaque passage à l'heure d'hiver et un relevé le 01/01/2026.

### Si la lecture depuis ton ordinateur échoue

Exécute cette cellule dans un notebook Fabric attaché à `lh_eco2mix` et colle la sortie :

```python
df = spark.table("silver_eco2mix")
df.printSchema()
df.selectExpr("min(ts_utc)", "max(ts_utc)", "count(*)").show()
```

## Exécution

```bash
python scripts/00_inspect_fabric.py      # une fois : schéma et couverture du silver
python scripts/01_download.py            # options : --source api, --sans-prix
python scripts/02_build_dataset.py       # jeu horaire + rapport de qualité
python scripts/03_evaluate_baselines.py  # références, suivies dans MLflow
python scripts/04_train_models.py        # régression, LightGBM, LSTM (réglage sur validation, puis test)
                                         # un seul modèle : --modeles lstm
python scripts/05_compare.py             # tableau comparatif relu depuis MLflow
python scripts/06_diagnostics.py         # biais, erreurs par saison / mois / heure
python scripts/07_ensemble.py            # combinaison des modèles (poids choisis sur la validation)
python scripts/08_figures.py             # graphiques du README (reports/figures/)
python scripts/09_train_final.py         # modèle servi par l'API (models/)
uvicorn elecfr.api:app --reload          # API : http://127.0.0.1:8000/docs

# Volet 2 : prix (fichier Ember dans data/raw/ember/, voir plus bas)
python scripts/10_inspect_ember.py       # vérifie le fichier de prix
python scripts/11_build_prices.py        # jeu de prix horaire (--source entsoe avec le jeton)
python scripts/12_explore_prices.py      # analyse exploratoire + diagnostics statsmodels
python scripts/13_simulate_prices.py     # 1 000 trajectoires par mois, validation, MLflow

# Passage aux prix ENTSO-E (jeton dans .env)
python scripts/14_compare_prices.py      # télécharge ENTSO-E, compare à Ember heure par heure
python scripts/11_build_prices.py --source entsoe   # puis 12 et 13 à nouveau
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

## Découpage

| Période | Dates | Durée |
|---|---|---|
| Entraînement | 2021-11-01 → 2024-10-31 | 3 ans |
| Validation | 2024-11-01 → 2025-10-31 | 12 mois |
| Test (walk-forward mensuel) | 2025-11-01 → 2026-10-31 | 12 mois |

## Résultats

Chiffres issus des exécutions réelles du 12/11/2026 (walk-forward mensuel), relus depuis MLflow par `05_compare.py`.

| Modèle | MAE validation (MW) | MAE test (MW) | MAPE test (%) | Test : écart vs RTE recalée |
|---|---|---|---|---|
| Naïf saisonnier (J-7) | 3 384 | 3 499 | 6,54 | +310 % |
| Prévision J-1 RTE, publiée | 1 308 | 1 267 | 2,49 | +48 % |
| **Prévision J-1 RTE, recalée** | **914** | **853** | **1,68** | 0 |
| Régression linéaire (Ridge, alpha = 100) | 1 300 | 1 329 | 2,67 | +56 % |
| LightGBM (1 000 arbres, 31 feuilles) | 1 204 | 1 185 | 2,28 | +39 % |
| LSTM encodeur-décodeur (32 neurones) | 1 089 | 1 091 | 2,09 | +28 % |
| **Combinaison (0,1 régression + 0,3 LightGBM + 0,6 LSTM)** | **1 015** | **1 024** | **1,96** | +20 % |

« RTE recalée » = prévision publiée moins son biais moyen par heure sur les 28 derniers jours connus la veille (aucune information du futur).

### 1. Le LSTM est le meilleur des modèles appris

Il fait 8 % de mieux que LightGBM sur le test (1 091 contre 1 185 MW), avec un score stable entre validation et test (1 089 et 1 091 MW). Ce n'est pas LightGBM qui bat le LSTM, mais l'inverse, et le diagnostic dit pourquoi.

- **Ce n'est pas d'abord un effet d'hiver.** Le LSTM gagne autant au printemps (test : 1 000 contre 1 148 MW, −13 %) et en automne (1 038 contre 1 189, −13 %) qu'en hiver (1 459 contre 1 600, −9 %). L'hypothèse « les arbres n'extrapolent pas au-delà des températures vues » n'explique donc pas l'essentiel.
- **Il gagne quand la consommation évolue de façon continue.** C'est le cas en janvier (test : 1 584 contre 2 093 MW, −24 % ; validation : −27 %), lors des vagues de froid et des intersaisons. Il lit les 168 dernières heures d'affilée, alors que LightGBM ne voit que trois valeurs isolées (J-2, J-7, J-14 à la même heure) : il suit mieux le niveau récent.
- **Il perd aux ruptures de calendrier.** En décembre (fêtes) et en août (congés d'été), il fait moins bien que LightGBM, et il surestime : biais de +1 333 MW en décembre 2024, +789 MW en décembre 2025, +437 MW en août 2025. La semaine qu'il vient de lire était une semaine normale ; il en prolonge le niveau.
- **Il est nettement meilleur la nuit** (environ 800 à 900 MW de 0 h à 5 h sur le test, contre environ 1 100 MW pour LightGBM). Il est comparable en journée.
- **En été, LightGBM garde un léger avantage** : 811 contre 876 MW sur le test.

### 2. LightGBM bat la prévision publiée de RTE, mais pas RTE

Face à la prévision publiée, LightGBM et le LSTM semblent gagner. Cet avantage tient à un **décalage de niveau**.

- **La prévision publiée vise la consommation temps réel.** Son biais est de −1 152 MW face à la vérité consolidée (jusqu'au 31/08/2026), mais seulement de +135 MW face à la vérité temps réel (sept.-oct. 2026). Sur cette dernière période, RTE fait 579 MW de MAE.
- **Recalée avec la seule information de la veille, RTE redevient nettement la meilleure** : 853 MW sur le test, devant tous nos modèles, dans chaque saison.
- **Le recalage a la limite attendue au changement de vérité.** En septembre 2026, le biais appris sur août (consolidé) est appliqué à une vérité temps réel : RTE recalée monte à 1 125 MW, contre 564 MW pour la version publiée. En octobre, elle retombe à 589 MW.
- **Le biais de RTE dépend de l'heure** : près de −1 900 MW entre 13 h et 15 h, −1 250 MW à minuit et à 20 h, presque nul de 6 h à 7 h. La production solaire ne peut pas l'expliquer seule ; la cause précise reste ouverte.

### 3. La combinaison fait mieux que chaque modèle

Poids choisis sur la validation (grille au pas de 0,1) : 0,1 régression, 0,3 LightGBM, 0,6 LSTM. Elle obtient 1 015 MW sur la validation et 1 024 MW sur le test, soit 6 % de mieux que le LSTM seul. Le gain se maintient de la validation au test : les poids ne sont pas sur-ajustés. Le LSTM porte l'essentiel du poids. LightGBM corrige ses écarts aux ruptures de calendrier ; la régression, avec un petit poids, apporte son effet linéaire du chauffage.

### 4. Ordre sur le test

RTE recalée (853) < **combinaison (1 024)** < LSTM (1 091) < LightGBM (1 185) < RTE publiée (1 267) < régression linéaire (1 329) < naïf (3 499).

**Pistes d'amélioration.** Des indicateurs de fêtes et de congés plus fins (semaine de Noël, ponts d'août) pour le LSTM ; un LSTM de 16 neurones (le réglage retenu est au bord de la grille).

Heures exclues, sur chaque période : l'heure absente de la source au passage à l'heure d'hiver, et, pour le naïf, l'heure dont la référence J-7 tombe sur 2 h le jour du passage à l'heure d'été (heure qui n'existe pas).

Qualité des entrées : la prévision de température à 8 villes s'écarte de la température observée de 1,31 °C en moyenne (95e centile : 3,38 °C). C'est le prix de la règle « aucune fuite ».

## Graphiques

Produits par `08_figures.py` à partir des prévisions du test.

![Réel et prévisions sur une semaine d'hiver et une semaine d'été](reports/figures/prevision_semaines.png)

Sur la semaine d'hiver du 12 au 18 janvier 2026, LightGBM surestime nettement en milieu de semaine (MAE 2 837 MW). Le LSTM suit la courbe de près (991 MW) et fait même mieux que RTE recalée (1 515 MW). En été (15 au 21 juin 2026), les trois sont proches : RTE recalée 690 MW, LSTM 719 MW, LightGBM 746 MW.

![MAE par mois sur le test](reports/figures/mae_par_mois.png)

RTE recalée est la plus régulière, sauf en septembre 2026, quand la vérité passe du consolidé au temps réel (voir plus haut). LightGBM décroche en novembre et en janvier ; le LSTM est plus stable en hiver.

![Biais de la prévision RTE selon l'heure](reports/figures/biais_rte_par_heure.png)

La prévision publiée est jusqu'à 1 900 MW sous la consommation consolidée en milieu de journée, et autour de 1 250 MW la nuit. Recalée sur les 28 derniers jours connus, son biais reste sous 250 MW à toute heure.

## API

```bash
python scripts/09_train_final.py         # une fois : entraîne et enregistre le modèle servi
uvicorn elecfr.api:app --reload          # documentation interactive : http://127.0.0.1:8000/docs
```

| Route | Rôle |
|---|---|
| `GET /health` | état, modèle chargé, période des données |
| `GET /forecast?date=AAAA-MM-JJ` | prévision horaire du jour (heure de Paris), avec la valeur réelle et la prévision RTE quand elles existent |
| `GET /simulate?month=AAAA-MM` | 1 000 trajectoires horaires du prix day-ahead du mois : quantiles 5, 25, 50, 75 et 95 % par heure, prix moyen du mois, et couverture si les prix réels sont connus |

- **Modèle servi : LightGBM.** Le LSTM est 8 % plus précis, mais imposerait PyTorch dans l'image (voir DECISIONS.md, D25).
- **Mode « rejeu ».** L'API prévoit tout jour présent dans le jeu de données, avec les variables connues la veille. Pour un jour vu à l'entraînement, la réponse le signale (`dans_la_periode_d_entrainement`) : l'erreur affichée y est optimiste.
- **Prévoir demain en direct** demanderait de construire la ligne de demain (prévision météo du jour, consommation jusqu'à J-2 depuis le Lakehouse). C'est l'étape suivante.

## Docker

```bash
docker build -t elecfr-api .
docker run --rm -p 8000:8000 -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" elecfr-api
```

L'image ne contient que les dépendances de l'API (pas MLflow, deltalake ni PyTorch) ; le jeu de données et le modèle sont montés au lancement. GitHub Actions lance les tests (sauf ceux du LSTM, qui demandent PyTorch) puis construit l'image à chaque push.

## Volet 2 : simulation des prix day-ahead

**Données.** Prix day-ahead de la France publiés par l'[ENTSO-E Transparency Platform](https://transparency.entsoe.eu/), ramenés à l'heure (moyenne des quarts d'heure depuis le 01/10/2025), du 01/10/2021 au jour du téléchargement. Les premiers résultats utilisaient la copie publiée par [Ember](https://ember-energy.org/data/european-wholesale-electricity-price-data) (CC BY 4.0). `14_compare_prices.py` compare les deux sources heure par heure : sur 44 779 heures communes, 99,90 % sont identiques à 0,01 €/MWh près (écart absolu moyen 0,023 €/MWh). Les 43 heures qui diffèrent et les 5 heures absentes d'Ember sont surtout des défauts de la copie (premières heures de mois manquantes, une valeur répétée cinq heures le 24/07/2024). ENTSO-E compte aussi 99 heures négatives de plus : des moyennes de quarts d'heure à peine sous zéro, qu'Ember arrondit à 0 ; voir DECISIONS.md, D39. `--source ember` reste disponible.

**Modèle** (`src/elecfr/simulation.py`) : prix = partie déterministe + résidu.
- **Partie déterministe** (moindres carrés) : profil de 24 heures pour chacun des 3 types de jour (semaine, samedi, dimanche/férié), saisonnalité annuelle (2 harmoniques), tendance.
- **Résidu à retour à la moyenne** : AR(1) horaire, x(t) = φ·x(t−1) + ε(t), version discrète d'un processus d'Ornstein-Uhlenbeck (demi-vie ln 2 / −ln φ). φ est contre-vérifié avec statsmodels (`AutoReg`), et la stationnarité du résidu par un test ADF.
- **Sauts** : les chocs supérieurs à 4 écarts-types robustes sont modélisés à part (probabilité horaire et tailles observées, positives et négatives). Le reste des chocs est gaussien.
- **Variante « ancrage »** : niveau recalé sur les 28 derniers jours connus.
- **Modèle à deux facteurs** : le résidu est séparé en un niveau journalier lent (AR(1) d'un jour sur l'autre, chocs tirés parmi ceux observés) et un écart horaire rapide (AR(1) horaire avec sauts). Il vise l'incertitude sur le niveau du mois (voir les résultats ci-dessous).

**Protocole.** Pour chaque mois, le modèle est estimé sur les 24 mois précédents, puis 1 000 trajectoires horaires du mois sont simulées. Les cinq variantes (base, sauts, sauts + ancrage, deux facteurs avec et sans ancrage) sont comparées sur la validation (nov. 2024 → oct. 2025) par le CRPS, une règle de score qui juge à la fois la justesse et la largeur des intervalles. La variante retenue est évaluée une seule fois sur le test (nov. 2025 → oct. 2026).

**Validation face aux prix réels.**
- Couverture des intervalles à 90 % et à 50 % (cibles : 90 % et 50 %).
- Largeur des intervalles.
- CRPS.
- Erreur de la médiane.
- Part des prix négatifs simulés et réels.
- Couverture du prix moyen mensuel.
- Histogramme PIT : plat si le modèle est bien calibré.

### Ce que montrent les données (exploration, 2024 → oct. 2026)

- **La « courbe en canard ».** Les prix sont au plus bas vers 13–14 h, quand le solaire produit le plus, et au plus haut vers 19–22 h. L'écart entre les deux atteint 70 à 81 €/MWh selon le type de jour.
- **Des prix négatifs de plus en plus fréquents.** 352 heures en 2024, 513 en 2025, 568 en 2026 jusqu'au 12 novembre. Ils touchent 5,8 % des heures de janvier 2024 à octobre 2026, et 23,6 % des jours à 14 h. Ils surviennent quand l'éolien et le solaire couvrent 32 % de la consommation en moyenne, contre 18 % le reste du temps. Depuis le passage au quart d'heure, une partie d'entre eux sont des moyennes à peine sous zéro (entre −0,01 et 0 €/MWh).
- **Les renouvelables pèsent plus que la consommation.** Corrélation de rang du prix avec la part éolien + solaire : −0,58 ; avec la consommation : +0,29.
- **Le niveau change d'une année à l'autre.** Moyenne annuelle de 276 €/MWh pendant la crise de 2022, 58 en 2024, 61 en 2025, 96,5 en 2026 (jusqu'au 12 novembre). La moyenne mensuelle grimpe de 66 €/MWh en juin 2026 à 166 €/MWh en octobre 2026.
- **Le résidu revient bien vers sa moyenne, mais lentement**, avec une mémoire longue (diagnostics statsmodels, 24 mois avant la validation) :
  - le test ADF confirme la stationnarité (p < 0,0001) ;
  - φ = 0,973, soit une demi-vie de 25 h ;
  - l'autocorrélation vaut 0,84 à 24 h, contre 0,52 attendus pour un AR(1) pur, et 0,57 à une semaine.

### Résultats de la simulation (exécution réelle du 12/11/2026, prix ENTSO-E)

| Variante | CRPS validation | Couv. 90 % validation | CRPS test | Couv. 90 % test | Couv. 50 % test | Négatifs simulés / réels (test) | Moyenne mensuelle dans l'intervalle 90 % (test) |
|---|---|---|---|---|---|---|---|
| Base (AR(1)) | 26,6 | 81,2 % | 37,3 | 62,2 % | 33,8 % | 11,6 % / 6,5 % | 1 mois sur 12 |
| + sauts | 26,2 | 81,3 % | 37,2 | 61,2 % | 32,1 % | 10,6 % / 6,5 % | 1 mois sur 12 |
| **+ sauts + ancrage (retenue)** | **21,5** | **88,9 %** | **30,7** | **70,8 %** | **30,5 %** | **8,3 % / 6,5 %** | **2 mois sur 12** |
| Deux facteurs | 24,7 | 83,5 % | 36,7 | 62,5 % | 32,6 % | 10,8 % / 6,5 % | 3 mois sur 12 |
| Deux facteurs + ancrage | 24,7 | 83,5 % | 36,7 | 62,5 % | 32,6 % | 10,8 % / 6,5 % | 3 mois sur 12 |

CRPS en €/MWh (plus petit = meilleur). Cibles de couverture : 90 % et 50 %. Avec les prix d'Ember, les scores étaient presque identiques (variante retenue : CRPS 21,45 sur la validation et 30,67 sur le test) ; seule la part de négatifs réels change (5,4 % au lieu de 6,5 %), pour la raison donnée plus haut.

**Lecture :**
- **L'ancrage est l'amélioration décisive** : le CRPS baisse d'environ 18 %, sur la validation comme sur le test, et le classement des variantes est le même sur les deux périodes. Les sauts n'apportent presque rien (−1,5 % sur la validation).
- **Sur la validation, la variante retenue est presque bien calibrée** (89 % de couverture à 90 %). **Sur le test, elle ne l'est plus** (70,8 %). Le test contient une forte hausse continue des prix, de juin à octobre 2026, qu'aucune information passée ne laissait prévoir.
- **Le défaut principal est l'incertitude sur le niveau du mois.** Le prix moyen réel ne tombe dans l'intervalle 90 % simulé que 2 mois sur 12. Dans un AR(1) de demi-vie 25 h, les écarts se compensent en quelques jours, si bien que la moyenne d'un mois simulé varie très peu. Les vrais prix, eux, restent durablement hauts ou bas (autocorrélation de 0,57 à une semaine), car leur niveau dépend du gaz, du CO2 et de la disponibilité nucléaire, absents du modèle.
- **L'ancrage sur 28 jours réagit parfois trop fort.** En février 2026, il prolonge un mois de janvier cher (101 €/MWh) alors que les prix retombent (46 €/MWh réels, contre 83 à 104 simulés). En mars et en mai 2026, il part de mois bas et sous-estime nettement.
- **Les prix négatifs sont trop fréquents dans les simulations** : 8,3 % contre 6,5 %. Un bruit symétrique autour d'un creux de mi-journée déjà bas passe trop souvent sous zéro.
- **Quand le niveau est stable, le modèle fonctionne** : en novembre et décembre 2025, la couverture atteint 89–90 %.

**Le modèle à deux facteurs** (niveau journalier persistant + écart horaire) **corrige le défaut visé, mais le paie à l'échelle horaire** :
- **Il améliore nettement l'incertitude sur le niveau du mois.** Le prix moyen réel tombe dans l'intervalle 90 % simulé 9 mois sur 12 sur la validation, contre 5 pour la variante retenue, et 3 mois sur 12 sur le test, contre 2.
- **Il dégrade la précision heure par heure** : CRPS de 24,7 contre 21,5 sur la validation, et de 36,7 contre 30,7 sur le test. La validation retient donc la variante sauts + ancrage.
- **Hypothèse pour cet écart.** Le niveau de départ est celui du **dernier jour** connu : un seul jour, marqué par son vent et sa météo. Comme ce niveau persiste presque tout le mois, son bruit s'y propage. L'ancrage, lui, part d'une moyenne sur 28 jours, plus stable. Les deux variantes à deux facteurs donnent presque les mêmes scores, avec et sans ancrage : le niveau journalier est donc très persistant, et sa valeur de départ pèse plus que la cible de long terme.
- **Piste naturelle** : un niveau de départ lissé sur plusieurs jours, ou un troisième facteur qui sépare la tendance de fond (gaz, CO2) du bruit météo d'un jour. Les variantes ont déjà été comparées plusieurs fois sur le test : toute nouvelle variante devra être choisie sur la validation, puis jugée sur une période encore jamais vue (à partir de novembre 2026), pour ne pas ajuster le modèle au test.

**Pistes, par ordre d'impact attendu :**
1. Un niveau de départ lissé, ou un troisième facteur (tendance de fond + bruit journalier), pour garder le gain du modèle à deux facteurs sur la moyenne du mois sans perdre en précision horaire.
2. Un AR à décalages 1 h et 24 h, pour la mémoire journalière.
3. Une volatilité qui dépend de l'heure et du niveau.
4. Le prix du gaz comme variable explicative, si une source gratuite est disponible.

## Choix méthodologiques

Voir [DECISIONS.md](DECISIONS.md).

## Sources de données

- RTE éCO2mix : Lakehouse Fabric du projet A, ou ODRÉ (`eco2mix-national-cons-def`, `eco2mix-national-tr`)
- Open-Meteo : API Previous Runs (prévisions archivées) et Archive (observations, analyse seulement)
- ENTSO-E Transparency Platform : prix day-ahead, zone France
- Calendrier scolaire : data.education.gouv.fr ; jours fériés : bibliothèque `holidays`
- Prix day-ahead (volet 2) : Ember, *European Wholesale Electricity Price Data* (CC BY 4.0), d'après ENTSO-E
