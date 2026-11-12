# Journal de décisions

Chaque entrée : le choix, pourquoi, ce qu'on a écarté, et ce qui reste à vérifier.

## D1 — Instant d'émission de la prévision : J-1 à 10 h (Paris)
**Choix.** Toutes les variables d'une heure du jour J doivent être connues la veille à 10 h.
**Pourquoi.** C'est avant la fermeture du marché day-ahead (midi), donc une prévision réellement utilisable. Fixer cet instant rend la règle « aucune fuite » vérifiable : on peut tester chaque variable contre lui.
**Écarté.** « La veille » sans heure précise : trop flou pour savoir si la consommation de J-1 ou un run météo de 18 h est autorisé.

## D2 — Travail en UTC, variables en heure locale
**Choix.** L'index est en UTC ; heure, jour, jours fériés sont calculés en heure de Paris.
**Pourquoi.** En UTC chaque heure dure 60 min et aucune n'est dupliquée ou absente. Mais la consommation suit la vie des gens, donc l'heure affichée à Paris. Les jours de changement d'heure (23 ou 25 h) sont repérés par `changement_heure`.

## D3 — Agrégation horaire par moyenne
**Choix.** Moyenne des valeurs à 15/30 min ; `n_cons` garde le nombre de valeurs utilisées.
**Pourquoi.** Ce sont des puissances en MW, pas des énergies en MWh : on moyenne, on ne somme pas.

## D4 — Température connue la veille (Open-Meteo « Previous Runs », modèle GFS)
**Choix.** Pour chaque heure t, on prend la prévision émise 24 h avant (`previous_day1`) si elle était publiée avant J-1 10 h, sinon celle émise 48 h avant (`previous_day2`). Avec un délai de publication supposé de 6 h, `previous_day1` sert pour 0 h–4 h, `previous_day2` pour le reste de la journée.
**Pourquoi.** La température observée du jour J est inconnue la veille : l'utiliser gonflerait artificiellement les performances. Les prévisions archivées reproduisent ce qu'on aurait vraiment eu.
**Pourquoi GFS.** C'est un des rares modèles archivés depuis mars 2021 dans cette API ; la plupart ne le sont que depuis janvier 2024, ce qui ne couvrirait pas 3 ans d'entraînement. Utiliser le même modèle sur tout l'historique garantit que l'erreur de prévision a la même nature à l'entraînement et au test.
**À vérifier.** (a) La sémantique exacte de « 24 h avant » (alignement sur les runs de 0/6/12/18 UTC) ; le délai de 6 h est volontairement prudent. Réglable dans `config.py`. (b) La couverture réelle par année, imprimée par `02_build_dataset.py`.
**Mesure.** `temp_obs` (réanalyse ERA5) est conservée uniquement pour quantifier l'erreur de prévision météo ; elle est dans la liste `FORBIDDEN` et un test l'interdit comme variable.

## D5 — Température nationale = moyenne de 8 villes pondérée par la population
**Choix.** Paris, Lyon, Marseille, Lille, Toulouse, Bordeaux, Nantes, Strasbourg ; poids ≈ population des aires d'attraction (ordre de grandeur).
**Pourquoi.** La consommation suit la température là où vivent les gens. Ce n'est pas la « température France » de RTE (construite sur des stations), seulement une approximation raisonnable et documentée.

## D6 — Retards de consommation : J-2, J-7, J-14 uniquement
**Choix.** Même heure locale, 2, 7 et 14 jours avant. Pas de J-1.
**Pourquoi.** À J-1 10 h, la journée de J-1 n'est connue que jusqu'à 10 h : utiliser J-1 à 18 h pour prévoir J à 18 h serait une fuite. Les retards comparent des heures « murales » pour que 18 h reste 18 h quand l'heure change.
**Limite connue.** On utilise des valeurs consolidées du passé ; en réalité, à J-1, on n'aurait eu que les valeurs temps réel. **Révisé le 12/11/2026 :** l'écart entre les deux n'est pas faible. La prévision RTE, qui semble viser le temps réel, est ~950 MW sous les données consolidées (voir D20).

## D7 — Sources éCO2mix : consolidé prioritaire, temps réel ensuite
**Choix.** On télécharge les deux jeux ODRÉ et, pour chaque horodatage, on garde le consolidé s'il a une consommation.
**Pourquoi.** Le consolidé est plus fiable ; le temps réel couvre la période récente (test).

## D8 — Découpage chronologique 3 ans / 12 mois / 12 mois (révisé le 11/11/2026)
**Choix.** Entraînement 2021-11-01 → 2024-10-31, validation 2024-11-01 → 2025-10-31, test 2025-11-01 → 2026-10-31.
**Historique.** Première version : validation et test de 6 mois chacun, avec un test de mai à octobre 2026, donc sans hiver. Or l'hiver est la saison la plus difficile : la consommation y dépend le plus de la température (chauffage électrique). Test porté à 12 mois.
**Pourquoi la validation aussi.** Avec un test de 12 mois et une validation de 6 mois juste avant, la validation serait tombée de mai à octobre 2025 : on aurait réglé les hyperparamètres sans jamais voir d'hiver. La validation passe donc aussi à 12 mois, et tout recule de 6 mois. Retour en arrière possible en modifiant une ligne de `config.py`.
**Conséquence.** L'entraînement contient l'hiver 2022-2023 : crise énergétique, plan de sobriété, consommation nettement en baisse par rapport aux années précédentes. C'est une rupture de régime. Un modèle entraîné sur toute la fenêtre peut sous-estimer la consommation récente, ou l'inverse. À regarder dans l'analyse exploratoire ; la fenêtre glissante (`train_days`, D9) est une parade si besoin. Pour les prix (volet 2), les pics de 2022 compteront encore davantage.

## D9 — Walk-forward mensuel, entraînement jusqu'à B-2
**Choix.** Le test est découpé en mois. Au début de chaque mois B, le modèle est réentraîné sur toutes les journées ≤ B-2, puis prévoit le mois entier sans mise à jour. Fenêtre qui s'agrandit par défaut ; fenêtre glissante de longueur fixe possible (`train_days`).
**Pourquoi.** Imite un usage réel (réentraînement périodique) sans jamais utiliser une donnée inconnue à l'émission. Un test vérifie la règle B-2.
**Écarté.** Réentraînement quotidien : plus fidèle, mais coûteux pour le LSTM ; le mensuel est plus prudent, pas plus optimiste.

## D10 — Métriques et heures exclues
**Choix.** MAE (MW) et MAPE (%) globales, par heure locale, par saison. Les heures sans vérité ou sans prévision sont exclues et comptées (`n_exclues`).
**Pourquoi.** La MAE parle en unités physiques ; la MAPE permet de comparer été et hiver. Compter les exclusions évite de cacher des trous (par exemple l'heure qui n'existe pas en mars, pour le naïf saisonnier).

## D12 — Consommation lue dans le Lakehouse Fabric du projet A
**Choix.** La source éCO2mix par défaut est la table du Lakehouse (`ECO2MIX_SOURCE = "fabric"`). Elle est lue directement dans OneLake avec la bibliothèque `deltalake`, en s'authentifiant avec l'Azure CLI. Les noms de tables et de colonnes sont dans `sources.toml`, non commité. Météo, prix et calendrier restent téléchargés par API.
**Pourquoi.** Réutiliser le projet A montre une chaîne de bout en bout : ingestion, nettoyage, puis modélisation. Une table de Lakehouse étant au format Delta ouvert, pas besoin de Spark ni d'exécuter le code dans Fabric.
**Pourquoi silver plutôt que gold.** On veut le pas de 15/30 min et la colonne `prevision_j1`, puis refaire nous-mêmes l'agrégation horaire en UTC (D2, D3). Une table gold déjà horaire convient seulement si elle contient `prevision_j1` et si son agrégation suit ces deux règles.
**Garde-fous.** Les colonnes indispensables manquantes provoquent une erreur qui liste les colonnes vues. Les horodatages en double sont signalés. Des horodatages naïfs en heure de Paris sont convertis en UTC, avec arrêt en cas d'ambiguïté au changement d'heure. Le rapport de qualité (`02_build_dataset.py`) vérifie ensuite la couverture comme pour la source API.
**Repli.** `--source api` retélécharge depuis ODRÉ ; les deux sources produisent le même fichier `eco2mix_fusion.parquet`.
**Ce que le projet A apporte déjà (d'après son README).** Clé `ts_utc` en UTC, ce qui évite l'ambiguïté du changement d'heure. Le `MERGE` garde toujours la version la plus fiable (définitive > consolidée > temps réel). Les doublons de l'heure d'été sont déjà retirés.
**Ce qu'on en hérite.** Pas de 30 min jusqu'au 31/08/2026 puis de 15 min : la moyenne horaire (D3) s'en accommode, et `n_cons` vaut 2 ou 4 selon la période. Il manque une heure par an au passage à l'heure d'hiver (défaut de la source) et un relevé le 01/01/2026. Ces heures sont exclues de l'évaluation et comptées dans `n_exclues` (D10), et les retards qui les visent valent NaN.
**Historique trop court (constat du 11/11/2026).** Le Lakehouse commence le 01/01/2024, alors que le découpage (D8) demande les données depuis le 01/10/2021. Décision retenue : compléter l'historique dans le Lakehouse avec ses propres notebooks (README, « Compléter l'historique »), plutôt que de mélanger deux sources. Une seule source de vérité, et les contrôles qualité du projet A s'appliquent aussi aux années ajoutées.

## D13 — Lire une table d'export plutôt que le silver (12/11/2026)
**Constat.** Le `MERGE` du silver fait écrire à Fabric des vecteurs de suppression : au lieu de réécrire un fichier Parquet, Fabric note à côté quelles lignes ignorer. La bibliothèque `deltalake` ne sait pas les appliquer. Elle refuse donc de lire la table (`DeltaProtocolError: ... {'deletionVectors'}`) plutôt que de renvoyer d'anciennes versions des lignes en double, et c'est le bon comportement.
**Écarté : DuckDB.** Son extension Delta lit les vecteurs de suppression, mais Windows (contrôle des applications) bloque le chargement des extensions DuckDB sur le poste de travail. Désactiver cette protection affaiblirait la sécurité de toute la machine pour un seul projet.
**Écarté : retirer les vecteurs de suppression du silver.** Il faudrait modifier le protocole d'une table de production du projet A (`REORG ... PURGE`, puis `DROP FEATURE`) : opération délicate, pour un bénéfice limité au projet C.
**Choix.** Une dernière cellule de `02_silver_qualite` recrée `export_ml_eco2mix` à chaque exécution (`CREATE OR REPLACE TABLE ... TBLPROPERTIES ('delta.enableDeletionVectors' = 'false') AS SELECT ... FROM silver_eco2mix`). Comme la table est réécrite en entier et jamais modifiée ligne à ligne, Fabric n'a jamais besoin de vecteurs de suppression : `minReaderVersion` = 1, lisible par tout lecteur Delta.
**Pourquoi c'est défendable.** Le silver reste la source de vérité ; l'export est un contrat explicite entre les deux projets (colonnes figées, rafraîchi par le même pipeline de 7 h). Le coût est négligeable : environ 100 000 lignes réécrites par jour.

## D11 — Prix : passage au quart d'heure le 1er octobre 2025
**Choix.** Prix ramenés à l'heure par moyenne des quatre quarts d'heure ; résolution d'origine gardée.
**Pourquoi.** Le marché day-ahead européen est passé à des produits de 15 min (premier jour de livraison : 1er octobre 2025). Sans cette étape, la série changerait de nature au milieu de l'historique.

## D14 — Régression linéaire : Ridge et variables croisées
**Choix.** Ridge (régression linéaire pénalisée) sur environ 270 colonnes construites à la main :
- profil heure × type de jour (24 × 7 indicatrices) ;
- degrés-jours de chauffage et de climatisation × heure ;
- sin/cos de la saison × heure ;
- retards J-2, J-7, J-14 et indicateurs calendaires.

Un jour férié est traité comme un dimanche (`type_jour`). Les valeurs manquantes (retards aux changements d'heure, trous météo) sont remplacées par la médiane, puis toutes les colonnes sont standardisées.
**Pourquoi.** Un modèle linéaire n'additionne que des effets : sans croisements, il apprendrait un seul profil journalier et un seul effet du froid, identique à 4 h et à 19 h. La pénalité Ridge stabilise des coefficients très corrélés ; standardiser est nécessaire pour qu'elle traite toutes les colonnes de la même façon.
**Réglage.** alpha ∈ {1, 10, 100, 1 000, 10 000}. Les petites valeurs n'ont presque pas d'effet avec ~26 000 heures d'entraînement : on balaie plusieurs ordres de grandeur.

## D15 — LightGBM : variables brutes, perte absolue
**Choix.** Les variables de `FEATURES` plus `type_jour`, sans croisements : les arbres trouvent eux-mêmes les interactions. Perte `regression_l1`, la même que la métrique principale (MAE). Grille : 400 ou 1 000 arbres × 31 ou 63 feuilles, avec un taux d'apprentissage de 0,05. Fixés : au moins 50 heures par feuille, 80 % des lignes et des colonnes par arbre, graine 0.
**Pas d'arrêt précoce.** Il exigerait un jeu de contrôle découpé dans chaque fenêtre d'entraînement. Le nombre d'arbres est réglé comme les autres hyperparamètres, sur la validation.

## D16 — Protocole de réglage et de test
1. Chaque réglage de la grille est évalué en walk-forward sur la validation (nov. 2024 → oct. 2025).
2. Le réglage de plus faible MAE sur la validation est retenu, sans jamais regarder le test.
3. Ce seul réglage est évalué en walk-forward sur le test (nov. 2025 → oct. 2026).

L'entraînement part toujours du 01/11/2021 (`train_start`) et s'agrandit chaque mois. Pendant le test, il inclut donc la validation, comme le ferait un modèle mis en production.

## D17 — MLflow
**Base.** SQLite explicite (`mlflow.db` à la racine), pour que les scripts et `mlflow ui` lisent la même base.
**Contenu de chaque run** (un modèle, un réglage, une période) :
- les paramètres ;
- les métriques globales ;
- les tags `modele`, `periode`, `famille`, `selection` (candidat ou retenu) ;
- les métriques par heure et par saison ;
- les prévisions heure par heure ;
- pour le test, le dernier modèle entraîné, avec les coefficients ou les importances.

**Tableau comparatif.** Il est relu depuis MLflow (`05_compare.py`), jamais recalculé à côté.
**Format de sauvegarde.** MLflow 3 sauvegarde les modèles scikit-learn avec skops, qui refuse de recharger les types non déclarés : on déclare `numpy.dtype`. LightGBM est sauvegardé dans son format natif.

## D18 — Trou dans l'archive des prévisions météo (12/11/2026)
**Constat.** La température prévue manque pour 513 heures, presque toutes d'un seul bloc de fin décembre 2023 à janvier 2024 (47 h en décembre, 464 h en janvier, plus 2 h en octobre 2021, avant l'entraînement). La température observée existe pour toutes ces heures : c'est un trou de l'archive des prévisions GFS (Open-Meteo, Previous Runs), pas de la météo elle-même. Validation et test sont complets.
**Choix.** Les heures d'entraînement sans température prévue sont retirées de l'apprentissage (1,9 % des heures d'entraînement, environ trois semaines).
**Écarté : remplacer par la médiane.** La médiane des températures (≈ 13 °C) imputée en plein mois de janvier apprendrait au modèle qu'il faisait doux pendant trois semaines d'hiver : le pire endroit pour se tromper sur l'effet du chauffage.
**Écarté : boucher le trou avec la température observée.** On mélangerait, à l'entraînement seulement, des températures exactes et des prévisions imparfaites.
**Garde-fou conservé.** L'imputation par la médiane reste dans le pipeline Ridge pour l'inférence ; elle ne sert aujourd'hui à aucune heure de validation ni de test.

## D19 — Grilles élargies et diagnostic avant de conclure (12/11/2026)
**Constat de la première exécution réelle** (sans la correction D18) : LightGBM bat la prévision J-1 de RTE sur le test, 1 190 MW contre 1 267 MW. Les deux réglages retenus étaient au bord de leur grille : alpha = 1, la plus petite valeur testée pour Ridge ; 1 000 arbres, la plus grande pour LightGBM.
**Grilles élargies.** alpha ∈ {0,01 … 10 000} ; 400, 1 000 et 2 000 arbres. Le choix se fait toujours sur la validation seule : élargir une grille parce que l'optimum de validation est au bord est légitime, régler sur le test ne le serait pas.
**Diagnostic avant de conclure.** Un modèle sur données publiques qui bat RTE doit être expliqué avant d'être annoncé. `06_diagnostics.py` mesure :
- le biais moyen de chaque modèle ;
- l'erreur par saison, par mois et par heure ;
- l'erreur selon la source de la vérité (consolidé jusqu'au 31/08/2026, temps réel ensuite) ;
- la MAE de RTE corrigée de son biais mensuel, uniquement comme diagnostic. Si RTE corrigée repasse devant, l'avance du modèle tient au niveau moyen visé plutôt qu'à une meilleure forme de courbe.

**Bug corrigé.** La copie locale des prévisions de validation était écrasée par chaque essai de la grille. Les métriques de MLflow n'étaient pas touchées. Seul le réglage retenu est désormais copié localement.

## D20 — Le biais de RTE explique l'« avance » de LightGBM (12/11/2026)
**Constat** (`06_diagnostics.py`, exécution réelle) :
- **Un biais massif.** Prévision J-1 de RTE − consommation consolidée = −937 MW en moyenne sur le test (−985 MW sur la validation), avec 75 % des heures sous-estimées.
- **RTE reprend l'avantage une fois le biais retiré.** Corrigée de son biais mensuel (diagnostic seulement), sa MAE passe de 1 267 à 937 MW, contre 1 185 MW pour LightGBM.
- **Le biais dépend de la source de la vérité.** Sur sept.-oct. 2026, où la vérité est la consommation temps réel, RTE fait 579 MW contre 857 MW pour LightGBM.
- **Il se concentre en milieu de journée et au printemps.**

**Interprétation.** La prévision publiée vise vraisemblablement la consommation temps réel, dont le niveau est inférieur à celui des données consolidées, surtout en milieu de journée. Hypothèse à confirmer : production solaire diffuse estimée après coup. Nos modèles, entraînés sur les données consolidées, apprennent le bon niveau ; RTE non. L'écart de MAE mesure donc surtout ce décalage.
**Choix.** Ajout d'une troisième référence, « RTE J-1 recalée » : la prévision publiée moins son biais moyen par heure sur les 28 derniers jours connus à l'émission (≤ J-2, mis à jour à chaque bloc mensuel). Elle n'utilise que de l'information disponible la veille. C'est la référence juste pour dire si un modèle « bat RTE ».
**Limite.** La vérité change de nature au 01/09/2026 (consolidé, puis temps réel). Le recalage de septembre, appris sur août (consolidé), corrigera donc à tort une prévision qui n'en avait plus besoin. Les diagnostics séparent les deux régimes.
**Ce qu'on n'écrit pas.** « LightGBM bat RTE. » La formulation juste est : « LightGBM bat la prévision publiée de RTE jugée sur les données consolidées, grâce au niveau ; à niveau égal, RTE reste meilleure. »

## D21 — Confirmation par la référence recalée (12/11/2026)
- **RTE recalée** : 853 MW sur le test (1,68 %) et 914 MW sur la validation. C'est la meilleure référence, loin devant LightGBM (1 185 MW).
- **Biais de RTE publiée selon la vérité** : −1 152 MW face au consolidé, +135 MW face au temps réel. Cela confirme D20 : la prévision publiée vise le temps réel.
- **La limite prévue en D20 s'observe.** En septembre 2026, RTE recalée tombe à 1 125 MW contre 564 MW pour la version publiée, puis revient à 589 MW en octobre.
- **L'hypothèse solaire seule est insuffisante.** Le biais horaire est aussi fort à minuit et à 20 h (≈ −1 250 MW) qu'en milieu de journée (≈ −1 900 MW). Cause ouverte.
- **Point faible de LightGBM : l'hiver** (≈ 2 000 MW en janvier). Hypothèse : les arbres n'extrapolent pas au-delà des températures vues. Piste à tester avec le LSTM et une combinaison des modèles.

## D22 — LSTM encodeur-décodeur
**Architecture.** Deux LSTM :
- un **encodeur** lit la consommation horaire de J-8 à J-2 (168 h), avec l'heure (sin/cos) et un indicateur de jour chômé ;
- un **décodeur** part de cet état et avance sur les 24 heures du jour J. À chaque heure, il reçoit ce qu'on en sait la veille : température prévue, degrés-jours, calendrier, type de jour, retards J-2/J-7/J-14 de la même heure.

Une couche linéaire produit la consommation de chaque heure.
**Pourquoi cette forme.** Elle sépare ce que le réseau doit résumer (la dynamique récente) de ce qu'on connaît pour demain (météo, calendrier). C'est la forme standard pour prévoir une séquence entière d'un coup.
**Sans fuite.** L'historique s'arrête à J-2. Les trous (heure absente en mars) sont comblés par la dernière valeur connue, jamais par une valeur future. Deux tests le vérifient :
- modifier J et J-1 ne change pas la prévision de J ;
- modifier J-2 la change.

**Format des jours.** Chaque jour est ramené à 24 heures murales : l'heure doublée d'octobre est moyennée, l'heure absente de mars est comblée. Ce format fixe est indispensable pour un réseau.
**Entraînement.** Perte absolue (comme la MAE), Adam (lr = 0,001), lots de 32 jours, gradients plafonnés à 1, graine fixe. Variables standardisées sur la fenêtre d'entraînement.
**Nombre d'époques par arrêt précoce, en deux phases.**
1. On met de côté les 56 derniers jours de la fenêtre et on retient l'époque où l'erreur sur ces jours est la plus basse (patience 8, maximum 80).
2. On réentraîne sur toute la fenêtre pendant ce nombre d'époques, pour ne pas perdre les 8 semaines les plus récentes.

Les époques retenues à chaque réentraînement sont enregistrées dans MLflow.
**Réglage.** Taille de l'état caché ∈ {32, 64}, choisie sur la validation comme pour les autres modèles. Les jours sans température prévue sont exclus de l'entraînement (D18).
**Limites assumées.**
- Avec un échantillon par jour, il n'y a qu'environ 1 100 à 1 800 exemples d'entraînement : c'est peu pour un réseau de neurones, et un avantage structurel pour LightGBM.
- Pas de recherche étendue d'hyperparamètres (couches, dropout, taux d'apprentissage), faute de temps de calcul. Le walk-forward impose 12 réentraînements par réglage.
- PyTorch est optionnel (`pip install -e ".[lstm]"`) : sans lui, `04_train_models.py` ignore le LSTM et les tests correspondants sont sautés.

## D23 — PyTorch bloqué par Windows : version 2.10 (12/11/2026)
**Constat.** Sur le poste de développement (Windows 11), le contrôle des applications bloque `torch.dll` de PyTorch 2.15, sorti deux semaines plus tôt (WinError 4551). Il avait déjà bloqué les extensions DuckDB (D13). La DLL de LightGBM, elle, passe.
**Hypothèse.** Ce contrôle se fonde en partie sur la réputation des fichiers. Une version récente, encore peu diffusée, est bloquée ; une version plus ancienne et très téléchargée passe.
**Vérification.** `pip install "torch==2.10.*"` installe la version 2.10.0+cpu, qui se charge sans blocage. Le LSTM tourne.
**Choix.** Pas de version imposée dans `pyproject.toml`, puisque le blocage dépend du poste. Le contournement est documenté dans le README. On ne désactive pas la protection de Windows.
**Repli prévu si besoin.** WSL (Ubuntu) : les binaires Linux ne sont pas soumis à ce contrôle, et Docker en aura besoin de toute façon.

## D24 — Résultats du LSTM et explication (12/11/2026)
**Réglage.** 32 neurones (validation 1 089 MW, contre 1 102 MW pour 64). Le choix est au bord de la grille, mais l'écart est faible. Essayer 16 neurones reste une piste.
**Test.** 1 091 MW (2,09 %), le meilleur des modèles appris, 8 % devant LightGBM, mais 28 % derrière RTE recalée.
**Explication** (`06_diagnostics.py`) :
- Le gain sur LightGBM est réparti : environ −13 % au printemps et en automne, −9 % en hiver, mais +8 % en été. L'hypothèse « extrapolation en hiver » (D21) est donc secondaire.
- Le LSTM gagne surtout quand la consommation évolue de façon continue : janvier −24 à −27 %, et la nuit.
- Il perd aux ruptures de calendrier (décembre, août), où il surestime : il prolonge le niveau de la semaine normale qu'il vient de lire.

**Lecture.** La force du LSTM vient de l'historique complet de la semaine (168 h), pas d'une meilleure prise en compte de la température. Sa faiblesse est le revers de cette force.
**Écart avec les données factices.** La crainte d'une erreur plus forte aux extrémités de la journée (0 h, 23 h), vue sur données factices, ne se confirme pas sur les vraies données : le LSTM y est au contraire le meilleur la nuit.

## D25 — API : LightGBM servi, mode « rejeu »
**Modèle servi : LightGBM**, et non le LSTM pourtant 8 % plus précis. PyTorch alourdirait fortement l'image Docker et pose des problèmes d'installation sous Windows (D23). LightGBM prédit en quelques millisecondes dans une image légère. Le compromis est assumé, et réversible.
**Entraînement final.** `09_train_final.py` réentraîne LightGBM sur tout le passé disponible, avec le réglage retenu sur la validation, relu dans MLflow : aucun nouveau réglage. Les performances à annoncer restent celles du walk-forward.
**Pas de décalage entre entraînement et service.** L'API calcule les variables avec `LightGBMModel.features`, la fonction même de l'entraînement, et vérifie l'ordre des variables enregistré avec le modèle.
**Mode rejeu.** L'API prévoit les jours présents dans le jeu construit (variables connues la veille). Elle signale quand le jour demandé a été vu à l'entraînement. Le mode direct (construire la ligne de demain) est une étape ultérieure.
**Image Docker.** `python:3.12-slim`, `libgomp1` pour LightGBM, uniquement les dépendances de l'API, utilisateur sans privilèges, contrôle de santé. Données et modèle montés en volumes, hors de l'image et hors de Git.

## D26 — Combinaison des modèles
Moyenne pondérée régression / LightGBM / LSTM. Les poids (positifs, de somme 1, au pas de 0,1) sont choisis sur la validation, à partir de prévisions elles-mêmes hors échantillon (walk-forward), puis appliqués une seule fois au test.
Motivation (D24) : les faiblesses du LSTM (fêtes, août) et de LightGBM (vagues de froid) sont complémentaires.

## D27 — Graphiques
- Une couleur par modèle, la même dans tous les graphiques : LSTM bleu, LightGBM orange, RTE recalée vert d'eau. Consommation réelle en noir, RTE publiée en gris.
- Au plus trois modèles en couleur par graphique : au-delà, un contrôle des daltonismes (outil de validation de palette) signale deux teintes trop proches pour des courbes qui se croisent (magenta et orange).
- Légendes sous les titres, jamais sur les courbes. Pas de libellés en bout de courbe quand les courbes se rejoignent.

## D28 — Modèle final sans scikit-learn (12/11/2026)
**Constat.** Sur le poste Windows (Python 3.14, après l'installation de PyTorch 2.10), `09_train_final.py` a échoué : « scikit-learn is required for lightgbm.sklearn ». Pourtant, lancé seul, LightGBM voit bien scikit-learn. Le problème dépend donc de ce qui est importé avant (MLflow en tête du script). Il ne se reproduit pas sous Python 3.13. Cause exacte non établie.
**Choix.** L'entraînement final passe par l'interface native `lgb.train`, avec exactement les mêmes réglages (`native_params`, qui fait correspondre les noms scikit-learn et les noms natifs) et le même filtrage des lignes. Un test vérifie que les prévisions sont identiques à celles de l'interface scikit-learn utilisée dans le walk-forward. LightGBM est aussi importé avant MLflow dans `09`.
**Bénéfice.** Ni l'entraînement final ni l'API ne dépendent plus de scikit-learn, ce qui allège aussi l'image Docker.

## D29 — Volet 2 : prix Ember en attendant ENTSO-E (12/11/2026)
**Choix.** Les prix horaires de la France viennent du fichier public d'Ember, qui republie les données d'ENTSO-E (CC BY 4.0, citation obligatoire), en attendant le jeton ENTSO-E.
**Vérification** (`10_inspect_ember.py`) : fichier `France.csv`, pas horaire, de janvier 2015 au 9 novembre 2026, 5 heures manquantes sur toute la période. La fenêtre du projet est couverte à 100 %.
**Format.** Il n'est pas documenté par Ember. Le lecteur reconnaît donc les colonnes par leur nom et s'arrête avec un message explicite sinon. Sa sortie est identique à celle du lecteur ENTSO-E.
**Heures manquantes.** Le fichier Ember compte 5 heures manquantes. Deux d'entre elles sont la première heure d'un mois, en heure de Paris : le 1er octobre 2026 à 0 h et le 1er novembre 2026 à 0 h (vu dans les réponses de `/simulate`). C'est peut-être un défaut de raccord mensuel chez Ember, à vérifier contre ENTSO-E. Ces heures sont exclues de l'évaluation, sans effet sur l'estimation.
**À faire avec le jeton.** Comparer les deux séries heure par heure, puis relancer les scripts 11 à 13 avec `--source entsoe`.

## D30 — Modèle de prix : déterministe + AR(1) (Ornstein-Uhlenbeck discret)
**Partie déterministe** (moindres carrés) :
- 72 indicatrices heure × type de jour (semaine, samedi, dimanche/férié) ;
- 2 harmoniques annuelles ;
- une tendance linéaire.

**Résidu.** AR(1) horaire. φ est estimé par régression de x(t) sur x(t−1), sur les paires d'heures consécutives connues.
**Implémentation en NumPy**, pour que le script de validation et l'API partagent la même fonction (`simulate_month`). statsmodels sert à la contre-vérification : `AutoReg` donne le même φ que NumPy, et le test ADF établit la stationnarité qui justifie un retour à la moyenne. Il sert aussi au diagnostic, avec les autocorrélations à 1 h, 24 h et 168 h.
**Prix bruts, sans logarithme** : les prix négatifs l'interdisent.

## D31 — Sauts
**Détection.** Un choc est un saut si |ε − médiane| > 4 × écart-type robuste (1,4826 × MAD). L'écart-type classique serait lui-même gonflé par les sauts.
**Simulation.** À chaque heure, un saut survient avec la probabilité observée. Sa taille est tirée parmi les sauts réellement observés, pics vers le haut comme prix négatifs, sans imposer de loi. Les autres chocs sont gaussiens, avec l'écart-type calculé hors sauts. Les sauts se résorbent ensuite par le même retour à la moyenne.
**Pourquoi.** Sans sauts, la seule façon d'expliquer les pics est un écart-type élevé, ce qui donne des intervalles trop larges en temps normal et trop étroits lors des pics. Sur données factices construites avec des sauts, le modèle de base sur-couvre (environ 94 % pour une cible de 90 %) ; la variante avec sauts tombe juste (88 %).

## D32 — Variante « ancrage »
Le niveau déterministe est recalé sur la moyenne des 28 derniers jours connus.
**Motivation.** Le niveau des prix change avec le gaz et le CO2 : 61 €/MWh en moyenne en 2025, 95 €/MWh en 2026. Une tendance estimée sur 24 mois réagit trop lentement à de tels changements.
**Choix.** C'est une variante candidate, retenue ou non sur la validation comme les autres.

## D33 — Protocole de simulation et de validation
- **Simulation mensuelle.** Pour chaque mois, estimation sur les 24 mois précédents, puis 1 000 trajectoires. Les prix d'un jour sont publiés la veille, donc tout ce qui précède le 1er du mois est connu.
- **Choix de la variante par le CRPS moyen sur la validation.** Le CRPS est une règle de score propre : il est minimal quand la loi simulée est la vraie, et pénalise aussi bien des intervalles faux que des intervalles trop larges. La couverture seule récompenserait des intervalles immenses.
- **Une seule passe sur le test**, avec la variante retenue. Les autres variantes sont aussi évaluées sur le test, pour la discussion, mais ne servent pas au choix.

## D34 — Ce qui n'est pas fait (volet 2)
- **Prix dépendant de la consommation prévue.** Le cahier des charges propose les sauts ou la consommation : les sauts sont faits. Pour simuler un mois entier, il faudrait des scénarios de consommation sur tout le mois, car la prévision du volet 1 ne porte que sur le lendemain. Ce serait une variante « conditionnelle » à J+1.
- **Mémoire journalière du résidu.** Un AR(1) ignore la dépendance à 24 h. Le diagnostic la mesure (autocorrélation à 24 h comparée à φ^24) ; une extension serait un AR à décalages 1 et 24.

## D35 — /simulate dans l'API
Même fonction que la validation (`simulation.simulate_month`), avec la variante retenue lue dans `models/simulation_prix.json` (écrit par le script 13).
**Réponses de l'API :**
- les quantiles horaires et ceux du prix moyen du mois ;
- la couverture, si les prix réels du mois sont connus ;
- 503 sans données de prix, 422 si les prix précédant le mois sont inconnus.

Rapide (moins d'une seconde) et sans statsmodels, ce qui garde l'image Docker légère.

## D36 — Résultats réels de la simulation (12/11/2026)
**Variante retenue sur la validation : sauts + ancrage.** CRPS 21,5 €/MWh, contre 26,1 avec les sauts seuls et 26,6 pour la base ; couverture à 90 % de 89 %. Le classement des variantes est le même sur le test.
**Calibration dégradée sur le test.** Couverture à 90 % de 71 % (CRPS 30,7). La moyenne mensuelle réelle tombe dans l'intervalle 90 % simulé 2 mois sur 12. L'histogramme PIT est en U et penche à droite : les prix réels dépassent souvent toutes les trajectoires, au moment où ils montent de 66 à 166 €/MWh entre juin et octobre 2026.
**Diagnostic.** La mémoire longue des prix (autocorrélation de 0,57 à une semaine) et leur niveau, porté par le gaz, le CO2 et la disponibilité nucléaire, échappent à un AR(1) de demi-vie 25 h. L'amplitude journalière grandit avec le niveau, alors que le modèle suppose une variance constante. Les prix négatifs sont simulés trop souvent (8,3 % contre 5,4 %).
**Suite logique.** Un modèle à deux facteurs (niveau journalier lent + écart horaire rapide), une volatilité dépendant du niveau, et un AR à décalages 1 h et 24 h. Détail et chiffres dans le README.
