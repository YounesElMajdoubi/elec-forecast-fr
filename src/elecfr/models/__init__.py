"""Modèles de prévision. Chacun expose la même interface que les références :
un objet appelable `modele(train, target) -> np.ndarray`, utilisable tel quel
par `evaluation.walk_forward`. Le dernier modèle entraîné reste disponible
dans `modele.dernier` pour être enregistré dans MLflow."""
