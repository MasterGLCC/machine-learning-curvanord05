#  Projet : Régression Linéaire, Multiple, Polynomiale & Logistique

**Auteur :** Ismail Erguig 

---

##  Objectif
Implémenter et comparer 4 modèles de Machine Learning :
1. Régression Linéaire Simple
2. Régression Linéaire Multiple
3. Régression polynomiale (degré 2)
4. Régression Logistique (classification binaire)

### Classification :
5. KNN - K-Nearest Neighbors (k=3)
6. SVM - Support Vector Machine (noyau linéaire)

Chaque modèle est codé deux fois :
- From Scratch : Implémentation manuelle des formules mathématiques
- Bibliothèque : Utilisation de `scikit-learn` pour validation

---

##  Dataset & Contraintes
Conformément aux consignes du projet :
- Petite dataset manuelle : Définition explicite de `X` et `y` dans le code (10 à 8 échantillons)

---

## Méthodologie
File: regression.py (six algorithms)

* Type : Régression supervisée
* Implémentation : Équation normale θ = (XᵀX)⁻¹Xᵀ (from scratch)y
* Méthode : Pseudo-inverse pour stabilité numérique
* Visualisations : Courbe de prédiction, résidus, réel vs prédit

2. Régression Polynomiale (Degré 2)

* Type : Régression non-linéaire
* Principe : Transformation X → [X, X²] + régression linéaire
* From Scratch: Génération manuelle des features polynomiales
* Visualisations : Comparaison linéaire vs polynomial

3. Régression Logistique

* Type : Classification binaire
* From Scratch : descente de gradient + fonction sigmoïde
* Optimisation : Mise à jour itérative des poids
* Visualisations : Frontière de décision, courbe d'apprentissage

4. KNN (K-Nearest Neighbors)

* Type : Classification par voisins
* From Scratch : Distance euclidienne + vote majoritaire
* Paramètre : k = 3 voisins
* Visualisations : Frontière de décision, accuracy vs k

5. SVM (Support Vector Machine)

* Type : Classification à marges maximales
* From Scratch : Descente de gradient sur Hinge Loss
* Noyau : Linéaire
* Visualisations : Frontière SVM, convergence de la loss

---

 Fichier : xgboost_modele.py

6. XGBoost (eXtreme Gradient Boosting)

* Type : Régression par arbres boostés
* Méthode : Application de la bibliothèque XGBoost, qui construit séquentiellement 100 arbres à faible profondeur (max=3) afin d’améliorer la précision des prédictions.
* Visualisations :
  * Courbe de convergence (RMSE vs itérations)
  * Réel vs prédit
  * Analyse des résidus
  * Importance des variables

---

 Fichier : dbscan_modele.py

7. DBSCAN (Density-Based Spatial Clustering)

* Type : Clustering non supervisé basé sur la densité
* Avantages :
  * Détecte automatiquement le nombre de clusters
  * Gère les formes arbitraires (non convexes)
  * Identifie les outliers
* Paramètres : eps=0.3, min_samples=5
* Visualisations :
  * Clustering sur 3 datasets (lunes, cercles, custom)
  * Comparaison des scores Silhouette
  * Sensibilité à epsilon
  * Vue 3D

---

 Fichier : naive_bayes_modele.py

8. Naive Bayes Gaussian

* Type: Classification probabiliste
* From Scratch : Théorème de Bayes + hypothèse d'indépendance
* Principe : Estimation des densités gaussiennes par classe
* Visualisations :
  * Densités de probabilité apprises
  * Frontières de décision (Scratch vs Sklearn)
  * Matrices de confusion comparatives

---

Fichier : qlearning_modele.py

9. Q-Learning (Reinforcement Learning)

* Type : Apprentissage par renforcement
* Environnement : Grid World 5×5 (départ → objectif + pièges)
* From Scratch :
  * Q-Table initialisée à zéro
  * Équation de Bellman pour mise à jour
  * Stratégie ε-greedy avec decay
* Visualisations :
  * Courbe d'apprentissage (récompenses)
  * Décroissance de l'exploration (ε)
  * Longueur des épisodes
  * Heatmap de la Q-Table finale
  * Chemin optimal appris

---

Fichier : pca_modele.py

10. PCA (Principal Component Analysis)

* Type : Réduction de dimension non supervisée
* From Scratch :
  * Centrage des données
  * Matrice de covariance
  * Décomposition en valeurs/vecteurs propres
  * Projection sur composantes principales
* Objectif : 2D → 1D avec variance maximale
* Visualisations :
  * Données originales + direction PC1
  * Projections orthogonales
  * Variance expliquée



* Comprendre le fonctionnement interne
* Visualiser les frontières de décision
* Analyser la convergence et les erreurs


