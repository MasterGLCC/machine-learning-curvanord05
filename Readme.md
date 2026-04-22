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

### 1. Régression Linéaire & Multiple (From Scratch)
- Formule : Équation normale `θ = (XᵀX)⁻¹Xᵀy`
- Implémentation : `np.linalg.pinv()` pour une inversion matricielle stable
- Multiple : Même formule, seule la dimension de `X` change (2 colonnes au lieu d'1)

### 2. Régression Polynomiale
- Principe : Transformation non-linéaire `X → [X, X², X³...]` suivie d'une régression linéaire
- From Scratch: Génération manuelle des puissances via `np.column_stack()`
- Bibliothèque : `PolynomialFeatures` + `LinearRegression`

### 3. Régression Logistique
- Fonction : Sigmoïde `σ(z) = 1 / (1 + e⁻ᶻ)` pour mapper les sorties vers [0,1]
- Optimisation : Descente de gradient manuelle (mise à jour des poids et biais)
- Seuil de décision: `0.5` pour convertir les probabilités en classes `0` ou `1`



