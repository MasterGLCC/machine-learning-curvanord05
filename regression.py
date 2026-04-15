#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# PROJET : Régression Linéaire, Multiple, Polynomiale & Logistique
# CONSIGUE : From Scratch + Bibliothèque | Petit dataset | Pas de test/scaling
# =============================================================================

import numpy as np                # Importe NumPy pour les calculs matriciels et tableaux
import matplotlib.pyplot as plt   # Importe Matplotlib pour tracer les courbes et points
from sklearn.linear_model import LinearRegression, LogisticRegression # Importe les modèles linéaires et logistiques de Scikit-Learn
from sklearn.preprocessing import PolynomialFeatures  # Importe le générateur de features polynomiales X, X², X³...
import warnings                   # Importe le module de gestion des avertissements
warnings.filterwarnings('ignore') # Désactive les warnings pour garder la console propre

# =============================================================================
#  DÉFINITION DE LA PETITE DATASET (Manuelle pour la clarté)
# =============================================================================
# Régression Simple & Polynomiale (1 variable, cible continue)
X_simple = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1) # Crée un tableau colonne de 10 valeurs
y_simple = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.9, 16.2, 17.8, 20.1]) # Cible linéaire bruitée

# Régression Multiple (2 variables, cible continue)
X_multi = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]) # 8 échantillons, 2 features
y_multi = np.array([5.1, 7.2, 9.1, 11.3, 13.2, 15.1, 17.3, 19.2]) # Cible dépendant des 2 features

# Régression Logistique (1 variable, cible binaire 0/1)
X_log = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]).reshape(-1, 1) # 10 échantillons
y_log = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) # Classes 0 ou 1 pour la classification

# =============================================================================
#  1. RÉGRESSION LINÉAIRE SIMPLE (From Scratch & Bibliothèque)
# =============================================================================
class LinearRegressionFromScratch:
    def __init__(self):
        self.theta = None  # Initialise les coefficients à None avant l'entraînement

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X] # Ajoute une colonne de 1 pour le biais (intercept)
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y # Résout θ = (XX)⁻¹Xᵀy avec pseudo-inverse stable
        return self                                     # Retourne l'instance pour chaînage

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X] # Réajoute la colonne de 1 pour correspondre aux dimensions de θ
        return X_b @ self.theta                      # Calcule ŷ = X·θ

# Exécution Linéaire Simple
lin_scratch = LinearRegressionFromScratch().fit(X_simple, y_simple) # Entraîne le modèle from scratch sur toutes les données
y_pred_lin_scratch = lin_scratch.predict(X_simple)    # Prédit sur les mêmes données (pas de split test)

lin_lib = LinearRegression().fit(X_simple, y_simple)  # Entraîne le modèle Scikit-Learn
y_pred_lin_lib = lin_lib.predict(X_simple)            # Prédit avec la bibliothèque

print(" Régression Linéaire Simple")
print("Coefficients Scratch:", lin_scratch.theta)     # Affiche les poids appris (biais + pente)
print("Coefficients Sklearn:", lin_lib.intercept_, lin_lib.coef_) # Affiche intercept et coef de sklearn

# =============================================================================
#  2. RÉGRESSION MULTIPLE (From Scratch & Bibliothèque)
# =============================================================================
# La même classe LinearRegressionFromScratch fonctionne pour la multiple (la formule est identique)
multi_scratch = LinearRegressionFromScratch().fit(X_multi, y_multi) # Entraîne sur 2 features
y_pred_multi_scratch = multi_scratch.predict(X_multi)

multi_lib = LinearRegression().fit(X_multi, y_multi)  # Entraîne le modèle multiple via Scikit-Learn
y_pred_multi_lib = multi_lib.predict(X_multi)

print("\n Régression Linéaire Multiple")
print("Coefficients Scratch:", multi_scratch.theta)
print("Coefficients Sklearn:", multi_lib.intercept_, multi_lib.coef_)

# =============================================================================
#  3. RÉGRESSION POLYNOMIALE (Degré 2) (From Scratch & Bibliothèque)
# =============================================================================
class PolynomialRegressionFromScratch:
    def __init__(self, degree=2):
        self.degree = degree                           # Stocke le degré du polynôme
        self.linear_model = LinearRegressionFromScratch() # Utilise la régression linéaire sur les features transformées

    def _generate_features(self, X):
        return np.column_stack([X**i for i in range(1, self.degree + 1)]) # Crée [X, X², X³...] manuellement

    def fit(self, X, y):
        X_poly = self._generate_features(X)            # Transforme X en features polynomiales
        self.linear_model.fit(X_poly, y)               # Entraîne la régression linéaire sur ces nouvelles features
        return self

    def predict(self, X):
        X_poly = self._generate_features(X)            # Regénère les features polynomiales pour la prédiction
        return self.linear_model.predict(X_poly)       # Retourne les prédictions du modèle linéaire sous-jacent

# Exécution Polynomiale
poly_scratch = PolynomialRegressionFromScratch(degree=2).fit(X_simple, y_simple) # Fit polynômial from scratch
y_pred_poly_scratch = poly_scratch.predict(X_simple)

poly_lib = LinearRegression().fit(PolynomialFeatures(degree=2).fit_transform(X_simple), y_simple) # Pipeline sklearn
y_pred_poly_lib = poly_lib.predict(PolynomialFeatures(degree=2).transform(X_simple))

print("\n Régression Polynomiale (Degré 2)")
print("Coefficients Scratch:", poly_scratch.linear_model.theta)
print("Coefficients Sklearn:", poly_lib.intercept_, poly_lib.coef_)

# =============================================================================
#  4. RÉGRESSION LOGISTIQUE (From Scratch & Bibliothèque)
# =============================================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z)) # Fonction sigmoïde qui mappe ℝ vers [0,1] pour les probabilités

class LogisticRegressionFromScratch:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr        # Taux d'apprentissage pour la descente de gradient
        self.epochs = epochs # Nombre d'itérations d'optimisation
        self.weights = None # Coefficients des features
        self.bias = 0       # Terme de biais

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) # Initialise les poids à 0
        self.bias = 0                       # Initialise le biais à 0
        for _ in range(self.epochs):
            linear_pred = np.dot(X, self.weights) + self.bias # Calcule z = w·X + b
            predictions = sigmoid(linear_pred)                # Applique la sigmoïde pour obtenir P(y=1)
            dw = (1/n_samples) * np.dot(X.T, (predictions - y)) # Gradient des poids
            db = (1/n_samples) * np.sum(predictions - y)      # Gradient du biais
            self.weights -= self.lr * dw                      # Mise à jour des poids
            self.bias -= self.lr * db                         # Mise à jour du biais
        return self

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias) # Retourne les probabilités prédites

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int) # Classe 1 si proba ≥ 0.5, sinon 0

# Exécution Logistique
log_scratch = LogisticRegressionFromScratch().fit(X_log, y_log) # Entraîne le classifieur from scratch
y_pred_log_scratch = log_scratch.predict(X_log)

log_lib = LogisticRegression().fit(X_log, y_log)      # Entraîne le classifieur Scikit-Learn
y_pred_log_lib = log_lib.predict(X_log)

print("\n Régression Logistique")
print("Weights Scratch:", log_scratch.weights, "Bias:", log_scratch.bias)
print("Weights Sklearn:", log_lib.coef_, "Intercept:", log_lib.intercept_)
print("Accuracy Scratch:", np.mean(y_pred_log_scratch == y_log))
print("Accuracy Sklearn:", log_lib.score(X_log, y_log))

# =============================================================================
#  VISUALISATION SIMPLE (Courbes & Points)
# =============================================================================
plt.figure(figsize=(12, 8))                           # Crée une figure de 12x8 pouces

# Graphique 1 : Linéaire & Polynomiale
plt.subplot(1, 2, 1)                                  # Place le graphique à gauche
plt.scatter(X_simple, y_simple, color='black', label='Données') # Trace les points réels
plt.plot(X_simple, y_pred_lin_lib, 'r-', label='Linéaire')      # Trace la droite linéaire
plt.plot(X_simple, y_pred_poly_lib, 'g--', label='Polynomiale D2') # Trace la courbe polynomiale
plt.title("Régression Simple & Polynomiale")          # Titre du sous-graphique
plt.xlabel("X"); plt.ylabel("y")                      # Nommage des axes
plt.legend(); plt.grid(True, alpha=0.3)               # Affiche légende et grille

# Graphique 2 : Logistique
plt.subplot(1, 2, 2)                                  # Place le graphique à droite
plt.scatter(X_log, y_log, color='black', label='Classes réelles') # Trace les points 0/1
plt.plot(X_log, log_lib.predict_proba(X_log), 'b-', label='Probabilité (Sklearn)') # Trace la courbe de probabilité
plt.axhline(0.5, color='gray', linestyle=':', label='Seuil 0.5') # Ligne de décision
plt.title("Régression Logistique")                    # Titre du sous-graphique
plt.xlabel("X"); plt.ylabel("P(y=1)")                 # Nommage des axes
plt.legend(); plt.grid(True, alpha=0.3)               # Affiche légende et grille

plt.tight_layout()                                    # Ajuste l'espacement entre les graphiques
plt.show()                                            # Affiche la fenêtre graphique finale

print("\n Exécution terminée. Code prêt pour le rendu.") # Message de fin
