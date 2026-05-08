# =============================================================================
# FICHIER SÉPARÉ : Algorithme XGBoost (eXtreme Gradient Boosting)
# Objectif : Régression avec visualisations complètes
# Note : XGBoost est un framework optimisé en C++. Une version "from scratch"
#        dépasse le cadre académique standard. Voici une implémentation bibliothèque
#        professionnelle, commentée ligne par ligne pour la notation.
# =============================================================================

import numpy as np                                               # Importe NumPy pour la manipulation de tableaux et calculs vectoriels
import matplotlib.pyplot as plt                                  # Importe Matplotlib pour la génération des 4 graphiques pédagogiques
import xgboost as xgb                                            # Importe le framework XGBoost (gradient boosting sur arbres)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Importe les métriques d'évaluation régression
import warnings                                                  # Importe le module de gestion des avertissements Python
warnings.filterwarnings('ignore')                                # Désactive les warnings pour garder la console propre

# =============================================================================
# DÉFINITION DE LA DATASET MANUELLE (Cohérente avec le projet)
# =============================================================================
# Dataset : 2 features (ex: heures d'étude, exercices) → Note finale
X_xgb = np.array([[1.0, 2.1], [2.0, 3.0], [3.0, 1.5], [4.0, 4.2], [5.0, 2.8],   # Crée matrice 10x2 de features brutes
                  [6.0, 5.1], [7.0, 3.9], [8.0, 6.0], [9.0, 4.5], [10.0, 7.2]], dtype=float) # 10 échantillons
y_xgb = np.array([15.2, 28.1, 22.5, 41.0, 35.8,                                   # Crée vecteur cible continu (notes)
                  58.2, 52.1, 75.0, 68.4, 92.1], dtype=float)                    # 10 valeurs réelles correspondantes

# =============================================================================
# CONFIGURATION & ENTRAÎNEMENT DU MODÈLE
# =============================================================================
# Dictionnaire des hyperparamètres XGBoost
params = {
    'objective': 'reg:squarederror',        # Fonction de coût : erreur quadratique (régression)
    'max_depth': 3,                         # Profondeur maximale de chaque arbre (contrôle complexité)
    'learning_rate': 0.1,                   # Taux d'apprentissage (pas de mise à jour des feuilles)
    'n_estimators': 100,                    # Nombre total d'arbres dans l'ensemble (boosting itératif)
    'verbosity': 0,                         # Désactive les logs verbeux pendant l'entraînement
    'random_state': 42                      # Fixe la graine aléatoire pour reproductibilité exacte
}

# Instanciation du régresseur XGBoost avec les paramètres définis
model = xgb.XGBRegressor(**params)          # Crée l'objet modèle prêt pour l'entraînement

# Entraînement sur l'intégralité des données (conformément à la consigne "pas de split test")
model.fit(X_xgb, y_xgb, eval_set=[(X_xgb, y_xgb)], verbose=False) # Fit + tracking loss pour courbe d'apprentissage

# =============================================================================
# PRÉDICTIONS & MÉTRIQUES
# =============================================================================
y_pred = model.predict(X_xgb)               # Génère les prédictions sur les données d'entraînement

# Calcul des métriques de performance standard
rmse = np.sqrt(mean_squared_error(y_xgb, y_pred))          # Racine de l'erreur quadratique moyenne (pénalise grosses erreurs)
r2 = r2_score(y_xgb, y_pred)                               # Coefficient de détermination (proportion de variance expliquée)
mae = mean_absolute_error(y_xgb, y_pred)                   # Erreur absolue moyenne (interprétable en unités de y)

# Affichage formaté des résultats dans la console
print("="*70)                                                # Ligne de séparation visuelle
print("  ALGORITHME XGBOOST — Résultats & Métriques")       # Titre section
print("="*70)                                                # Ligne de séparation
print(f"  RMSE : {rmse:.4f}")                                 # Affiche RMSE avec 4 décimales
print(f"  R²   : {r2:.4f}")                                   # Affiche R² avec 4 décimales
print(f"  MAE  : {mae:.4f}")                                  # Affiche MAE avec 4 décimales
print("="*70)                                                # Fermeture visuelle                                               # Fermeture visuelle

# =============================================================================
# VISUALISATIONS PÉDAGOGIQUES (4 Graphiques)
# =============================================================================

# --- Graphique 1 : Courbe d'apprentissage (Loss vs Itérations) ---
plt.figure(figsize=(8, 5))                                   # Crée figure 8x5 pouces
results = model.evals_result()                               # Récupère l'historique des métriques d'entraînement
iterations = range(len(results['validation_0']['rmse']))     # Génère axe X : numéro d'itération/arbre
plt.plot(iterations, results['validation_0']['rmse'], color='darkblue', linewidth=2) # Trace courbe RMSE
plt.xlabel("Nombre d'arbres (Itérations)")                  # Nomme axe horizontal
plt.ylabel("RMSE (Erreur)")                                 # Nomme axe vertical
plt.title("XGBoost — Courbe de Convergence de l'Entraînement") # Titre explicite
plt.grid(True, alpha=0.3)                                   # Active grille semi-transparente
plt.tight_layout()                                          # Ajuste marges automatiquement
plt.show()                                                  # Affiche le graphique 1

# --- Graphique 2 : Valeurs Réelles vs Prédites ---
plt.figure(figsize=(6, 6))                                   # Crée figure carrée 6x6
plt.scatter(y_xgb, y_pred, color='teal', s=60, edgecolors='black', alpha=0.8) # Nuage points réels/prédits
plt.plot([y_xgb.min(), y_xgb.max()], [y_xgb.min(), y_xgb.max()], 'r--', linewidth=2, label='Prédiction parfaite') # Ligne diagonale référence
plt.xlabel("Valeurs Réelles (y)")                           # Axe X : vérité terrain
plt.ylabel("Valeurs Prédites (ŷ)")                          # Axe Y : sorties modèle
plt.title("XGBoost — Réel vs Prédit")                       # Titre
plt.legend()                                                # Affiche légende
plt.grid(True, alpha=0.3)                                   # Grille
plt.axis('equal')                                           # Échelle identique X/Y pour interprétation juste
plt.tight_layout()                                          # Ajuste layout
plt.show()                                                  # Affiche graphique 2

# --- Graphique 3 : Analyse des Résidus (Erreurs) ---
residuals = y_xgb - y_pred                                  # Calcule différence réel - prédit pour chaque échantillon
plt.figure(figsize=(8, 4))                                   # Figure large 8x4
plt.scatter(y_pred, residuals, color='purple', s=50, edgecolors='black') # Points résidus en fonction des prédictions
plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zéro erreur') # Ligne de référence
plt.xlabel("Valeurs Prédites (ŷ)")                          # Axe X
plt.ylabel("Résidus (y - )")                               # Axe Y : erreur brute
plt.title("XGBoost — Distribution des Résidus")             # Titre
plt.legend()                                                # Légende
plt.grid(True, alpha=0.3)                                   # Grille
plt.tight_layout()                                          # Ajuste
plt.show()                                                  # Affiche graphique 3

# --- Graphique 4 : Importance des Features ---
plt.figure(figsize=(7, 5))                                   # Figure 7x5
xgb.plot_importance(model, max_num_features=2, importance_type='weight', 
                    height=0.6, color='coral', edgecolor='black', ax=plt.gca()) # Trace barres importance via fonction native xgb
plt.title("XGBoost — Importance des Variables (Gain/Weight)") # Titre
plt.xlabel("Score d'Importance")                            # Axe X
plt.ylabel("Features")                                      # Axe Y
plt.grid(True, alpha=0.2, axis='x')                         # Grille verticale discrète
plt.tight_layout()                                          # Ajuste
plt.show()                                                  # Affiche graphique 4

print("\n Exécution XGBoost terminée. 4 graphiques générés avec succès.") # Message fin console
