# =============================================================================
# FICHIER SÉPARÉ : Algorithme Naive Bayes (Gaussien)
# Objectif : Classification supervisée avec estimation de probabilités
# Note : Version Gaussienne adaptée aux features continues
# =============================================================================

import numpy as np                                               # Importe NumPy pour calculs vectoriels et statistiques
import matplotlib.pyplot as plt                                  # Importe Matplotlib pour tracés de densités et frontières
from sklearn.naive_bayes import GaussianNB                       # Importe le classifieur Bayésien Gaussien optimisé de Sklearn
from sklearn.metrics import ConfusionMatrixDisplay    # Importe métriques : précision et matrice de confusion
from sklearn.metrics import accuracy_score                       # Importe mesure de precision
import warnings                                                  # Importe module de gestion des avertissements
warnings.filterwarnings('ignore')                                # Masque les warnings pour garder la console lisible

# =============================================================================
#  DÉFINITION DE LA PETITE DATASET MANUELLE
# =============================================================================
# Dataset : 2 features (ex: temps de révision, qualité des notes) → Résultat examen
X_nb = np.array([[2.1, 0.8], [3.0, 1.2], [4.5, 2.1], [5.0, 2.8], [6.2, 3.5],   # Crée matrice 10x2 de features brutes
                 [7.0, 4.1], [8.5, 4.8], [9.0, 5.2], [9.5, 5.8], [10.0, 6.5]], dtype=float) # 10 échantillons
y_nb = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)                    # Vecteur binaire : 0=échec, 1=réussite

# =============================================================================
#  CLASSE : Naive Bayes Gaussien — From Scratch
#             Principe : Théorème de Bayes + hypothèse d'indépendance conditionnelle
# =============================================================================
class GaussianNBFromScratch:
    def __init__(self):
        self.classes = None                                      # Liste des classes uniques (ex: [0, 1])
        self.priors = None                                       # Probabilités a priori P(classe)
        self.means = None                                        # Moyennes par classe et par feature μ_c
        self.vars = None                                         # Variances par classe et par feature σ²_c

    def fit(self, X, y):
        self.classes = np.unique(y)                              # Identifie les classes présentes dans y
        self.priors = np.array([np.mean(y == c) for c in self.classes]) # Calcule P(c) = fréquence de chaque classe
        self.means = np.zeros((len(self.classes), X.shape[1]))   # Initialise matrice des moyennes (classes × features)
        self.vars = np.zeros((len(self.classes), X.shape[1]))    # Initialise matrice des variances (classes × features)
        for i, c in enumerate(self.classes):                     # Parcours chaque classe individuellement
            X_c = X[y == c]                                      # Extrait les échantillons appartenant à la classe c
            self.means[i, :] = X_c.mean(axis=0)                  # Calcule moyenne de chaque feature pour la classe c
            self.vars[i, :] = X_c.var(axis=0) + 1e-9             # Calcule variance + epsilon pour éviter division par zéro

    def _gaussian_pdf(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean)**2 / var)) # Formule de la densité normale N(μ,σ²)

    def predict(self, X):
        log_posteriors = []                                      # Liste pour stocker log(P(c|X)) pour chaque classe
        for i, c in enumerate(self.classes):                     # Parcours chaque classe pour calculer le posterior
            prior = np.log(self.priors[i])                       # Log de la probabilité a priori P(c)
            likelihood = self._gaussian_pdf(X, self.means[i, :], self.vars[i, :]) # P(X|c) via densité gaussienne
            log_likelihood = np.sum(np.log(likelihood), axis=1)  # Somme des log-likelihoods (hypothèse d'indépendance)
            log_posteriors.append(prior + log_likelihood)        # Règle de Bayes : log P(c|X) ∝ log P(c) + log P(X|c)
        log_posteriors = np.array(log_posteriors)                # Convertit liste en tableau NumPy (classes × échantillons)
        return self.classes[np.argmax(log_posteriors, axis=0)]   # Retourne la classe ayant le posterior maximal

# =============================================================================
#  ENTRAÎNEMENT & PRÉDICTIONS
# =============================================================================
nb_sc = GaussianNBFromScratch()                                  # Instancie le modèle Bayésien from scratch
nb_sc.fit(X_nb, y_nb)                                            # Entraîne sur l'intégralité des données
y_pred_sc = nb_sc.predict(X_nb)                                  # Prédit les classes avec l'implémentation manuelle

nb_lib = GaussianNB()                                            # Instancie le modèle Sklearn
nb_lib.fit(X_nb, y_nb)                                           # Entraîne le modèle bibliothèque
y_pred_lib = nb_lib.predict(X_nb)                                # Prédit avec Sklearn

# =============================================================================
#  MÉTRIQUES & AFFICHAGE CONSOLE
# =============================================================================
acc_sc = accuracy_score(y_nb, y_pred_sc)                         # Calcule précision du from scratch
acc_lib = accuracy_score(y_nb, y_pred_lib)                       # Calcule précision de Sklearn
print("="*70)                                                    # Ligne de séparation visuelle
print("  ALGORITHME NAIVE BAYES (GAUSSIEN) — Résultats")        # Titre section
print("="*70)                                                    # Ligne de séparation
print(f"  Précision (From Scratch) : {acc_sc:.4f}")             # Affiche score Scratch
print(f"  Précision (Scikit-Learn) : {acc_lib:.4f}")            # Affiche score Sklearn
print(f"  Moyennes par classe (Scratch) : \n{nb_sc.means}")     # Affiche paramètres appris μ
print(f"  Variances par classe (Scratch) : \n{nb_sc.vars}")     # Affiche paramètres appris σ²
print("="*70)                                                    # Fermeture visuelle

# =============================================================================
#  VISUALISATIONS PÉDAGOGIQUES (3 Graphiques)
# =============================================================================

# --- Graphique 1 : Densités Gaussiennes apprises par classe ---
plt.figure(figsize=(9, 6))                                       # Crée figure 9x6 pouces
colors = ['red', 'blue']                                         # Définit couleurs pour classes 0 et 1
x_axis = np.linspace(X_nb.min(), X_nb.max(), 100)               # Génère axe X continu pour tracer les courbes
for i, c in enumerate(nb_sc.classes):                            # Parcours chaque classe
    mean = nb_sc.means[i, 0]                                     # Moyenne de la feature 1 pour la classe c
    var = nb_sc.vars[i, 0]                                       # Variance de la feature 1 pour la classe c
    pdf = nb_sc._gaussian_pdf(x_axis, mean, var)                 # Calcule densité de probabilité sur l'axe
    plt.plot(x_axis, pdf, color=colors[i], linewidth=2,          # Trace courbe PDF
             label=f'Classe {int(c)} (μ={mean:.2f}, σ²={var:.2f})') # Légende avec paramètres
    plt.fill_between(x_axis, pdf, alpha=0.2, color=colors[i])   # Remplit sous la courbe pour lisibilité
plt.scatter(X_nb[:, 0], np.zeros_like(X_nb[:, 0]),             # Superpose points réels sur l'axe X
            c=y_nb, cmap='coolwarm', s=60, edgecolors='black', zorder=5) # Colorés par classe
plt.xlabel("Feature 1 (Temps de révision)")                     # Nomme axe horizontal
plt.ylabel("Densité de probabilité P(X|classe)")                # Nomme axe vertical
plt.title("Naive Bayes — Densités Gaussiennes apprises")        # Titre explicite
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()     # Légende, grille, ajustement
plt.show()                                                       # Affiche graphique 1

# --- Graphique 2 : Frontière de décision 2D ---
def plot_nb_boundary(X, y, model, titre):
    x0_min, x0_max = X[:,0].min()-1, X[:,0].max()+1             # Limites axe X avec marge
    x1_min, x1_max = X[:,1].min()-1, X[:,1].max()+1             # Limites axe Y avec marge
    xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, 200),      # Crée grille dense 200x200
                         np.linspace(x1_min, x1_max, 200))      # pour évaluation continue
    grid = np.c_[xx.ravel(), yy.ravel()]                        # Aplatis grille en liste (N,2)
    Z = model.predict(grid)                                     # Prédit classe pour chaque point grille
    Z = Z.reshape(xx.shape)                                     # Remet forme 2D pour contourf
    plt.figure(figsize=(7, 6))                                  # Figure dédiée
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')         # Zones de décision colorées
    plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm',           # Points réels superposés
                s=70, edgecolors='black', zorder=5)             # Bordures noires pour contraste
    plt.xlabel("Feature 1"); plt.ylabel("Feature 2")            # Axes nommés
    plt.title(titre); plt.grid(True, alpha=0.3); plt.tight_layout() # Titre, grille, ajustement
    plt.show()                                                  # Affichage

plot_nb_boundary(X_nb, y_nb, nb_sc, "Naive Bayes — Frontière (From Scratch)") # Trace frontière manuelle
plot_nb_boundary(X_nb, y_nb, nb_lib, "Naive Bayes — Frontière (Scikit-Learn)") # Trace bibliothèque

# --- Graphique 3 : Matrices de confusion comparatives ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))                 # 2 sous-graphiques côte à côte
ConfusionMatrixDisplay.from_predictions(y_nb, y_pred_sc,       # Matrice from scratch
                                        ax=axes[0], cmap='Blues',
                                        display_labels=['Échec (0)', 'Réussite (1)']) # Labels lisibles
axes[0].set_title("From Scratch")                               # Titre subplot gauche
ConfusionMatrixDisplay.from_predictions(y_nb, y_pred_lib,      # Matrice Sklearn
                                        ax=axes[1], cmap='Greens',
                                        display_labels=['Échec (0)', 'Réussite (1)']) # Labels lisibles
axes[1].set_title("Scikit-Learn")                               # Titre subplot droite
fig.suptitle("Naive Bayes — Matrices de Confusion", fontweight='bold') # Titre global
plt.tight_layout(); plt.show()                                  # Ajuste et affiche

print("\n Exécution Naive Bayes terminée. 4 graphiques générés.") # Message fin console

