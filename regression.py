# =============================================================================
# PROJET : Régression & Classification — From Scratch & Scikit-Learn
#          1. Régression Linéaire Simple
#          2. Régression Linéaire Multiple
#          3. Régression Polynomiale
#          4. Régression Logistique
#          5. Algorithme KNN  (K-Nearest Neighbors)
#          6. Algorithme SVM  (Support Vector Machine)
# =============================================================================

import numpy as np                                               # Importe NumPy pour le calcul matriciel, vectoriel et les tableaux multidimensionnels
import matplotlib.pyplot as plt                                  # Importe Matplotlib pour la création et l'affichage de graphiques 2D
from sklearn.linear_model  import LinearRegression, LogisticRegression  # Importe les classes de régression linéaire et logistique optimisées de Scikit-Learn
from sklearn.preprocessing import PolynomialFeatures             # Importe le transformateur qui génère automatiquement les puissances de X (X², X³...)
from sklearn.neighbors     import KNeighborsClassifier           # Importe l'implémentation optimisée de KNN dans Scikit-Learn
from sklearn.svm           import SVC                            # Importe le classifieur à marges maximales (SVM) de Scikit-Learn
from sklearn.metrics       import mean_squared_error, r2_score, accuracy_score  # Importe les fonctions d'évaluation : RMSE, R², Précision

# =============================================================================
# DATASETS MANUELS (petits, sans fichier externe)
# =============================================================================

# --- Dataset régression (1 feature) : heures d'étude → note ---
X_simple = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], dtype=float)  # Crée un tableau colonne de 10 valeurs flottantes (heures)
y_simple  = np.array([10,20,25,30,45,50,55,65,72,80],          dtype=float)  # Crée le vecteur cible correspondant (notes sur 100)

# --- Dataset régression multiple (2 features) : heures + exercices → note ---
X_multi = np.array([[1,2],[2,3],[3,1],[4,4],[5,2],
                    [6,5],[7,3],[8,6],[9,4],[10,7]], dtype=float) # Crée une matrice 10x2 : colonne 0 = heures, colonne 1 = exercices
y_multi = np.array([12,22,20,35,38,52,58,70,74,85], dtype=float) # Crée le vecteur cible pour la régression multiple

# --- Dataset classification (2 features) : heures + exercices → admis/refusé ---
X_clf = np.array([[1,1],[2,1],[2,2],[3,2],[4,3],
                  [5,4],[6,4],[7,5],[8,5],[9,6]], dtype=float)   # Crée la matrice de features pour la classification
y_clf = np.array([0,0,0,0,1,1,1,1,1,1],            dtype=float) # Crée le vecteur binaire : 0 = refusé, 1 = admis

# --- Labels SVM : -1 / +1  (convention mathématique SVM) ---
y_svm_labels = np.where(y_clf == 0, -1, 1)                       # Remplace les 0 par -1 et garde les 1, conformément à la formulation SVM

# =============================================================================
# NORMALISATION MIN-MAX  (uniquement pour SVM — stabilise la descente de gradient)
# =============================================================================
def normaliser(X):
    X_min = X.min(axis=0)                                        # Calcule le minimum de chaque colonne (feature)
    X_max = X.max(axis=0)                                        # Calcule le maximum de chaque colonne (feature)
    return (X - X_min) / (X_max - X_min + 1e-8)                 # Applique la formule min-max + epsilon pour éviter la division par zéro

X_clf_n = normaliser(X_clf)                                       # Normalise les données de classification spécifiquement pour le SVM

# =============================================================================
# CLASSE 1 : Régression Linéaire & Multiple — From Scratch
#            Méthode : équation normale  θ = (XᵀX)¹ Xᵀy
# =============================================================================
class LinearRegressionScratch:
    def __init__(self):
        self.theta = None                                         # Initialise l'attribut theta (coefficients) à None avant l'entraînement

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]                # Concatène une colonne de 1 à gauche de X pour modéliser le biais (ordonnée à l'origine)
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y   # Résout analytiquement θ via le pseudo-inverse (plus stable que l'inverse classique)
        return self                                               # Retourne l'instance self pour permettre l'enchaînement .fit().predict()

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]                # Réapplique l'ajout de la colonne de 1 pour aligner les dimensions avec theta
        return X_b @ self.theta                                  # Effectue le produit matriciel X·θ pour obtenir les prédictions ŷ

# =============================================================================
# CLASSE 2 : Régression Polynomiale — From Scratch
#            Principe : X → [X, X², X³] puis régression linéaire
# =============================================================================
class PolynomialRegressionScratch:
    def __init__(self, degree=2):
        self.degree = degree                                      # Stocke le degré polynomial demandé (ex: 2 → X et X²)
        self.model  = LinearRegressionScratch()                   # Instancie le modèle de régression linéaire qui résoudra le problème transformé

    def _transform(self, X):
        return np.column_stack([X**i for i in range(1, self.degree+1)]) # Génère manuellement les colonnes [X¹, X², ..., X^degree] et les assemble

    def fit(self, X, y):
        self.model.fit(self._transform(X), y)                    # Entraîne le modèle linéaire sous-jacent sur les features polynomiales générées
        return self                                               # Retourne l'instance pour chaînage de méthodes

    def predict(self, X):
        return self.model.predict(self._transform(X))            # Transforme X puis délègue la prédiction au modèle linéaire entraîné

# =============================================================================
# CLASSE 3 : Régression Logistique — From Scratch
#            Méthode : descente de gradient + sigmoïde σ(z) = 1/(1+e⁻ᶻ)
# =============================================================================
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=1000):
        self.lr     = lr                                          # Définit le taux d'apprentissage (pas de mise à jour des poids)
        self.n_iter = n_iter                                      # Définit le nombre d'itérations de la descente de gradient
        self.theta  = None                                        # Initialise les paramètres du modèle à None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))                              # Applique la fonction sigmoïde pour mapper toute valeur réelle dans [0, 1]

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]                # Ajoute la colonne de biais pour inclure l'intercept dans le calcul
        self.theta = np.zeros(X_b.shape[1])                      # Initialise theta avec des zéros (taille = nombre de features + 1)
        for _ in range(self.n_iter):                             # Boucle principale de la descente de gradient
            y_hat      = self._sigmoid(X_b @ self.theta)         # Calcule les probabilités prédites P(y=1|X) pour tous les échantillons
            grad       = X_b.T @ (y_hat - y) / len(y)           # Calcule le gradient de la fonction de coût (log-loss) par rapport à theta
            self.theta -= self.lr * grad                         # Met à jour theta en soustrayant le gradient pondéré par le learning rate
        return self                                               # Retourne le modèle entraîné

    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]                # Ajoute la colonne de biais pour la prédiction
        return self._sigmoid(X_b @ self.theta)                   # Retourne le vecteur de probabilités pour chaque échantillon

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)        # Applique un seuil à 0.5 : proba ≥ 0.5 → classe 1, sinon classe 0

# =============================================================================
# CLASSE 4 : KNN — K-Nearest Neighbors — From Scratch
#            Principe : vote des K voisins les plus proches (distance euclidienne)
# =============================================================================
class KNNScratch:
    def __init__(self, k=3):
        self.k = k                                                # Stocke le nombre de voisins à considérer pour le vote

    def fit(self, X, y):
        self.X_train = X                                         # Mémorise le dataset d'entraînement (KNN est un algorithme paresseux)
        self.y_train = y                                         # Mémorise les étiquettes correspondantes
        return self                                               # Retourne l'instance prête pour la prédiction

    def _distance(self, a, b):
        return np.sqrt(np.sum((a - b)**2))                       # Calcule la distance euclidienne L2 entre deux vecteurs de features

    def predict(self, X):
        predictions = []                                         # Initialise la liste qui stockera les prédictions finales
        for x in X:                                              # Parcours chaque échantillon à classifier
            dists    = [self._distance(x, xtr) for xtr in self.X_train] # Calcule la distance entre x et tous les points d'entraînement
            k_idx    = np.argsort(dists)[:self.k]               # Trie les distances par ordre croissant et garde les indices des k plus proches
            k_labels = self.y_train[k_idx]                      # Récupère les classes réelles de ces k voisins
            predictions.append(int(np.bincount(k_labels.astype(int)).argmax())) # Compte les occurrences et retourne la classe majoritaire
        return np.array(predictions)                             # Convertit la liste en tableau NumPy pour compatibilité sklearn

# =============================================================================
# CLASSE 5 : SVM — Support Vector Machine — From Scratch
#            Méthode : descente de gradient stochastique sur Hinge Loss
#            L = (1/2)||w||² + C · Σ max(0, 1 - yᵢ(w·xᵢ + b))
# =============================================================================
class SVMScratch:
    def __init__(self, lr=0.001, C=1.0, n_iter=1000):
        self.lr     = lr                                         # Taux d'apprentissage pour la mise à jour des paramètres
        self.C      = C                                          # Hyperparamètre de régularisation (compromis marge/erreur)
        self.n_iter = n_iter                                     # Nombre d'époques d'entraînement
        self.w      = None                                       # Vecteur de poids normal à l'hyperplan séparateur
        self.b      = 0                                          # Terme de biais (décalage de l'hyperplan)
        self.losses = []                                         # Liste pour enregistrer l'évolution de la Hinge Loss à chaque itération

    def fit(self, X, y):
        n, p   = X.shape                                        # Récupère le nombre d'échantillons (n) et de features (p)
        self.w = np.zeros(p)                                    # Initialise le vecteur de poids avec des zéros
        self.b = 0                                              # Initialise le biais à zéro

        for _ in range(self.n_iter):                            # Boucle principale d'optimisation
            loss = 0                                            # Accumulateur pour la perte sur l'epoch courante
            for i in range(n):                                  # Parcours chaque échantillon un par un (SGD)
                cond = y[i] * (np.dot(self.w, X[i]) + self.b) # Calcule yᵢ·f(xᵢ) pour vérifier si le point est bien classé avec marge

                if cond >= 1:                                   # Cas 1 : point correctement classé et hors de la marge
                    self.w -= self.lr * self.w                  # Mise à jour : seule la régularisation L2 agit sur w
                else:                                           # Cas 2 : point mal classé ou dans la marge de tolérance
                    self.w -= self.lr * (self.w - self.C * y[i] * X[i]) # Mise à jour : régularisation + pénalité d'erreur
                    self.b += self.lr * self.C * y[i]          # Le biais est mis à jour uniquement en cas d'erreur
                    loss   += 1 - cond                          # Accumule la valeur de la Hinge Loss pour cet échantillon

            # Calcule et stocke la loss totale de l'epoch : terme de régularisation + terme d'erreur moyen
            self.losses.append(0.5 * np.dot(self.w, self.w) + self.C * loss / n)

        return self                                               # Retourne le modèle SVM entraîné

    def predict(self, X):
        return np.sign(X @ self.w + self.b)                     # Retourne le signe de la fonction de décision : +1 ou -1 pour chaque échantillon

# =============================================================================
# FONCTIONS UTILITAIRES : affichage métriques et graphiques
# =============================================================================
def afficher_regression(nom, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))          # Calcule la racine de l'erreur quadratique moyenne (pénalise les grosses erreurs)
    r2   = r2_score(y_true, y_pred)                             # Calcule le coefficient de détermination R² (1 = ajustement parfait)
    print(f"  {nom:<42} | RMSE: {rmse:.4f} | R²: {r2:.4f}")    # Affiche le nom du modèle et ses métriques formatées

def afficher_classification(nom, y_true, y_pred):
    y_true = y_true.astype(int) ; y_pred = y_pred.astype(int)  # Convertit les tableaux en entiers pour éviter les warnings de sklearn
    acc  = accuracy_score(y_true, y_pred)                       # Calcule la proportion de prédictions correctes
    TP   = np.sum((y_pred == 1) & (y_true == 1))                # Compte les vrais positifs (prédit 1, réel 1)
    TN   = np.sum((y_pred == 0) & (y_true == 0))                # Compte les vrais négatifs (prédit 0, réel 0)
    FP   = np.sum((y_pred == 1) & (y_true == 0))                # Compte les faux positifs (prédit 1, réel 0)
    FN   = np.sum((y_pred == 0) & (y_true == 1))                # Compte les faux négatifs (prédit 0, réel 1)
    prec = TP / (TP + FP + 1e-8)                                # Calcule la précision : TP / (TP + FP) + epsilon pour éviter div/0
    rec  = TP / (TP + FN + 1e-8)                                # Calcule le rappel : TP / (TP + FN) + epsilon
    f1   = 2 * prec * rec / (prec + rec + 1e-8)                 # Calcule le F1-Score : moyenne harmonique de Précision et Rappel
    print(f"  {nom:<42} | Acc: {acc:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | F1: {f1:.2f}") # Affiche les métriques de classification
    print(f"  {'  → Matrice confusion':<42}   TP={TP} FP={FP} | FN={FN} TN={TN}") # Affiche la matrice de confusion simplifiée

def plot_regression_simple(X, y, models_dict, titre):
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1) # Génère 100 points régulièrement espacés entre min et max pour tracer une courbe lisse
    plt.figure(figsize=(8, 5))                                  # Crée une nouvelle figure graphique de 8x5 pouces
    plt.scatter(X, y, color='black', zorder=5, label='Données réelles') # Trace les points observés en noir au premier plan
    for nom, modele in models_dict.items():                     # Parcours chaque modèle fourni dans le dictionnaire
        plt.plot(X_line, modele.predict(X_line), linewidth=2, label=nom) # Trace la courbe de prédiction du modèle
    plt.xlabel("Heures d'étude") ; plt.ylabel("Note")          # Nomme les axes X et Y
    plt.title(titre) ; plt.legend() ; plt.grid(True, alpha=0.3) # Ajoute titre, légende et grille semi-transparente
    plt.tight_layout() ; plt.show()                             # Ajuste les marges et affiche la fenêtre graphique

def plot_multiple(y_true, preds_dict, titre):
    plt.figure(figsize=(8, 5))                                  # Crée une figure dédiée à la comparaison multiple
    plt.plot(y_true, color='black', marker='o', label='Réel')  # Trace les valeurs réelles avec des marqueurs circulaires
    for nom, y_pred in preds_dict.items():                      # Parcours les prédictions de chaque modèle
        plt.plot(y_pred, linestyle='--', linewidth=2, label=nom) # Trace les prédictions en pointillés pour distinction visuelle
    plt.xlabel("Échantillon") ; plt.ylabel("Note")             # Nomme les axes
    plt.title(titre) ; plt.legend() ; plt.grid(True, alpha=0.3) # Configure titre, légende et grille
    plt.tight_layout() ; plt.show()                             # Affiche le graphique

def plot_decision_boundary(X, y, model_sc, model_sk, titre, svm=False):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))             # Crée une figure avec 2 sous-graphiques côte à côte
    noms = ["From Scratch", "Scikit-Learn"]                     # Définit les légendes pour chaque sous-graphique
    modeles = [model_sc, model_sk]                              # Regroupe les deux modèles à visualiser

    # Correction importante : on travaille dans l'espace d'entraînement du modèle
    X_plot = normaliser(X) if svm else X                        # Utilise les données normalisées si SVM, sinon données brutes
    
    for ax, modele, nom in zip(axes, modeles, noms):            # Boucle sur chaque axe, modèle et nom correspondant
        x0_min, x0_max = X_plot[:,0].min()-0.5, X_plot[:,0].max()+0.5 # Définit les limites de l'axe X avec une marge
        x1_min, x1_max = X_plot[:,1].min()-0.5, X_plot[:,1].max()+0.5 # Définit les limites de l'axe Y avec une marge
        xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, 200),
                             np.linspace(x1_min, x1_max, 200))  # Crée une grille dense de points pour évaluer la frontière
        grid = np.c_[xx.ravel(), yy.ravel()]                    # Aplatis la grille en liste de coordonnées (N, 2)

        Z = modele.predict(grid)                                # Prédit la classe pour chaque point de la grille
        Z = np.where(Z == -1, 0, 1) if nom == "From Scratch" and hasattr(modele, 'w') else Z # Harmonise les sorties SVM scratch (-1/1 → 0/1)
        Z = Z.reshape(xx.shape)                                 # Remet la forme 2D pour le tracé de contours

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')     # Remplit les zones de décision avec des couleurs semi-transparentes
        ax.scatter(X_plot[:,0], X_plot[:,1], c=y, cmap='coolwarm',
                   edgecolors='k', s=80, zorder=5)              # Superpose les points réels avec bordure noire
        ax.set_xlabel("Feature 1" + (" (norm)" if svm else "")) ; ax.set_ylabel("Feature 2" + (" (norm)" if svm else ""))
        ax.set_title(f"{titre} — {nom}") ; ax.grid(True, alpha=0.3) # Configure titres et grille

    plt.tight_layout() ; plt.show()                             # Ajuste et affiche la figure complète

def plot_svm_loss(losses, titre):
    plt.figure(figsize=(8, 4))                                  # Crée une figure pour la courbe de convergence
    plt.plot(losses, color='crimson', linewidth=2)               # Trace l'évolution de la loss au fil des itérations
    plt.xlabel("Itération") ; plt.ylabel("Hinge Loss")          # Nomme les axes
    plt.title(titre) ; plt.grid(True, alpha=0.3)                # Ajoute titre et grille
    plt.tight_layout() ; plt.show()                             # Affiche le graphique

# =============================================================================
# BLOC PRINCIPAL
# =============================================================================
if __name__ == "__main__":

    sep = "=" * 70                                              # Crée une ligne de séparation visuelle de 70 caractères
    print(sep)                                                  # Affiche la ligne de séparation en haut
    print("  PROJET : Régression & Classification — From Scratch & Scikit-Learn") # Affiche le titre du projet
    print(sep)                                                  # Affiche la ligne de séparation

    # -----------------------------------------------------------------------
    # 1. RÉGRESSION LINÉAIRE SIMPLE
    # -----------------------------------------------------------------------
    print("\n🔹 1. RÉGRESSION LINÉAIRE SIMPLE")                 # Annonce la section 1 dans la console
    print("   Dataset : heures d'étude → note\n")              # Décrit le jeu de données utilisé

    lin_sc = LinearRegressionScratch().fit(X_simple, y_simple)   # Entraîne le modèle from scratch sur toutes les données
    lin_sk = LinearRegression().fit(X_simple, y_simple)           # Entraîne le modèle Scikit-Learn sur les mêmes données

    afficher_regression("Linéaire Simple  (From Scratch)", y_simple, lin_sc.predict(X_simple)) # Évalue et affiche les métriques Scratch
    afficher_regression("Linéaire Simple  (Scikit-Learn)", y_simple, lin_sk.predict(X_simple)) # Évalue et affiche les métriques Sklearn
    plot_regression_simple(X_simple, y_simple,
                           {"Scratch": lin_sc, "Sklearn": lin_sk},
                           "Régression Linéaire Simple")        # Trace et affiche le graphique comparatif

    # -----------------------------------------------------------------------
    # 2. RÉGRESSION LINÉAIRE MULTIPLE
    # -----------------------------------------------------------------------
    print("\n🔹 2. RÉGRESSION LINÉAIRE MULTIPLE")               # Annonce la section 2
    print("   Dataset : heures d'étude + exercices → note\n")  # Décrit le dataset multiple

    mul_sc = LinearRegressionScratch().fit(X_multi, y_multi)      # Entraîne le modèle from scratch sur 2 features
    mul_sk = LinearRegression().fit(X_multi, y_multi)             # Entraîne le modèle Scikit-Learn sur 2 features

    afficher_regression("Linéaire Multiple (From Scratch)", y_multi, mul_sc.predict(X_multi)) # Évaluation Scratch
    afficher_regression("Linéaire Multiple (Scikit-Learn)", y_multi, mul_sk.predict(X_multi)) # Évaluation Sklearn
    plot_multiple(y_multi,
                  {"Scratch": mul_sc.predict(X_multi), "Sklearn": mul_sk.predict(X_multi)},
                  "Régression Linéaire Multiple — Réel vs Prédit") # Graphique comparatif multiple

    # -----------------------------------------------------------------------
    # 3. RÉGRESSION POLYNOMIALE (degré 2)
    # -----------------------------------------------------------------------
    print("\n🔹 3. RÉGRESSION POLYNOMIALE (Degré 2)")           # Annonce la section 3
    print("   Dataset : heures d'étude → note\n")              # Décrit le dataset

    DEGREE    = 2                                               # Fixe le degré polynomial à 2
    poly_sc   = PolynomialRegressionScratch(degree=DEGREE).fit(X_simple, y_simple) # Entraîne from scratch
    poly_feat = PolynomialFeatures(degree=DEGREE, include_bias=False) # Instancie le transformateur polynomial sklearn
    poly_sk   = LinearRegression().fit(poly_feat.fit_transform(X_simple), y_simple) # Entraîne la régression sur X transformé

    class PolyWrapper:                                            # Classe utilitaire pour uniformiser l'appel .predict()
        def __init__(self, feat, model):
            self.feat, self.model = feat, model                 # Stocke le transformateur et le modèle entraîné
        def predict(self, X):
            return self.model.predict(self.feat.transform(X))    # Applique transform() puis predict()

    poly_sk_w = PolyWrapper(poly_feat, poly_sk)                 # Instancie le wrapper pour interface uniforme
    afficher_regression("Polynomiale D2   (From Scratch)", y_simple, poly_sc.predict(X_simple)) # Évaluation Scratch
    afficher_regression("Polynomiale D2   (Scikit-Learn)", y_simple, poly_sk_w.predict(X_simple)) # Évaluation Sklearn
    plot_regression_simple(X_simple, y_simple,
                           {"Scratch D2": poly_sc, "Sklearn D2": poly_sk_w},
                           "Régression Polynomiale (Degré 2)")  # Graphique comparatif polynomial

    # -----------------------------------------------------------------------
    # 4. RÉGRESSION LOGISTIQUE
    # -----------------------------------------------------------------------
    print("\n🔹 4. RÉGRESSION LOGISTIQUE")                      # Annonce la section 4
    print("   Dataset : heures d'étude + exercices → admis (1) / refusé (0)\n") # Description dataset

    log_sc = LogisticRegressionScratch(lr=0.1, n_iter=1000).fit(X_clf, y_clf) # Entraîne from scratch avec 1000 itérations
    log_sk = LogisticRegression(max_iter=1000).fit(X_clf, y_clf)              # Entraîne Scikit-Learn (augmente max_iter par défaut)

    afficher_classification("Logistique      (From Scratch)", y_clf, log_sc.predict(X_clf)) # Métriques classification Scratch
    afficher_classification("Logistique      (Scikit-Learn)", y_clf, log_sk.predict(X_clf)) # Métriques classification Sklearn
    plot_decision_boundary(X_clf, y_clf, log_sc, log_sk, "Régression Logistique", svm=False) # Frontière de décision

    # -----------------------------------------------------------------------
    # 5. KNN — K-Nearest Neighbors  (K=3)
    # -----------------------------------------------------------------------
    print("\n🔹 5. KNN — K-NEAREST NEIGHBORS  (K=3)")          # Annonce la section 5
    print("   Dataset : heures d'étude + exercices → admis (1) / refusé (0)\n") # Description dataset

    K      = 3                                                  # Fixe le nombre de voisins à 3
    knn_sc = KNNScratch(k=K).fit(X_clf, y_clf)                   # Entraîne KNN from scratch (mémorisation)
    knn_sk = KNeighborsClassifier(n_neighbors=K).fit(X_clf, y_clf) # Entraîne KNN Scikit-Learn

    afficher_classification("KNN k=3         (From Scratch)", y_clf, knn_sc.predict(X_clf)) # Métriques Scratch
    afficher_classification("KNN k=3         (Scikit-Learn)", y_clf, knn_sk.predict(X_clf)) # Métriques Sklearn
    plot_decision_boundary(X_clf, y_clf, knn_sc, knn_sk, "KNN (K=3)", svm=False) # Frontière KNN

    # -----------------------------------------------------------------------
    # 6. SVM — Support Vector Machine
    # -----------------------------------------------------------------------
    print("\n🔹 6. SVM — SUPPORT VECTOR MACHINE")              # Annonce la section 6
    print("   Dataset : heures d'étude + exercices → admis (1) / refusé (0)") # Description dataset
    print("   Note : normalisation min-max appliquée (nécessaire pour SVM)\n") # Précision technique

    svm_sc = SVMScratch(lr=0.001, C=1.0, n_iter=1000).fit(X_clf_n, y_svm_labels) # Entraîne SVM from scratch sur données normalisées
    svm_sk = SVC(kernel='linear', C=1.0).fit(X_clf_n, y_svm_labels)              # Entraîne SVM Scikit-Learn (noyau linéaire)

    pred_sc_01 = np.where(svm_sc.predict(X_clf_n) == -1, 0, 1)  # Convertit les sorties -1/+1 du scratch en 0/1
    pred_sk_01 = np.where(svm_sk.predict(X_clf_n) == -1, 0, 1)  # Convertit les sorties -1/+1 de sklearn en 0/1

    afficher_classification("SVM             (From Scratch)", y_clf, pred_sc_01) # Évaluation Scratch
    afficher_classification("SVM             (Scikit-Learn)", y_clf, pred_sk_01) # Évaluation Sklearn

    print(f"\n  Vecteur w (Scratch) : {svm_sc.w}")             # Affiche le vecteur normal appris par le SVM scratch
    print(f"  Biais   b (Scratch) : {svm_sc.b:.4f}")           # Affiche le biais appris

    plot_svm_loss(svm_sc.losses, "SVM From Scratch — Convergence de la Hinge Loss") # Trace la courbe de convergence

    # Wrappers pour unifier l'interface predict() dans plot_decision_boundary
    class _SVMsc:
        def __init__(self, m): self.m = m                      # Stocke le modèle SVM from scratch
        def predict(self, X): return np.where(self.m.predict(X) == -1, 0, 1) # Retourne 0/1 au lieu de -1/1

    class _SVMsk:
        def __init__(self, m): self.m = m                      # Stocke le modèle SVM sklearn
        def predict(self, X): return np.where(self.m.predict(X) == -1, 0, 1) # Retourne 0/1 au lieu de -1/1

    plot_decision_boundary(X_clf, y_clf, _SVMsc(svm_sc), _SVMsk(svm_sk), "SVM", svm=True) # Frontière SVM (avec gestion normalisation)

    print(f"\n{sep}")                                           # Affiche ligne de séparation finale
    print("  Exécution terminée — tous les modèles entraînés et visualisés.") # Message de succès
    print(sep)                                                  # Ferme la ligne de séparation
