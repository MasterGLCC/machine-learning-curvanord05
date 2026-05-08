import numpy as np                                               # Importe NumPy pour calculs matriciels et statistiques
import matplotlib.pyplot as plt                                  # Importe Matplotlib pour la visualisation graphique

# =============================================================================
#  DATASET MANUEL SIMPLE (2D → 1D via PCA)
# =============================================================================
# Données corrélées : 2 features avec relation linéaire + bruit
np.random.seed(42)                                               # Fixe la graine aléatoire pour reproductibilité
X = np.random.randn(100, 2) @ np.array([[2, 0.5], [0.5, 1]])    # Génère 100 points 2D corrélés via produit matriciel

# =============================================================================
#  PCA FROM SCRATCH — Implémentation manuelle étape par étape
# =============================================================================

# Étape 1 : Centrage des données (soustraire la moyenne de chaque feature)
X_mean = np.mean(X, axis=0)                                      # Calcule la moyenne de chaque colonne (feature)
X_centered = X - X_mean                                          # Soustrait la moyenne → données centrées en (0,0)

# Étape 2 : Matrice de covariance (mesure les corrélations entre features)
cov_matrix = np.cov(X_centered, rowvar=False)                   # Calcule covariance 2×2 (rowvar=False → colonnes = features)

# Étape 3 : Décomposition en valeurs/vecteurs propres (core mathématique du PCA)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)          # eigh() pour matrices symétriques → valeurs propres + vecteurs propres

# Étape 4 : Tri par ordre décroissant des valeurs propres (la plus grande = 1ère composante principale)
sorted_idx = np.argsort(eigenvalues)[::-1]                      # Trie les indices des valeurs propres du plus grand au plus petit
eigenvalues = eigenvalues[sorted_idx]                           # Réordonne les valeurs propres
eigenvectors = eigenvectors[:, sorted_idx]                      # Réordonne les vecteurs propres (colonnes)

# Étape 5 : Projection sur la 1ère composante principale (réduction 2D → 1D)
pc1 = eigenvectors[:, 0]                                         # Extrait le 1er vecteur propre (direction de variance max)
X_pca = X_centered @ pc1                                         # Produit matriciel : projection des points sur l'axe PC1

# =============================================================================
#  VISUALISATION UNIQUE : Données originales + Direction PCA + Projection
# =============================================================================
plt.figure(figsize=(9, 7))                                       # Crée une figure de 9x7 pouces

# 1. Nuage de points original (2D)
plt.scatter(X_centered[:, 0], X_centered[:, 1], 
            color='steelblue', alpha=0.6, s=40, label='Données centrées') # Points bleus semi-transparents

# 2. Flèche de la 1ère composante principale (direction de variance maximale)
plt.arrow(0, 0, pc1[0]*3, pc1[1]*3,                              # Dessine une flèche depuis l'origine
          head_width=0.3, head_length=0.4, fc='red', ec='red',  # Tête de flèche rouge
          linewidth=2.5, label='1ère Composante Principale (PC1)') # Légende

# 3. Projections des points sur PC1 (lignes pointillées vers l'axe principal)
for i in range(0, len(X_centered), 5):                          # Un point sur 5 pour éviter la surcharge visuelle
    plt.plot([X_centered[i, 0], X_pca[i]*pc1[0]],              # Ligne du point original...
             [X_centered[i, 1], X_pca[i]*pc1[1]],              # ...à sa projection sur PC1
             'r--', alpha=0.3, linewidth=0.8)                  # Trait rouge pointillé fin

# 4. Points projetés sur l'axe PC1 (visualisation de la réduction)
plt.scatter(X_pca * pc1[0], X_pca * pc1[1],                    # Coordonnées des projections
            color='coral', alpha=0.8, s=30, 
            edgecolors='black', label='Projections sur PC1')   # Points orange avec bordure

# 5. Décoration du graphique
plt.xlabel("Feature 1 (centrée)", fontsize=11)                  # Nomme l'axe horizontal
plt.ylabel("Feature 2 (centrée)", fontsize=11)                  # Nomme l'axe vertical
plt.title("PCA From Scratch — Réduction 2D → 1D\n(Variance expliquée par PC1 : {:.1f}%)".format(
          eigenvalues[0] / np.sum(eigenvalues) * 100),         # Affiche % de variance expliquée dans le titre
          fontsize=13, fontweight='bold')
plt.axhline(0, color='gray', linestyle=':', linewidth=0.5)     # Axe horizontal de référence
plt.axvline(0, color='gray', linestyle=':', linewidth=0.5)     # Axe vertical de référence
plt.legend(fontsize=10); plt.grid(True, alpha=0.3)             # Légende et grille discrète
plt.tight_layout(); plt.show()                                  # Ajuste les marges et affiche le graphique

# =============================================================================
#  AFFICHAGE DES RÉSULTATS NUMÉRIQUES DANS LA CONSOLE
# =============================================================================
print("="*70)                                                   # Ligne de séparation visuelle
print("  PCA FROM SCRATCH — Résultats")                         # Titre section console
print("="*70)                                                   # Séparateur
print(f"  Variance totale : {np.sum(eigenvalues):.4f}")        # Somme des variances (trace de la covariance)
print(f"  PC1 : {eigenvalues[0]:.4f} ({eigenvalues[0]/np.sum(eigenvalues)*100:.1f}% de la variance)") # Info 1ère composante
print(f"  PC2 : {eigenvalues[1]:.4f} ({eigenvalues[1]/np.sum(eigenvalues)*100:.1f}% de la variance)") # Info 2ème composante
print(f"  Direction PC1 : [{pc1[0]:.4f}, {pc1[1]:.4f}]")       # Vecteur directeur de la 1ère composante
print(f"  Données réduites : {X_pca.shape} (100 échantillons → 1 feature)") # Shape après réduction
print("="*70)                                                   # Fermeture visuelle
print("\n PCA terminé — 1 graphique généré avec succès.")    # Message de fin