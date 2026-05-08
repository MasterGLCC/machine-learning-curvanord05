# =============================================================================
# FICHIER SÉPARÉ : Algorithme DBSCAN (Density-Based Spatial Clustering)
# Objectif : Clustering non supervisé avec visualisations complètes
# Note : DBSCAN est un algorithme de clustering basé sur la densité.
#        Il ne nécessite pas de spécifier le nombre de clusters à l'avance.
# =============================================================================

import numpy as np                                               # Importe NumPy pour calculs matriciels et génération de données
import matplotlib.pyplot as plt                                  # Importe Matplotlib pour visualisations 2D/3D
from sklearn.cluster import DBSCAN                               # Importe l'algorithme DBSCAN de Scikit-Learn
from sklearn.datasets import make_moons, make_circles            # Importe générateurs de datasets non linéaires
from sklearn.metrics import silhouette_score, davies_bouldin_score # Importe métriques de qualité de clustering
from sklearn.preprocessing import StandardScaler                 # Importe standardisation (recommandé pour DBSCAN)
import warnings                                                  # Importe gestion des avertissements
warnings.filterwarnings('ignore')                                # Désactive les warnings pour console propre

# =============================================================================
#  GÉNÉRATION DE DATASETS DE TEST (Formes complexes pour DBSCAN)
# =============================================================================

# Dataset 1 : Forme de lunes (non linéairement séparable)
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)  # Génère 2 demi-lunes entrelacées avec bruit

# Dataset 2 : Cercles concentriques (défi pour K-Means, facile pour DBSCAN)
X_circles, _ = make_circles(n_samples=400, noise=0.05, factor=0.5, random_state=42)  # 2 cercles imbriqués

# Dataset 3 : Données artificielles personnalisées (clusters de densités variables)
np.random.seed(42)                                                   # Fixe graine aléatoire pour reproductibilité
X_custom = np.vstack([                                               # Combine verticalement 4 clusters
    np.random.randn(100, 2) * 0.5 + np.array([0, 0]),               # Cluster 1 : centre (0,0), densité élevée
    np.random.randn(80, 2) * 0.3 + np.array([3, 3]),                # Cluster 2 : centre (3,3), très dense
    np.random.randn(60, 2) * 0.8 + np.array([-3, 2]),               # Cluster 3 : centre (-3,2), dispersé
    np.random.uniform(-4, 4, (40, 2))                               # Bruit uniforme (outliers)
])

# =============================================================================
#  FONCTION UTILITAIRE : Standardisation des données
# =============================================================================
def standardiser(X):
    """Normalise les données (moyenne=0, écart-type=1) - recommandé pour DBSCAN"""
    scaler = StandardScaler()                                        # Instancie le standardiseur
    return scaler.fit_transform(X)                                   # Applique fit_transform et retourne résultat

# Standardisation des 3 datasets
X_moons_s = standardiser(X_moons)                                    # Lune standardisée
X_circles_s = standardiser(X_circles)                                # Cercles standardisés
X_custom_s = standardiser(X_custom)                                  # Dataset custom standardisé

# =============================================================================
#  CONFIGURATION DES HYPERPARAMÈTRES DBSCAN
# =============================================================================
EPSILON = 0.3                                                        # Rayon de voisinage (distance maximale)
MIN_SAMPLES = 5                                                      # Nombre minimum de points pour former un cluster

# =============================================================================
#  ENTRAÎNEMENT DBSCAN SUR LES 3 DATASETS
# =============================================================================

# Dataset 1 : Lunes
dbscan_moons = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES)         # Instancie DBSCAN avec paramètres
labels_moons = dbscan_moons.fit_predict(X_moons_s)                   # Entraîne et prédit les labels de cluster

# Dataset 2 : Cercles
dbscan_circles = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES)       # Instancie DBSCAN
labels_circles = dbscan_circles.fit_predict(X_circles_s)             # Fit + predict sur cercles

# Dataset 3 : Custom
dbscan_custom = DBSCAN(eps=0.4, min_samples=MIN_SAMPLES)            # Epsilon légèrement plus grand pour custom
labels_custom = dbscan_custom.fit_predict(X_custom_s)                # Fit + predict sur données custom

# =============================================================================
#  CALCUL DES MÉTRIQUES DE QUALITÉ
# =============================================================================
def calculer_metriques(X, labels, nom_dataset):
    """Calcule et affiche les métriques de clustering"""
    # Filtre les points non bruit (labels != -1) pour silhouette score
    mask = labels != -1                                              # Crée masque booléen pour clusters valides
    if mask.sum() > 1 and len(set(labels[mask])) > 1:               # Vérifie qu'il y a au moins 2 clusters
        silhouette = silhouette_score(X[mask], labels[mask])         # Calcule coefficient de silhouette [-1, 1]
        davies_bouldin = davies_bouldin_score(X[mask], labels[mask]) # Calcule score Davies-Bouldin (plus bas = mieux)
    else:
        silhouette, davies_bouldin = np.nan, np.nan                 # Métriques non calculables si 1 seul cluster
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)      # Compte clusters (exclut bruit=-1)
    n_noise = list(labels).count(-1)                                 # Compte points étiquetés comme bruit
    
    # Affichage formaté
    print(f"\n{'='*70}")                                            # Séparateur
    print(f"  DATASET : {nom_dataset:<50}")                         # Nom du dataset
    print(f"{'='*70}")                                              # Séparateur
    print(f"  Nombre de clusters détectés : {n_clusters}")          # Affiche nombre de clusters
    print(f"  Points de bruit (outliers)  : {n_noise} ({n_noise/len(labels)*100:.1f}%)") # Affiche bruit
    print(f"  Silhouette Score            : {silhouette:.4f}")      # Qualité de séparation
    print(f"  Davies-Bouldin Score        : {davies_bouldin:.4f}")  # Compacité des clusters
    print(f"{'='*70}\n")                                            # Fermeture
    
    return n_clusters, n_noise, silhouette, davies_bouldin

# Calcul métriques pour les 3 datasets
metriques_moons = calculer_metriques(X_moons_s, labels_moons, "Lunes (Moons)")
metriques_circles = calculer_metriques(X_circles_s, labels_circles, "Cercles Concentriques")
metriques_custom = calculer_metriques(X_custom_s, labels_custom, "Données Personnalisées")

# =============================================================================
#  VISUALISATIONS PÉDAGOGIQUES (6 Graphiques)
# =============================================================================

# --- Graphique 1 : DBSCAN sur Lunes ---
plt.figure(figsize=(8, 6))                                           # Crée figure 8x6 pouces
plt.scatter(X_moons_s[:, 0], X_moons_s[:, 1], c=labels_moons, 
            cmap='viridis', s=40, edgecolors='black', alpha=0.7)    # Points colorés par cluster
plt.title(f"DBSCAN sur Lunes\nClusters: {metriques_moons[0]}, Bruit: {metriques_moons[1]}", 
          fontsize=12, fontweight='bold')                           # Titre avec stats
plt.xlabel("Feature 1 (standardisée)")                              # Axe X
plt.ylabel("Feature 2 (standardisée)")                              # Axe Y
plt.colorbar(label='Cluster ID (-1 = bruit)')                       # Légende de couleur
plt.grid(True, alpha=0.3)                                           # Grille discrète
plt.tight_layout()                                                  # Ajuste marges
plt.show()                                                          # Affiche graphique 1

# --- Graphique 2 : DBSCAN sur Cercles ---
plt.figure(figsize=(8, 6))                                           # Figure 8x6
plt.scatter(X_circles_s[:, 0], X_circles_s[:, 1], c=labels_circles, 
            cmap='coolwarm', s=40, edgecolors='black', alpha=0.7)   # Points avec colormap coolwarm
plt.title(f"DBSCAN sur Cercles\nClusters: {metriques_circles[0]}, Bruit: {metriques_circles[1]}", 
          fontsize=12, fontweight='bold')                           # Titre
plt.xlabel("Feature 1 (standardisée)")                              # Axe X
plt.ylabel("Feature 2 (standardisée)")                              # Axe Y
plt.colorbar(label='Cluster ID (-1 = bruit)')                       # Légende
plt.grid(True, alpha=0.3)                                           # Grille
plt.tight_layout()                                                  # Ajuste
plt.show()                                                          # Affiche graphique 2

# --- Graphique 3 : DBSCAN sur Données Custom ---
plt.figure(figsize=(8, 6))                                           # Figure 8x6
plt.scatter(X_custom_s[:, 0], X_custom_s[:, 1], c=labels_custom, 
            cmap='plasma', s=50, edgecolors='black', alpha=0.8)     # Points avec colormap plasma
plt.title(f"DBSCAN sur Données Custom\nClusters: {metriques_custom[0]}, Bruit: {metriques_custom[1]}", 
          fontsize=12, fontweight='bold')                           # Titre
plt.xlabel("Feature 1 (standardisée)")                              # Axe X
plt.ylabel("Feature 2 (standardisée)")                              # Axe Y
plt.colorbar(label='Cluster ID (-1 = bruit)')                       # Légende
plt.grid(True, alpha=0.3)                                           # Grille
plt.tight_layout()                                                  # Ajuste
plt.show()                                                          # Affiche graphique 3

# --- Graphique 4 : Comparaison des Métriques (Barres) ---
noms_datasets = ['Lunes', 'Cercles', 'Custom']                      # Noms pour légende
silhouettes = [metriques_moons[2], metriques_circles[2], metriques_custom[2]] # Scores silhouette
x_pos = np.arange(len(noms_datasets))                               # Positions X

plt.figure(figsize=(8, 5))                                           # Figure 8x5
plt.bar(x_pos, silhouettes, color=['steelblue', 'coral', 'seagreen'], 
        edgecolors='black', linewidth=1.2, alpha=0.8)               # Barres colorées
plt.axhline(0, color='black', linewidth=0.8)                        # Ligne zéro
plt.xticks(x_pos, noms_datasets)                                    # Labels X
plt.ylabel("Silhouette Score")                                      # Axe Y
plt.title("Comparaison de la Qualité de Clustering\n(Silhouette Score : plus élevé = mieux)", 
          fontweight='bold')                                        # Titre
plt.ylim([-0.2, 1.0])                                               # Limite Y pour visibilité
plt.grid(True, alpha=0.3, axis='y')                                 # Grille horizontale
plt.tight_layout()                                                  # Ajuste
plt.show()                                                          # Affiche graphique 4

# --- Graphique 5 : Impact de Epsilon sur le nombre de clusters (Dataset Lunes) ---
eps_values = np.arange(0.1, 0.6, 0.05)                              # Test epsilon de 0.1 à 0.55
n_clusters_list = []                                                 # Liste pour stocker résultats
n_noise_list = []                                                    # Liste pour bruit

for eps in eps_values:                                               # Boucle sur valeurs epsilon
    db_test = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)              # Crée DBSCAN avec epsilon courant
    labels_test = db_test.fit_predict(X_moons_s)                     # Fit + predict
    n_clust = len(set(labels_test)) - (1 if -1 in labels_test else 0) # Compte clusters
    n_noise = list(labels_test).count(-1)                            # Compte bruit
    n_clusters_list.append(n_clust)                                  # Stocke
    n_noise_list.append(n_noise)                                     # Stocke

plt.figure(figsize=(9, 5))                                           # Figure 9x5
plt.plot(eps_values, n_clusters_list, 'o-', color='blue', 
         linewidth=2, markersize=8, label='Nombre de clusters')     # Courbe clusters
plt.plot(eps_values, n_noise_list, 's--', color='red', 
         linewidth=2, markersize=8, label='Points de bruit')        # Courbe bruit
plt.xlabel("Valeur de Epsilon (rayon de voisinage)", fontweight='bold') # Axe X
plt.ylabel("Nombre", fontweight='bold')                             # Axe Y
plt.title("Sensibilité de DBSCAN à Epsilon\n(Dataset: Lunes)", 
          fontweight='bold')                                        # Titre
plt.legend()                                                        # Légende
plt.grid(True, alpha=0.3)                                           # Grille
plt.xticks(eps_values)                                              # Affiche tous les epsilon
plt.tight_layout()                                                  # Ajuste
plt.show()                                                          # Affiche graphique 5

# --- Graphique 6 : Visualisation 3D du Dataset Custom (optionnel mais pédagogique) ---
from mpl_toolkits.mplot3d import Axes3D                            # Importe toolkit 3D

# Ajoute une 3ème dimension artificielle pour visualisation
np.random.seed(42)                                                  # Graine reproductible
X_custom_3d = np.hstack([X_custom_s, np.random.randn(len(X_custom_s), 1) * 0.3]) # Ajoute feature Z

fig = plt.figure(figsize=(10, 7))                                   # Figure 10x7
ax = fig.add_subplot(111, projection='3d')                          # Crée subplot 3D
scatter = ax.scatter(X_custom_3d[:, 0], X_custom_3d[:, 1], X_custom_3d[:, 2], 
                     c=labels_custom, cmap='plasma', s=60, 
                     edgecolors='black', alpha=0.8)                 # Points 3D colorés
ax.set_title("DBSCAN - Vue 3D du Dataset Custom", fontsize=14, fontweight='bold') # Titre 3D
ax.set_xlabel("Feature 1")                                          # Axe X 3D
ax.set_ylabel("Feature 2")                                          # Axe Y 3D
ax.set_zlabel("Feature 3 (artificielle)")                          # Axe Z 3D
plt.colorbar(scatter, label='Cluster ID', shrink=0.6)              # Légende couleur
plt.tight_layout()                                                  # Ajuste
plt.show()                                                          # Affiche graphique 6

# =============================================================================
# RÉCAPITULATIF FINAL
# =============================================================================
print("\n" + "="*70)                                                # Séparateur final
print("  DBSCAN - EXÉCUTION TERMINÉE AVEC SUCCÈS")              # Message succès
print("="*70)                                                       # Séparateur
print(f"\n   Résumé des résultats :")                             # Titre résumé
print(f"  • Lunes        : {metriques_moons[0]} clusters détectés, {metriques_moons[1]} points de bruit") # Résumé moons
print(f"  • Cercles      : {metriques_circles[0]} clusters détectés, {metriques_circles[1]} points de bruit") # Résumé circles
print(f"  • Custom       : {metriques_custom[0]} clusters détectés, {metriques_custom[1]} points de bruit") # Résumé custom
print(f"\n   Points clés de DBSCAN :")                            # Section pédagogique
print(f"  ✓ Détecte automatiquement le nombre de clusters")        # Avantage 1
print(f" ✓ Gère les clusters de formes arbitraires (non convexes)") # Avantage 2
print(f"  ✓ Identifie les outliers (bruit)")                        # Avantage 3
print(f"  ✓ Ne nécessite pas de spécifier K à l'avance")           # Avantage 4
print(f"\n{'='*70}\n")                                              # Fermeture

print(" 6 graphiques générés avec succès !")                      # Confirmation visuelle