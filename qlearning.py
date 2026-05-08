import numpy as np                                               # Importe NumPy pour calculs et tableaux
import matplotlib.pyplot as plt                                  # Importe Matplotlib pour le graphique

# =============================================================================
# PARAMÈTRES DE DÉCROISSANCE (EPSILON-GREEDY)
# =============================================================================
epsilon_start = 1.0                                              # Taux d'exploration initial (100% aléatoire)
epsilon_end = 0.08                                               # Taux minimal final (8% exploration résiduelle)
epsilon_decay = 0.995                                            # Facteur de décroissance exponentielle
n_episodes = 500                                                 # Nombre total d'épisodes d'entraînement

# =============================================================================
# CALCUL DE LA COURBE
# =============================================================================
epsilon_history = []                                             # Liste pour stocker ε à chaque épisode
epsilon = epsilon_start                                          # Initialise ε à sa valeur de départ

for _ in range(n_episodes):                                      # Boucle sur les 500 épisodes
    epsilon_history.append(epsilon)                              # Enregistre la valeur courante
    epsilon = max(epsilon_end, epsilon * epsilon_decay)          # Applique decay : ε ← max(ε_min, ε × 0.995)

# =============================================================================
# TRACÉ DU GRAPHIQUE
# =============================================================================
plt.figure(figsize=(9, 6))                                       # Crée une figure de 9x6 pouces
plt.plot(epsilon_history, color='coral', linewidth=2.5)          # Trace la courbe de décroissance en orange
plt.title("Q-Learning — Décroissance de l'Exploration (ε-greedy)",  # Titre en gras
          fontsize=14, fontweight='bold')
plt.xlabel("Épisode", fontsize=12, fontweight='bold')            # Axe horizontal
plt.ylabel("Epsilon (taux d'exploration)", fontsize=12, fontweight='bold') # Axe vertical
plt.grid(True, alpha=0.3)                                        # Grille semi-transparente
plt.tight_layout()                                               # Ajuste les marges automatiquement
plt.show()                                                       # Affiche la fenêtre graphique