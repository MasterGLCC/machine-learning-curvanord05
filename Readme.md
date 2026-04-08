 Régression Linéaire, Multiple et Polynomiale
 
> Projet Machine Learning — Implémentation From Scratch & avec Bibliothèques
 

 
## Description
 
Ce projet implémente trois types de régression en Python :
 
- **Régression Linéaire Simple** — prédire une valeur à partir d'une seule variable
- **Régression Linéaire Multiple** — prédire une valeur à partir de plusieurs variables
- **Régression Polynomiale** — modéliser des relations non-linéaires avec des courbes
 
Chaque modèle est développé de **deux façons** :
1. **From Scratch** — implémentation manuelle avec NumPy (équation normale)
2. **Avec Scikit-Learn** — utilisation de la bibliothèque officielle
 
 ## Concepts utilisés
 
| Modèle | Méthode | Description |
|--------|---------|-------------|
| Linéaire Simple | Équation normale `θ = (XᵀX)⁻¹Xᵀy` | 1 variable → 1 droite |
| Linéaire Multiple | Même équation étendue | N variables → hyperplan |
| Polynomiale | Transformation `X → [X, X², X³]` + régression linéaire | 1 variable → courbe |
 
---
 
## Métriques d'évaluation
 
- **RMSE** *(Root Mean Squared Error)* — erreur moyenne de prédiction (plus bas = mieux)
- **R²** *(Coefficient de détermination)* — proportion de variance expliquée (plus proche de 1 = mieux)
 
---
 
## Bibliothèques requises
 
```bash
pip install numpy matplotlib scikit-learn
```
 
---
 
## Exécution
 
```bash
python regression.py
```
 
Le script affiche les métriques dans la console et génère un graphique comparatif des modèles.
 
---
 
## Résultats attendus
 
```
🔹 1. RÉGRESSION LINÉAIRE SIMPLE (1 feature)
  Linéaire (From Scratch)             | RMSE: ...  | R²: ...
  Linéaire (Scikit-Learn)             | RMSE: ...  | R²: ...
 
🔹 2. RÉGRESSION MULTIPLE (3 features)
  Multiple (From Scratch)             | RMSE: ...  | R²: ...
  Multiple (Scikit-Learn)             | RMSE: ...  | R²: ...
 
🔹 3. RÉGRESSION POLYNOMIALE (Degré 3)
  Polynomiale D3 (From Scratch)       | RMSE: ...  | R²: ...
  Polynomiale D3 (Scikit-Learn)       | RMSE: ...  | R²: ...


