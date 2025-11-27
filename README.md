# ML From Scratch

Ce projet contient des implémentations d'algorithmes de Machine Learning et de Deep Learning codés "from scratch" en Python.

## Objectif
Maîtriser en profondeur le fonctionnement interne des algorithmes et développer une solide compréhension des fondamentaux.

## Structure du Projet
- `src/mlfs/` : Le package Python principal contenant tout le code source.
  - `utils/` : Fonctions utilitaires partagées (métriques, manipulation de données).
  - `supervised/` : Algorithmes d'apprentissage supervisé (régression, classification).
  - `unsupervised/` : Algorithmes d'apprentissage non-supervisé (clustering, réduction de dimension).
  - `deep_learning/` : Bases pour construire des réseaux de neurones.
- `examples/` : Scripts exécutables montrant comment utiliser les implémentations.
- `tests/` : Tests unitaires pour valider le fonctionnement du code.
- `notebooks/` : Notebooks pour l'exploration, la visualisation et l'expérimentation.
- `data/` : (Optionnel) Pour stocker des jeux de données.

## Pour Commencer
1.  Créer et activer un environnement virtuel :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```
2.  Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```
3.  Lancer un exemple :
    ```bash
    python examples/run_linear_regression.py
    ```

