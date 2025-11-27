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

## Feuille de Route des Algorithmes

Voici la liste des algorithmes à implémenter, classés par catégorie et par ordre de difficulté croissant. Il est recommandé de suivre cet ordre pour construire ses connaissances de manière progressive.

### Légende de Difficulté
- 🟢 **Fondamental** : Les bases du ML, incontournables.
- 🟡 **Essentiel** : Algorithmes courants et très utiles.
- 🟠 **Avancé** : Concepts plus complexes, souvent plus performants.
- 🔴 **Expert** : Implémentations difficiles, excellentes pour briller en entretien.

---

### 📈 01 - Apprentissage Supervisé : Régression
*Prédire une valeur continue.*

- [X] 🟢 **Régression Linéaire** : Le "Hello World" du ML. Descente de gradient.
- [ ] 🟡 **Régression Polynomiale** : Gérer la non-linéarité en ajoutant des features.
- [ ] 🟡 **Ridge & Lasso Regression** : Comprendre la régularisation pour éviter l'overfitting.

---

### 🏷️ 02 - Apprentissage Supervisé : Classification
*Prédire une catégorie discrète.*

- [ ] 🟢 **Régression Logistique** : La base de la classification binaire.
- [ ] 🟢 **K-Plus Proches Voisins (KNN)** : Algorithme simple basé sur la distance.
- [ ] 🟡 **Naïve Bayes** : Classifieur probabiliste rapide et efficace.
- [ ] 🟡 **Arbre de Décision** : Le bloc de construction des modèles ensemblistes.
- [ ] 🟠 **Support Vector Machines (SVM)** : Classifieur puissant basé sur la notion de marge maximale.

---

### 🔍 03 - Apprentissage Non-Supervisé
*Explorer la structure des données sans étiquettes.*

- [ ] 🟡 **K-Moyennes (K-Means)** : L'algorithme de clustering le plus célèbre.
- [ ] 🟠 **Analyse en Composantes Principales (PCA)** : La méthode de référence pour la réduction de dimension.
- [ ] 🟠 **Gaussian Mixture Models (GMM)** : Modèle de clustering probabiliste plus flexible que K-Means.

---

### 🧠 04 - Deep Learning Fondamentaux
*Construire des réseaux de neurones from scratch.*

- [ ] 🟡 **Le Perceptron** : Le neurone artificiel de base.
- [ ] 🟡 **Fonctions d'Activation** : Implémenter Sigmoid, ReLU, Tanh.
- [ ] 🟡 **Fonctions de Perte** : Implémenter MSE, Cross-Entropy.
- [ ] 🟠 **Multi-Layer Perceptron (MLP)** : Assembler le tout pour créer un réseau de neurones simple (réseau "dense").
- [ ] 🟠 **Backpropagation** : L'algorithme d'optimisation au cœur de l'entraînement des réseaux de neurones.
- [ ] 🔴 **Gradient Boosting (Machine)** : Un des algorithmes les plus puissants pour les données tabulaires.

---