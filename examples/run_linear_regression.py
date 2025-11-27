import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ajoute le dossier 'src' au path Python pour pouvoir importer notre package
# Cela permet de faire `from mlfs...` peu importe d'où le script est lancé.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from mlfs.supervised.regression.linear_regression import LinearRegression
from mlfs.utils.metrics import mean_squared_error, r2_score

def generate_linear_data(n_samples=100, noise=15):
    """Génère des données de régression linéaire synthétiques."""
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    y = 5 + 3 * X + np.random.randn(n_samples, 1) * noise
    return X, y

if __name__ == "__main__":
    print("--- Exemple : Régression Linéaire From Scratch ---")
    
    # 1. Générer des données
    X, y = generate_linear_data()
    
    # 2. Instancier et entraîner notre modèle
    model = LinearRegression(learning_rate=0.05, n_iterations=500)
    model.fit(X, y)
    
    # 3. Faire des prédictions
    y_pred = model.predict(X)
    
    # 4. Évaluer le modèle
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Erreur Quadratique Moyenne (MSE): {mse:.2f}")
    print(f"Coefficient de Détermination (R²): {r2:.2f}")
    print(f"Poids appris (w): {model.weights[0][0]:.2f}")
    print(f"Biais appris (b): {model.bias:.2f}")
    
    # 5. Visualiser les résultats
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Données réelles')
    plt.plot(X, y_pred, color='red', linewidth=3, label='Notre modèle de régression')
    plt.xlabel('Feature (X)')
    plt.ylabel('Cible (y)')
    plt.title('Régression Linéaire From Scratch')
    plt.legend()
    plt.grid(True)
    plt.show()
