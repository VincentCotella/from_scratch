import sys
import os
import numpy as np

# Ajoute le dossier 'src' au path Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from mlfs.supervised.regression.linear_regression import LinearRegression

def test_linear_regression_initialization():
    """Test que le modèle s'initialise correctement."""
    model = LinearRegression()
    assert model.weights is None
    assert model.bias is None
    print("✅ Test d'initialisation réussi.")

def test_linear_regression_fit():
    """Test que le modèle s'entraîne et que les poids sont définis."""
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([[2], [4], [6], [8]])
    
    model = LinearRegression(learning_rate=0.1, n_iterations=100)
    model.fit(X_train, y_train)
    
    assert model.weights is not None
    assert model.bias is not None
    assert len(model.loss_history) == 100
    # Vérifie que la perte a diminué
    assert model.loss_history[-1] < model.loss_history[0]
    print("✅ Test d'entraînement (fit) réussi.")

def test_linear_regression_predict():
    """Test que les prédictions ont la bonne forme."""
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([[2], [4], [6], [8]])
    X_test = np.array([[5], [6]])
    
    model = LinearRegression(learning_rate=0.1, n_iterations=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    assert predictions.shape == (2, 1)
    print("✅ Test de prédiction (predict) réussi.")

if __name__ == '__main__':
    test_linear_regression_initialization()
    test_linear_regression_fit()
    test_linear_regression_predict()
    print("\n🎉 Tous les tests pour la régression linéaire ont passé !")
