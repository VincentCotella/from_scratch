import numpy as np

class LinearRegression:
    """
    Implémentation de la Régression Linéaire from scratch en utilisant la descente de gradient.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """Entraîne le modèle sur les données X (features) et y (cible)."""
        n_samples, n_features = X.shape
        
        # 1. Initialiser les paramètres
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 2. Descente de gradient
        for i in range(self.n_iterations):
            # Modèle linéaire (prédictions)
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Calcul de la perte (MSE)
            loss = np.mean((y_predicted - y)**2)
            self.loss_history.append(loss)
            
            # Calcul des gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Mise à jour des poids et du biais
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        print(f"Entraînement terminé. Perte finale: {loss:.4f}")

    def predict(self, X):
        """Prédit les valeurs pour X."""
        return np.dot(X, self.weights) + self.bias
