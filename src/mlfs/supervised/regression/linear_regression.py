import numpy as np

class LinearRegression:
    """
    Implémentation de la Régression Linéaire from scratch en utilisant la descente de gradient.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialise le modèle.
        Args:
            learning_rate (float): Le pas d'apprentissage pour la descente de gradient.
            n_iterations (int): Le nombre d'itérations pour l'entraînement.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Entraîne le modèle sur les données X (features) et y (cible).
        L'algorithme utilisé est la descente de gradient pour minimiser l'erreur quadratique moyenne (MSE).
        """
        n_samples, n_features = X.shape

        # On s'assure que y est un vecteur colonne (n_samples, 1).
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # 1. Initialiser les paramètres (poids et biais) à zéro
        # On initialise les poids comme un vecteur colonne (n_features, 1)
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        # 2. Boucle d'entraînement (Descente de Gradient)
        for i in range(self.n_iterations):
            # a. Calculer les prédictions actuelles du modèle
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # b. Calculer l'erreur (perte) actuelle (MSE)
            loss = np.mean((y_predicted - y)**2)
            self.loss_history.append(loss)
            
            # c. Calculer les gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # d. Mettre à jour les poids et le biais
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        print(f"Entraînement terminé. Perte finale (MSE): {loss:.4f}")

    def predict(self, X):
        """
        Prédit les valeurs pour de nouvelles données X en utilisant les poids appris.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Le modèle n'a pas été entraîné. Appelez la méthode 'fit' d'abord.")
        return np.dot(X, self.weights) + self.bias