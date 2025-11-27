import numpy as np

def mean_squared_error(y_true, y_pred):
    """Calcule l'erreur quadratique moyenne (MSE)."""
    return np.mean((y_true - y_pred)**2)

def r2_score(y_true, y_pred):
    """Calcule le coefficient de détermination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def accuracy(y_true, y_pred):
    """Calcule l'accuracy pour la classification."""
    return np.sum(y_true == y_pred) / len(y_true)
