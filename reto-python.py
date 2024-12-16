import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, datasets
from scipy.linalg import svd

# Función SSE
def sse_score(X, X_hat):
    return np.sum((X - X_hat) ** 2)

# Función SVD
def svd_descomposicion(X):
    U, S, Vt = svd(X, full_matrices=False)
    return U, np.diag(S), Vt

# Función para reconstruir la imagen
def reconstruction(U, S, Vt):
    return np.dot(U, np.dot(S, Vt))

# Función para comprimir la imagen
def image_compression(A, n_comp):
    if len(A.shape) == 2:  # Imagen en escala de grises
        U, S, Vt = svd_descomposicion(A)
        U_reducido = U[:, :n_comp]
        S_reducido = S[:n_comp, :n_comp]
        Vt_reducido = Vt[:n_comp, :]
        A_hat = reconstruction(U_reducido, S_reducido, Vt_reducido)
        sse = sse_score(A, A_hat)
    elif len(A.shape) == 3:  # Imagen en colores
        A_hat = np.zeros_like(A)
        sse = 0
        for i in range(3):  # Aplicar SVD a cada canal
            U, S, Vt = svd_descomposicion(A[:, :, i])
            U_reducido = U[:, :n_comp]
            S_reducido = S[:n_comp, :n_comp]
            Vt_reducido = Vt[:n_comp, :]
            A_hat[:, :, i] = reconstruction(U
