pip install notebook
juoyter notebook
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
            A_hat[:, :, i] = reconstruction(U_reducido, S_reducido, Vt_reducido)
            sse += sse_score(A[:, :, i], A_hat[:, :, i])
    else:
        raise ValueError("La imagen debe ser en escala de grises o en colores.")
    
    return A_hat, sse

# Cargar la imagen en escala de grises
A = datasets.face(gray=True)

# Mostrar la imagen original en escala de grises
plt.imshow(A, cmap=plt.cm.gray)
plt.title('Original en Escala de Grises')
plt.show()

# Cargar la imagen en colores
B = datasets.face(gray=False)

# Mostrar la imagen original en colores
plt.imshow(B)
plt.title('Original en Colores')
plt.show()

# Comprimir la imagen en escala de grises con 50 componentes
racoon_hat_gray, sse_gray = image_compression(A, n_comp=50)
print(f"Error de reconstrucción (escala de grises): {sse_gray}")

# Mostrar la imagen comprimida en escala de grises
plt.imshow(racoon_hat_gray, cmap=plt.cm.gray)
plt.title('Comprimida con 50 Componentes (Escala de Grises)')
plt.show()

# Comprimir la imagen en colores con 50 componentes
racoon_hat_color, sse_color = image_compression(B, n_comp=50)
print(f"Error de reconstrucción (colores): {sse_color}")

# Mostrar la imagen comprimida en colores
plt.imshow(racoon_hat_color)
plt.title('Comprimida con 50 Componentes (Colores)')
plt.show()

