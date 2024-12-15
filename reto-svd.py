import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, datasets
from scipy.linalg import svd

# Cargar la imagen en escala de grises y color
A = datasets.face(gray=True)
B = datasets.face(gray=False)
# Mostrar la imagen en escala de grises y color
plt.imshow(A, cmap=plt.cm.gray)
plt.show()
plt.imshow(B)
plt.show()
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
    # Paso 1: Aplicar SVD
    U, S, Vt = svd_descomposicion(A)
    
    # Paso 2: Reducir la cantidad de datos
    U_reducido = U[:, :n_comp]
    S_reducido = S[:n_comp, :n_comp]
    Vt_reducido = Vt[:n_comp, :]
    
    # Paso 3: Reconstruir la imagen comprimida
    A_hat = reconstruction(U_reducido, S_reducido, Vt_reducido)
    
    # Paso 4: Calcular el error SSE
    sse = sse_score(A, A_hat)
    
    # Paso 5: Devolver la imagen comprimida y el error SSE
    return A_hat, sse

# Ejemplo
racoon_gray = datasets.face(gray=True)
racoon_hat, sse = image_compression(racoon_gray, n_comp=50)
print(f"Error de reconstrucción: {sse}")
plt.imshow(racoon_hat_gray, cmap=plt.cm.gray)
plt.show()

racoon_color =datasets.face(gray=False)
racoon_hat_color, sse = image_compression(racoon_color, n_comp=50)
plt.imshow(racoon_hat_color)
plt.show()

  

