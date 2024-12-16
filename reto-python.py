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
# Cargar la imagen en colores
B = datasets.face(gray=False)

# Crear subplots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Mostrar la imagen original en escala de grises
axes[0, 0].imshow(A, cmap=plt.cm.gray)
axes[0, 0].set_title('Original en Escala de Grises')
axes[0, 0].axis('off')

# Mostrar la imagen original en colores
axes[1, 0].imshow(B)
axes[1, 0].set_title('Original en Colores')
axes[1, 0].axis('off')

# Comprimir y mostrar la imagen en escala de grises con 50 componentes
racoon_hat_gray_50, sse_gray_50 = image_compression(A, n_comp=50)
axes[0, 1].imshow(racoon_hat_gray_50, cmap=plt.cm.gray)
axes[0, 1].set_title('Grises con 50 Componentes')
axes[0, 1].axis('off')

# Comprimir y mostrar la imagen en colores con 50 componentes
racoon_hat_color_50, sse_color_50 = image_compression(B, n_comp=50)
axes[1, 1].imshow(racoon_hat_color_50)
axes[1, 1].set_title('Colores con 50 Componentes')
axes[1, 1].axis('off')

# Comprimir y mostrar la imagen en escala de grises con 25 componentes
racoon_hat_gray_25, sse_gray_25 = image_compression(A, n_comp=25)
axes[0, 2].imshow(racoon_hat_gray_25, cmap=plt.cm.gray)
axes[0, 2].set_title('Grises con 25 Componentes')
axes[0, 2].axis('off')

# Comprimir y mostrar la imagen en colores con 25 componentes
racoon_hat_color_25, sse_color_25 = image_compression(B, n_comp=25)
axes[1, 2].imshow(racoon_hat_color_25)
axes[1, 2].set_title('Colores con 25 Componentes')
axes[1, 2].axis('off')

# Comprimir y mostrar la imagen en escala de grises con 10 componentes
racoon_hat_gray_10, sse_gray_10 = image_compression(A, n_comp=10)
axes[0, 3].imshow(racoon_hat_gray_10, cmap=plt.cm.gray)
axes[0, 3].set_title('Grises con 10 Componentes')
axes[0, 3].axis('off')

# Comprimir y mostrar la imagen en colores con 10 componentes
racoon_hat_color_10, sse_color_10 = image_compression(B, n_comp=10)
axes[1, 3].imshow(racoon_hat_color_10)
axes[1, 3].set_title('Colores con 10 Componentes')
axes[1, 3].axis('off')

plt.tight_layout()
plt.show()

