import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, datasets
from scipy.linalg import svd

# Cargar la imagen
A = datasets.face(gray=True)

# Mostrar la imagen A
plt.imshow(A, cmap=plt.cm.gray)
plt.show()

# Cargar la imagen en colores
B = datasets.face(gray=False)

# Mostrar la imagen
plt.imshow(B)
plt.show()

def sse_score(x-x_hat):
  return np.sum((x-x_hat)**2)

def svd_descomposicion(x):
  u,s,vt=svd(x,full_matrices=False)
  return u,np.diag(s),vt
  print(u)
  print(s)
  print(vt)
  

