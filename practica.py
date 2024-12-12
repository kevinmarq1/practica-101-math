# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Descargar el dataset de Auto MPG
auto_mpg = fetch_ucirepo(id=9)

# Cargar los datos en dataframes de pandas
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Mostrar la información del dataset
print(auto_mpg.metadata)
print(auto_mpg.variables)

# Ver las primeras filas del dataset
print(X.head())
print(y.head())

# Función para visualizar los datos
def visualiza(df, y, nombre_columna):
    plt.scatter(df[nombre_columna], y)
    plt.xlabel(nombre_columna)
    plt.ylabel('mpg')
    plt.title(f'mpg vs {nombre_columna}')
    plt.grid(True)
    plt.show()

# Visualizar algunas variables
visualiza(X, y, 'horsepower')
visualiza(X, y, 'acceleration')
visualiza(X, y, 'weight')
