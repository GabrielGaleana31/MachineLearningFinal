#----------------------------------------------------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold
import time
#----------------------------------------------------------------------------------------------------------------------------------------#

inicio = time.time()
#----------------------------------------------------------------------------------------------------------------------------------------#

# Leer los datos
df = pd.read_csv("star_classification.csv")


# Eliminar columnas no deseadas
columnas_a_eliminar = ["rerun_ID", "obj_ID"]
df = df.drop(columnas_a_eliminar, axis=1)


# Codificar la columna 'class' en valores numéricos
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# Separar las características y la variable objetivo
X = df.drop('class', axis=1)
y = df['class']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#----------------------------------------------------------------------------------------------------------------------------------------#

param_grid = {
    'n_neighbors': [1, 21],
    'p': [1, 2]
}

num_splits = 5

cv = StratifiedKFold(n_splits=5, shuffle=True)


biases_p1 = []
variances_p1 = []
biases_p2 = []
variances_p2 = []
#----------------------------------------------------------------------------------------------------------------------------------------#

for n_neighbors in param_grid['n_neighbors']:
    for p in param_grid['p']:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)

        for train_idx, test_idx in cv.split(X_train, y_train):
            X_train_val, y_train_val = X_train[train_idx], y_train.iloc[train_idx]  # Asegura que y_train_val sea un vector
            X_test_val, y_test_val = X_train[test_idx], y_train.iloc[test_idx]  # Asegura que y_test_val sea un vector

            knn.fit(X_train_val, y_train_val)
            predictions = knn.predict(X_test_val)

            # Calcula el sesgo y la varianza para el pliegue actual.
            bias_fold = (np.mean(predictions == y_test_val) - np.mean(y_test_val))**2  # Eleva al cuadrado el sesgo
            variance_fold = np.var(predictions == y_test_val)

            if p == 1:
                biases_p1.append(bias_fold)
                variances_p1.append(variance_fold)
            elif p == 2:
                biases_p2.append(bias_fold)
                variances_p2.append(variance_fold)

fin = time.time()
tiempo_transcurrido = fin - inicio
#----------------------------------------------------------------------------------------------------------------------------------------#

matriz_biases_p1 = np.array(biases_p1).reshape(len(param_grid['n_neighbors']), num_splits)
promedio_intervalos_biases_p1 = np.mean(matriz_biases_p1, axis=1)

matriz_variances_p1 = np.array(variances_p1).reshape(len(param_grid['n_neighbors']), num_splits)
promedio_intervalos_variances_p1 = np.mean(matriz_variances_p1, axis=1)

matriz_biases_p2 = np.array(biases_p2).reshape(len(param_grid['n_neighbors']), num_splits)
promedio_intervalos_biases_p2 = np.mean(matriz_biases_p2, axis=1)

matriz_variances_p2 = np.array(variances_p2).reshape(len(param_grid['n_neighbors']), num_splits)
promedio_intervalos_variances_p2 = np.mean(matriz_variances_p2, axis=1)

print(promedio_intervalos_biases_p1)
print(promedio_intervalos_variances_p1)
print(promedio_intervalos_biases_p2)
print(promedio_intervalos_variances_p2)