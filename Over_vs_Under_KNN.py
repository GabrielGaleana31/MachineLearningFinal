import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Leer los datos
df = pd.read_csv("star_classification.csv")

# Eliminar columnas no deseadas
columnas_a_eliminar = ["rerun_ID", "obj_ID"]
df = df.drop(columnas_a_eliminar, axis=1)

# Separar las características y la variable objetivo
X = df.drop('class', axis=1)
y = df['class']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define el espacio de búsqueda de hiperparámetros para KNN.
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
    'p': [1, 2]
}

# Crea un objeto de validación cruzada estratificada.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inicializa listas para almacenar sesgo y varianza para cada modelo.
biases_p1 = []
variances_p1 = []
biases_p2 = []
variances_p2 = []

for n_neighbors in param_grid['n_neighbors']:
    for p in param_grid['p']:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)

        # Realiza validación cruzada para cada configuración de hiperparámetros.
        biases_fold = []
        variances_fold = []

        for train_idx, test_idx in cv.split(X_train, y_train):
            X_train_val, y_train_val = X_train[train_idx], y_train[train_idx]
            X_test_val, y_test_val = X_train[test_idx], y_train[test_idx]

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

# Crea gráficos para sesgo y varianza para p=1 y p=2.
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(param_grid['n_neighbors'], biases_p1, marker='o', label='Bias^2 (p=1)')
plt.plot(param_grid['n_neighbors'], variances_p1, marker='o', label='Variance (p=1)')
plt.xlabel('Número de Vecinos')
plt.ylabel('Valor')
plt.title('Bias^2 y Variance para p=1')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(param_grid['n_neighbors'], biases_p2, marker='o', label='Bias^2(p=2)')
plt.plot(param_grid['n_neighbors'], variances_p2, marker='o', label='Variance(p=2)')
plt.xlabel('Número de Vecinos')
plt.ylabel('Valor')
plt.title('Sesgo al Cuadrado y Varianza para p=2')
plt.legend()

plt.tight_layout()
plt.show()
