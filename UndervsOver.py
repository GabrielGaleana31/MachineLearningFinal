#ADVERTENCIA:
#Antes de correr, considerar que le toma 18.5 minutos aproximadamente dependiendo de las capacidades computacionales
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
df = df.head(500) #Demostrar underfitting

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
    'n_neighbors': [1, 3, 5, 7, 9,11, 13,15,17,19,21,23],
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
            bias_fold = (np.mean(predictions == y_test_val) - np.mean(y_test_val))**2 
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
#Sesgo y matrices

matriz_biases_p1 = np.array(biases_p1).reshape(len(param_grid['n_neighbors']), num_splits)
minimos_intervalos_biases_p1 = np.min(matriz_biases_p1, axis=1)
maximos_intervalos_biases_p1 = np.max(matriz_biases_p1, axis=1)
promedio_intervalos_biases_p1 = np.mean(matriz_biases_p1, axis=1)

matriz_variances_p1 = np.array(variances_p1).reshape(len(param_grid['n_neighbors']), num_splits)
minimos_intervalos_variances_p1 = np.min(matriz_variances_p1, axis=1)
maximos_intervalos_variances_p1 = np.max(matriz_variances_p1, axis=1)
promedio_intervalos_variances_p1 = np.mean(matriz_variances_p1, axis=1)
diff1 = abs(promedio_intervalos_biases_p1 - promedio_intervalos_variances_p1)
inter1 = param_grid['n_neighbors'][np.argmin(diff1)]

matriz_biases_p2 = np.array(biases_p2).reshape(len(param_grid['n_neighbors']), num_splits)
minimos_intervalos_biases_p2 = np.min(matriz_biases_p2, axis=1)
maximos_intervalos_biases_p2 = np.max(matriz_biases_p2, axis=1)
promedio_intervalos_biases_p2 = np.mean(matriz_biases_p2, axis=1)

matriz_variances_p2 = np.array(variances_p2).reshape(len(param_grid['n_neighbors']), num_splits)
minimos_intervalos_variances_p2 = np.min(matriz_variances_p2, axis=1)
maximos_intervalos_variances_p2 = np.max(matriz_variances_p2, axis=1)
promedio_intervalos_variances_p2 = np.mean(matriz_variances_p2, axis=1)

diff2 = abs(promedio_intervalos_biases_p2 - promedio_intervalos_variances_p2)
inter2 = param_grid['n_neighbors'][np.argmin(diff2)]
#----------------------------------------------------------------------------------------------------------------------------------------#
#Ploteamos
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['font.size'] = 24

fig, ax1 = plt.subplots()
ax1.fill_between(param_grid['n_neighbors'],minimos_intervalos_biases_p1, maximos_intervalos_biases_p1, color='blue', alpha = 0.3,label=r'Intervalo de Bias$^2$')
ax1.plot(param_grid['n_neighbors'], promedio_intervalos_biases_p1, color='blue', linestyle='--', label=r'Promedio de Bias$^2$')
ax1.fill_between(param_grid['n_neighbors'],minimos_intervalos_variances_p1, maximos_intervalos_variances_p1, color='green', alpha = 0.5,label=r'Intervalo de Variance')
ax1.plot(param_grid['n_neighbors'], promedio_intervalos_variances_p1, color='green', linestyle='--', label=r'Promedio de Variance')
ax1.axvline(x=inter1, color='black', linestyle='--', label='Mejor modelo')
ax1.set_xlabel(r'Numero de vecinos (u.a.)')
ax1.set_ylabel(r'Valor (u.a.)')
ax1.set_title(r'Intervalos de Bias$^2$ y Variance (Manhattan)')
ax1.grid(True, alpha = 0.5)
legend = ax1.legend()
legend.get_frame().set_linewidth(0.5)
for text in legend.get_texts():
    text.set_fontsize(18)
plt.show()

fig, ax2 = plt.subplots()
ax2.fill_between(param_grid['n_neighbors'],minimos_intervalos_biases_p2, maximos_intervalos_biases_p2, color='blue', alpha = 0.3,label=r'Intervalo de Bias$^2$')
ax2.plot(param_grid['n_neighbors'], promedio_intervalos_biases_p2, color='blue', linestyle='--', label=r'Promedio de Bias$^2$')
ax2.fill_between(param_grid['n_neighbors'],minimos_intervalos_variances_p2, maximos_intervalos_variances_p2, color='green', alpha = 0.5,label=r'Intervalo de Variance')
ax2.plot(param_grid['n_neighbors'], promedio_intervalos_variances_p2, color='green', linestyle='--', label=r'Promedio de Variance')
ax2.axvline(x=inter2, color='black', linestyle='--', label='Mejor modelo')
ax2.set_xlabel(r'Numero de vecinos (u.a.)')
ax2.set_ylabel(r'Valor (u.a.)')
ax2.set_title(r'Intervalos de Bias$^2$ y Variance (Euclideana)')
ax2.grid(True, alpha = 0.5)
legend = ax2.legend()
legend.get_frame().set_linewidth(0.5)
for text in legend.get_texts():
    text.set_fontsize(18)
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------------#
