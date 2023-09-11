import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Importar RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import time

inicio = time.time()

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

if False: 
    # Crear el modelo de Random Forest
    model = RandomForestClassifier()  # Utilizar RandomForestClassifier

    # Definir el espacio de búsqueda de hiperparámetros
    param_dist = {
        'n_estimators': [50, 100, 200],  # Número de árboles en el bosque
        'criterion': ['gini', 'entropy'],  # Criterio de división
        'max_depth': [None, 10, 20, 30, 40, 50],  # Profundidad máxima del árbol
        'min_samples_split': [2, 5, 10],  # Mínimo de muestras para dividir un nodo
        'min_samples_leaf': [1, 2, 4]  # Mínimo de muestras en una hoja
    }

    # Realizar Random Search para optimizar hiperparámetros
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Obtener los mejores hiperparámetros
    best_params = random_search.best_params_
    print("Mejores hiperparámetros:", best_params)
else: 
    best_params = {'n_estimators': 50, 'min_samples_split': 5,'min_samples_leaf': 1, 'max_depth': 20, 'criterion': 'entropy'}


# Ajustar el modelo con los mejores hiperparámetros
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train, y_train)

# Predecir con el modelo optimizado
y_pred = best_model.predict(X_test)
# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
matriz_confusion = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
nombres_clases = df['class'].unique()

#Guardamos el classreport
class_report = classification_report(y_test, y_pred, target_names=nombres_clases)
print(class_report)


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['font.size'] = 16
# Generar y mostrar la matriz de confusión
plt.figure(figsize=(8, 6))
plt.imshow(matriz_confusion, interpolation='nearest', cmap='Greens')
plt.colorbar()

# Mostrar los valores en la matriz de confusión
for i in range(len(nombres_clases)):
    for j in range(len(nombres_clases)):
        plt.text(j, i, str(matriz_confusion[i, j]), ha='center', va='center')

plt.xticks(np.arange(len(nombres_clases)), nombres_clases)
plt.yticks(np.arange(len(nombres_clases)), nombres_clases)
plt.xlabel(r"Valores Predichos")
plt.ylabel(r"Valores Reales")
titulo = r"Matriz de Confusión"
plt.title(titulo + "\nExactitud: {:.4f} \nPuntuación F1: {:.4f}".format(accuracy, f1))

fin = time.time()
total = fin-inicio
print(total)
plt.show()

