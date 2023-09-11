import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

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


best_params = {'n_neighbors': 3, 'p': 2}


# Ajustar el modelo con los mejores hiperparámetros
best_model = KNeighborsClassifier(**best_params)
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
plt.show()



