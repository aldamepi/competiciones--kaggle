#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June  28  2021

@author: Alberto Mengual
"""

# Competición en Kaggle de las House Prices:


# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ¿Para qué sirve seaborn? ¿Cómo puedo hacer una matriz de correlaciones, para que sirve?


# Importar el data set
dataset = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

dataset.info()

X = dataset.iloc[:, 1:-1]
# le quito el . values de momento para ver los nombres de las variables
y = dataset.iloc[:, -1].values

# ?pd.concat() #dataset.columns
# Me faltan varias cosas: Analisis de datos, etc.

# intro al ANALISIS EXPLORATORIO DE DATOS: Identificar los datos Categoricos y los Númericos
categoric, numeric = [],[] # esto crea dos listas

for z in dataset.columns:
    t = dataset.dtypes[z]
    if t=='object':
        categoric.append(z)
    else:
        numeric.append(z)
        """
        estamos creando dos listas con los nombres de las variables que se corresponden
        con datos numericos o datos categoricos
        """
print("CategoricaL:\n{}".format(categoric)) ## Esto que es??
print("\nNumericaL:\n{}".format(numeric)) 
"""
\n representa un salto de linea
{} indica que imprima lo que viene en el .format
.format indica lo que se tiene que imprimir dentro de las llaves {}
"""




# ANALISIS DE DATOS FALTANTES:
plt.figure(figsize=(20,6));
sns.heatmap(dataset.isnull(),yticklabels=False, cbar=False, cmap='mako') # paquete seaborn


# CORRELATION
corr = dataset.corr()
corr.sort_values(['SalePrice'], ascending = False, inplace=True)
"""
esto ordena el dataframe de corr por los valores de la columna Sale Price y podemos
observar cuales son las variables que tienen mayor incidencia directa en SalePrice
"""
print(corr.SalePrice)


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [2])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Ajustar el modelo XGBoost al Conjunto de Entrenamiento
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Aplicar k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

