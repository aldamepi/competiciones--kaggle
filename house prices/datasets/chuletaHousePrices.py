#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:59:27 2021

@author: albertomengual
"""


# CHULETA DE CODIGOS INTERESANTES PARA RESOLVER EL HOUSE PRICES DE KAGGLE

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

len(categoric)
len(numeric)
len(df.columns)


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


# Generating dumies for categorical feature
all_data = pd.get_dummies(all_data)
all_data.head()