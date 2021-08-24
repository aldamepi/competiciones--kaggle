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
"""
¿Para qué sirve seaborn? ¿Cómo puedo hacer una matriz de correlaciones, para que sirve?
"""


# Importar el data set
dataset = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

dataset.info()

df = pd.read_csv('train.csv')

df_cat = df.loc[:, df.dtypes == 'object']
df_num = df.loc[:, df.dtypes != 'object']
"""
x_cat = df.select_dtypes('object')
"""
test_cat = test.loc[:, test.dtypes == 'object']
test_num = test.loc[:, test.dtypes != 'object']


"""
Creo que debería dividir antes los campos Categoricos de los numéricos:
da igual porque estamos separando categorías. Quizas los que debería unir es 
el conjunto de test y el de entrenamiento para hacer la separación y el tramiento 
de los datos categoricos.

Lo que no tengo claro es como tratar los NAs en los datos categoricos
"""
X = dataset.iloc[:, 1:-1]
# le quito el . values de momento para ver los nombres de las variables
y = dataset.iloc[:, -1].values




# ANALISIS DE DATOS FALTANTES:
plt.figure(figsize=(20,6));
sns.heatmap(dataset.isnull(),yticklabels=False, cbar=False, cmap='mako') # paquete seaborn

plt.figure(figsize=(20,6));
sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='Spectral')

plt.figure(figsize=(20,6));
sns.heatmap(df_cat.isnull(),yticklabels=False, cbar=False, cmap='binary')

plt.figure(figsize=(20,6));
sns.heatmap(df_num.isnull(),yticklabels=False, cbar=False, cmap='gist_yarg')


# ANALISIS EXPLORATORIO DE DATOS

categoric, numeric = [],[] # esto crea dos listas

for z in dataset.columns:
    t = dataset.dtypes[z]
    if t=='object':
        categoric.append(z)
    else:
        numeric.append(z)
        
# Añadir categoricas numéricas
"""
¿Cuales son ordinales? Distinguir ordinales de nominales
"""  
c2 = ['MSSubClass', 'OverallQual', 'OverallCond']

categoric = c2 + categoric

for c in c2:
    numeric.remove(c)
    """
    me faltaba quitarlos de numeric
    """
    


# CORRELATION
corr = dataset.corr()
corr.sort_values(['SalePrice'], ascending = False, inplace=True)
"""
esto ordena el dataframe de corr por los valores de la columna Sale Price y podemos
observar cuales son las variables que tienen mayor incidencia directa en SalePrice
"""
print(corr.SalePrice)


# Tratamiento de los NAs para el conjunto de Entrenamiento
"""
--Para los NAN categoricos recomienda Juan Gabriel hacer un clustering y voto por mayoría
Curso ML clase 25 datos faltantes.--
¡¡¡OJO!!! Acabo de darme cuenta de que los NAN categoricos corresponden a que no existe, 0.
Es decir, tengo que cambiar los nan por NA o NE.
¿Que pasa entonces con los numéricos? ¿Son 0? Sip son 0
¿Que pasa con la fecha de construcción del garage?
Es decir, a las variables numericas les cambio el nan por 0 y a las categoricas el
nan por NA.
"""
from sklearn.impute import SimpleImputer

imputer_num = SimpleImputer(missing_values=np.nan, 
                            strategy="constant", fill_value=0)
imputer_num=imputer_num.fit(X.loc[:,X.dtypes !='object'])
X.loc[:,X.dtypes !='object']=imputer_num.transform(X.loc[:,X.dtypes !='object'])

X_num = X.loc[:,X.dtypes !='object']

plt.figure(figsize=(20,6));
sns.heatmap(X_num.isnull(),yticklabels=False, cbar=False, cmap='Blues')


imputer_cat = SimpleImputer(missing_values=np.nan, 
                            strategy="constant", fill_value="NA")
imputer_cat =imputer_cat.fit(X.loc[:,X.dtypes =='object'])
X.loc[:,X.dtypes =='object']=imputer_cat.transform(X.loc[:,X.dtypes =='object'])

X_cat = X.loc[:,X.dtypes =='object']

plt.figure(figsize=(20,6));
sns.heatmap(X_cat.isnull(),yticklabels=False, cbar=False, cmap='binary')

# Las fechas de construccion del garage por la media

imputer_fech = SimpleImputer(missing_values=0, strategy="mean")
imputer_fech=imputer_fech.fit(X.loc[:,X.columns =='GarageYrBlt'])
X.GarageYrBlt =imputer_fech.transform(X.loc[:,X.columns =='GarageYrBlt'])

"""
¿esto hay que hacerlo en el conjunto de testing? ¿Es decir, funcionará el algoritmo
de predicción al introducirle datos faltantes NAN? 

¿No sería mejor juntar los datos?
"""
# Tratamiento de los NAs para el conjunto de Test

imputer_num_test=imputer_num.fit(test.loc[:,test.dtypes !='object'])
test.loc[:,test.dtypes !='object']=imputer_num_test.transform(test.loc[:,test.dtypes !='object'])

imputer_cat_test = imputer_cat.fit(test.loc[:,test.dtypes =='object'])
test.loc[:,test.dtypes =='object']=imputer_cat_test.transform(test.loc[:,test.dtypes =='object'])

imputer_fech_test=imputer_fech.fit(test.loc[:,test.columns =='GarageYrBlt'])
test.GarageYrBlt =imputer_fech_test.transform(test.loc[:,test.columns =='GarageYrBlt'])


# Codificar datos categóricos
"""
¿que pasa cuando hay NAN?: se soluciona antes ¿Y si lo hago en R??
"""
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

# Codificar datos categoricos
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

for c in categoric:
    le_X = preprocessing.LabelEncoder()
    X.loc[:,X.columns==c]=le_X.fit_transform(X.loc[:,X.columns==c])
    
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto') ,[tuple(categoric)])], 
    remainder='passthrough'
    )
X2=pd.DataFrame(ct.fit_transform(X))   
X2 = np.array(ct.fit_transform(X), dtype=np.float_)


# La columna 'MSSubClass' aunque no sea objeto tambien es variable categórica
"""
Hay dos más, ver txt: OverallQual, OverallCond
"""
le_X = preprocessing.LabelEncoder()
X.loc[:,X.columns=='MSSubClass']=le_X.fit_transform(X.loc[:,X.columns=='MSSubClass'])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), ['MSSubClass'])], 
    remainder='passthrough'
    )
X2=pd.DataFrame(ct.fit_transform(X))


 

# One-Hot-Encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], 
    remainder='passthrough'
    )
X=pd.DataFrame(ct.fit_transform(X))
X = np.array(ct.fit_transform(X), dtype=np.float_)
X2=pd.DataFrame(ct.fit_transform(X[categoric].to_numpy())) 


X3 = pd.get_dummies(X)
X3.head()



# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X3_train, X3_test, y_train, y_test = train_test_split(X3, y, test_size = 0.2, random_state = 0)

# Ajustar el modelo XGBoost al Conjunto de Entrenamiento
from xgboost import XGBClassifier
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X3_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = regressor.predict(X3_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Aplicar k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X3_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# 0.8595
# 0.0697