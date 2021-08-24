#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 20:14:06 2021

@author: albertomengual
"""

"""
Pasos llevados a cabo en R:
    1 Importar Datos:
        * Eliminar columnas innecesarias - hecho
        * Traducir nombres de las variables y los valores - hecho
        - codificar la variable cubierta
    2 Preparar los datos para el analisis estadistico:
        * Codificar como factor la variable de clasificacion - ??
        * Convertir las variables categoricas a factores en los conjuntos de entrenamiento y test - ??
    2.1 Analisis Estadistico Descriptivo:
        * Información básica del dataset
        * Analisis de las variables cualitativas y ordinales
        * Tablas multidimensionales
        * Conslusiones del Analisis Descriptivo
        * Analisis de las variables cuantitativas:
            - Diagramas de bigotes: general y agrupados
            - Intervalos de agrupacion
            - Analisis bidemensional
            - Histograma   
        * Conclusiones del analisis cualitativo
    2.2 Ultimar la preparacion de los datos para el modelo:
        * Tratamiento de los NA
        * Codificar las variables categoricas
        * Seleccion de variables a entrenar
        * Escalado de valores
        * Dividir el conjunto de datos en entrenamiento y test
    3 Seleccionar modelo:
        * Aplicar el modelo de K-fold Cross Validation
        * Random Forest
        * XGBoost
        * Kernel SVM
        * RNA
    4 Entrenar modelo
    5 Evaluar modelo
    6 Tunear parametros
    7 Obtener predicciones y output
"""

# Como importar las librerias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sb

# Importar el data set
datatrain = pd.read_csv('../datasets/train.csv', index_col='PassengerId')
datatest = pd.read_csv('../datasets/test.csv')


df = datatrain.drop(['Name','Ticket'], axis = 'columns')
dt = datatest.drop(['Name','Ticket','PassengerId'], axis = 'columns')

df.columns = ['Superviviente', 'clase', 'genero', 'edad', 'SibSp', 'Parch', 
              'tarifa', 'cubierta', 'puerto']
dt.columns = ['clase', 'genero', 'edad', 'SibSp', 'Parch', 
              'tarifa', 'cubierta', 'puerto']

df.genero = df.genero.map({'female' : 'mujer',
                           'male' : 'hombre'}, na_action = None)
dt.genero = dt.genero.map({'female' : 'mujer',
                           'male' : 'hombre'}, na_action = None)

### Pendiente
#X = df.iloc[:, 1:]
#y = df.iloc[:, 0].values





# PREPARAR LOS DATOS PARA EL ANALISIS ESTADISTICO

## codificar la variable cubierta - error
def substring_in_string (value, substrings):
    for substring in substrings:
        if substring in value:
            return substring
####    print (value)
####    return np.nan

cubiertas = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

df.cubierta.fillna("Unknown", inplace=True)
dt.cubierta.fillna("Unknown", inplace=True)

df['cubierta'] = df ['cubierta'].map(lambda x: substring_in_string(x, cubiertas))
dt['cubierta'] = dt ['cubierta'].map(lambda x: substring_in_string(x, cubiertas))


#### ¿Que es un Dtype Object en python? - es un rollo filosofico, ver favoritos
#### ¿Se tienen que convertir las variables a factores en python? 
#### - parece que no existen factores en python - OJO esta la función pd.Categorical
#### - convertir a categorias no permite realizar operaciones numericas


## Codificar como factor la variable de clasificación

df.Superviviente = pd.Categorical(df.Superviviente)


## Convertir las variables categoricas a categorias
cual_col = ['clase', 'genero', 'SibSp', 'Parch', 'cubierta', 'puerto']
ordi_col = ['clase', 'SibSp', 'Parch']
cuanti_col = ['edad', 'tarifa']

for c in cual_col:
    if c in ordi_col:
        df[c] = pd.Categorical(df[c], ordered=True)
        dt[c] = pd.Categorical(dt[c], ordered=True)
    else:
        df[c] = pd.Categorical(df[c])
        dt[c] = pd.Categorical(dt[c])

df.clase = pd.Categorical(df.clase, ordered=True, categories=[3,2,1])
dt.clase = pd.Categorical(dt.clase, ordered=True, categories=[3,2,1])

df.Parch = pd.Categorical(df.Parch, ordered=True, categories=[0,1,2,3,4,5,6,9])





# Analisis Estadistico Descriptivo


## Informacion Basica del Dataset
df.head()
df.tail()
df.shape
describe = df.describe()
df.info()

## Analisis de las variables cualitativas y ordinales

### Tablas undimensionales
"""
tab_sup = pd.crosstab(index= df.Superviviente, columns = "pasajeros")
tab_sup_rel = 100 * tab_sup/tab_sup.sum()
tab_gen = pd.crosstab(index = df.genero, columns = "pasajeros")
tab_cub = pd.crosstab(index = df.cubierta, columns = "pasajeros")
tab_clas = pd.crosstab(index = df.clase, columns = "pasajeros")
tab_clas_rel = 100 * tab_clas/tab_clas.sum()
"""
for c in cual_col:
    tab_uni = pd.crosstab(index = df[c], columns="pasajeros")
    print(tab_uni)
    tab_uni_rel = round(100 * tab_uni/tab_uni.sum(),1)
    print(tab_uni_rel)
#    pd.crosstab(index = df[c], 
#                columns="pasajeros")/pd.crosstab(index = df[c],
#                                                 columns="pasajeros").sum()
    sb.countplot(x=df[c])
    plt.title(c)
    plt.show()
    plt.pie(tab_uni["pasajeros"], labels = tab_uni.index, autopct="%1.1f%%")
    plt.xlabel(c)
    plt.show()


plt.pie(tab_uni["pasajeros"], labels = tab_uni.index, autopct="%1.1f%%")
plt.xlabel(c)
plt.show()

### Tablas bidimensionales
"""
sup_gen = pd.crosstab(index = df.Superviviente, columns=df.genero, margins=True)
sup_clas = pd.crosstab(index = df.Superviviente, columns = df.clase, margins=True)
sup_clas.index = ["fallecido", "superviviente", "total_c"]
sup_clas.columns = ["tercera", "segunda", "primera", "total_p"]
"""
"""
for c in df.columns:
    df[[c,"Superviviente"]].groupby([c]).mean().plot.bar()
    
    sb.countplot(x=c,hue='Superviviente',data=df)
    plt.show()
    
    
    sb.countplot(x=df[c],hue='Superviviente',data=df)
    plt.show()
    
    sb.countplot(x=df[c]/df[c].sum(),hue='Superviviente',data=df)
    plt.show()
"""

#### Frecuencias relativas globales

###### ¿Cómo se hace los diagramas de frecuencias relativas en python? 
###### - creo que solo se puede hacer para tablas multidimensionales
###### ver barplots:https://seaborn.pydata.org/tutorial/categorical.html#categorical-tutorial 

for c in cual_col:
    pd.crosstab(index = df.Superviviente, columns=df[c], margins=True)
    pd.crosstab(index = df.Superviviente,
                columns=df[c],
                margins=True)/pd.crosstab(index = df.Superviviente,
                                                         columns=df[c],
                                                         margins=True).loc["All","All"]
    sb.catplot(x='clase',y ='Superviviente', hue = 'genero',data=df)
    plt.show()
        
#100 * sup_clas/sup_clas.loc["total_c", "total_p"]


#### Frecuencias relativas parciales

##### Frecuencias relativas paciales por columnas
#100 * sup_clas/sup_clas.loc["total_c"]

##### Frecuencias relativas parciales por filas
#100 * sup_clas.div(sup_clas["total_p"], axis=0)



### Tablas Multidimensionales

sup_gen_clas = pd.crosstab(index= df.Superviviente,
                           columns = [df.genero, df.clase],
                           margins = True)
sup_gen_clas_rel = sup_gen_clas/sup_gen_clas.loc["All","All"][0]

sup_cub_clas = pd.crosstab(index = df.Superviviente,
                           columns = [df.clase, df.cubierta],
                           margins = True)
sup_cub_clas_rel = 100* sup_cub_clas/sup_cub_clas.loc["All","All"][0]

sup_gen_cub_clas = pd.crosstab(index = [df.Superviviente, df.genero],
                           columns = [df.clase, df.cubierta],
                           margins = True)
sup_gen_cub_clas_rel = 100 * sup_gen_cub_clas/sup_gen_cub_clas.loc["All","All"][0]

sup_gen_clas_pue = pd.crosstab(index = [df.genero, df.Superviviente],
                           columns = [df.puerto, df.clase],
                           margins = True)
sup_gen_clas_pue_rel = 100 * sup_gen_clas_pue/sup_gen_clas_pue.loc["All","All"][0]

sup_gen_sib_clas = pd.crosstab(index = [df.genero, df.Superviviente], 
                               columns = [df.clase, df.SibSp],
                               margins = True)
sup_gen_sib_clas_rel = 100 * sup_gen_sib_clas/sup_gen_sib_clas.loc["All","All"][0]

sup_gen_par_clas = pd.crosstab(index = [df.genero, df.Superviviente], 
                               columns = [df.clase, df.Parch],
                               margins = True)
sup_gen_par_clas_rel = 100 * sup_gen_par_clas/sup_gen_par_clas.loc["All","All"][0]


###### sb.catplot(x="genero", y="Superviviente", hue="clase", kind="bar", data=df)
###### plt.show()

"""
for c in cual_col:
    plt.figure()
    sb.catplot(x='Superviviente', col = c, kind = "count", col_wrap = 2, data = df)
    plt.show()


    
plt.figure(figsize=(10,10))
sb.catplot(x='Superviviente', col='genero', kind="count", data = df)
plt.show()
"""


## ANALISIS DE VARIABLES CUANTITATIVAS

#### Analisis Basico 
cuanti_col

df.edad.min()
df.edad.max()
df.edad.mean()
df.edad.median()
df.edad.mode()
df.edad.quantile([0.25,0.75])

df.edad.isna().sum()

df.tarifa
df.tarifa.min()
df.tarifa.max()
df.tarifa.mean()
df.tarifa.median()
df.tarifa.mode()
df.tarifa.quantile([0.25,0.75])

df.tarifa.isna().sum()


sb.boxplot(y = 'edad', data = df)
plt.show()


sb.boxplot(y = 'tarifa', data = df)
plt.show()


#### Analisis bidimensional cualitativo
sb.boxplot(x = 'Superviviente', y = 'edad', data = df)
plt.show()

sb.boxplot(x = 'Superviviente', y = 'tarifa', data = df)
plt.show()



#### Analisis Multivariante cualitativo

###### Los diagramas de violin son un poco cutres, buscar otros

sb.violinplot(x = 'clase', y = 'edad', hue = 'Superviviente', data = df, split = True)
plt.show()

sb.violinplot(x = 'genero', y = 'edad', hue = 'Superviviente', data = df, split = True)
plt.show()

sb.violinplot(x = 'clase', y = 'tarifa', hue = 'Superviviente', data = df, split = True)
plt.show()

sb.violinplot(x = 'genero', y = 'tarifa', hue = 'Superviviente', data = df, split = True)
plt.show()

sb.violinplot(x = 'cubierta', y = 'tarifa', hue = 'Superviviente', data = df, split = True)
plt.show()


sb.pairplot(df[cuanti_col])
plt.show()


#### Histogramas

###### Intentar ponerlos más bonitos, meterles colorines y titulos. Intentar añadir el rug
###### Les falta las densidades, concepto que aún no entiendo


###### falta meter alguna pivot.table, no? - 
###### para hacer las pivot table, en concreto los cut, hay que quitar los nan
df.edad.plot.hist(bins=20, edgecolor='black')
plt.show()

df[df.Superviviente == 0].edad.plot.hist(bins=20, edgecolor='black')
plt.show()

#vector_edad = pd.cut(df["edad"], bins=20, include_lowest=True, right=False)
#tab_hist_edad = df.pivot_table(df.edad, index=df.Superviviente, columns=vector_edad)
#print(tab_hist_edad)

df[df.Superviviente == 1].edad.plot.hist(bins=20, edgecolor='black')
plt.show()

df.tarifa.plot.hist(bins = 100, edgecolor='black')
plt.show()

df[df.Superviviente == 0].tarifa.plot.hist(bins=50, edgecolor='black')
plt.show()

df[df.Superviviente == 1].tarifa.plot.hist(bins=100, edgecolor='black')
plt.show()


#### Conclusiones del Analisis Cualitativo

#### Los niños varones tienen un alto indice de supervivencia.






# PREPARAR LOS DATOS PARA EL MODELO

## Tratamiento de los NA

### En el conjunto de entrenamiento

edad_master = datatrain.Age[datatrain.Name.str.contains("Master")]
edad_master.mean()

df.loc[(datatrain.Age.isna()) & (datatrain.Name.str.contains("Master")), "edad"] = 4.574

df.edad.fillna(df.edad.mean(), inplace=True)

### En el conjunto de test

edad_master_test = datatest.Age[datatest.Name.str.contains("Master")]
edad_master_test.mean()
edad_master_test.max()

dt.loc[(datatest.Age.isna()) & (datatest.Name.str.contains("Master")), "edad"] = 7.406

dt.edad.fillna(dt.edad.mean(), inplace=True)

#### Hay un NAN en tarifa

dt.tarifa.fillna(dt.tarifa.mean(), inplace=True)




## Gestión de Variables: Edición de variables a entrenar y separar matriz de caracteristicas
### Crear variable adulto? O añadir niño a genero - creo la variable
df.columns

df.loc[(df.edad <= 15), "adulto"] = "joven"
df.adulto.fillna("adulto", inplace=True)

pd.crosstab(index=[df.Superviviente, df.genero], columns=[df.clase, df.adulto], margins=True)


#### En el conjunto de test
dt.loc[(dt.edad <= 15), "adulto"] = "joven"
dt.adulto.fillna("adulto", inplace=True)


df.adulto = pd.Categorical(df.adulto)
dt.adulto = pd.Categorical(dt.adulto)

cual_col.append("adulto")


### Obtener la matriz de caracteristicas y la variable independiente
X = df.iloc[:, 1:]
y = df.iloc[:, 0].values

x_t = dt.copy()

x0 = pd.concat([X,x_t], axis=0)




## Codificar variables categoricas
### Separar variables categoricas a obtener sus dummies, juntar antes con el conjunto de test??
###### Hay que sacar variables dummies de las que no son ordinales

#for c in cual_col:
#    if c not in ordi_col:
dummi_col = cual_col.copy()    

for x in ordi_col:
    dummi_col.remove(x)     
       
###### X = pd.get_dummies(X, prefix=dummi_col, columns=dummi_col, drop_first=True)        
x0 = pd.get_dummies(x0, prefix=dummi_col, columns=dummi_col, drop_first=True)




## Dividir el conjunto de datos en entrenamiento y test

X = x0.iloc[0:891,:]
x_t = x0.iloc[891:,:]

###### ¿Es necesario con la k-fold cross validation? 
###### Si, porque la k-fold se hace con el conjunto de entrenamiento

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



## Escalado de valores
###### Me espero a aplicar a el modelo oportuno?? 
"""
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

!!ojo con el escalado que tendre que escalar tambien el x_t para la prediccion de kaggle!!
"""





# Seleccionar los modelos

## Aplicar el modelo de k-fold cross validation y el grid search
###### En el Grid Search entra la K-fold Cross Validation
from sklearn.model_selection import GridSearchCV
"""
parameters = [{'C': [0,5, 1, 10, 100, 1000],'kernel': ['linear']},
              {'C': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],'kernel': ['rbf'], 
               'gamma': [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]}
              ]


grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)


grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_
"""



## Aplicar el modelo RANDOM FOREST
### Ajustar el clasificador  Random Forest en el Conjunto de Entrenamiento
from sklearn.ensemble import RandomForestClassifier

classiRF = RandomForestClassifier()

parametersRF = [{'n_estimators' : [10,50,100, 500],
                 'criterion' : ["entropy", "gini"],
                 'random_state' : [0]
                 }]

grid_search = GridSearchCV(estimator = classiRF,
                           param_grid= parametersRF,
                           scoring='accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search_RF = grid_search.fit(X_train, y_train)

best_accuracy_RF = grid_search.best_score_

best_parameters_RF = grid_search.best_params_


"""
classifier = RandomRadForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)
"""

### Predicción de los resultados con el Conjunto de Testing
classiRF = RandomForestClassifier(n_estimators = 500, criterion = "entropy", random_state = 0)
classiRF.fit(X_train, y_train)

y_pred_RF  = classiRF.predict(X_test)

### Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix, accuracy_score

cmRF = confusion_matrix(y_test, y_pred_RF)
precisionRF = (cmRF[0,0]+cmRF[1,1])/cmRF.sum()
accuracy_score(y_test, y_pred_RF)




# CREAR EL OUTPUT

## Output RandomForests

classiRF_out = classiRF.fit(X,y)

y_out_RF = classiRF_out.predict(x_t)

y_out_RF = pd.DataFrame(y_out_RF, columns=["Survived"])

outRF = pd.concat([datatest.PassengerId, y_out_RF], axis=1)
outRF.set_index('PassengerId', inplace=True)

#outRF.to_csv('../datasets/subm_am_py_rf_050821.csv')
##### 0.73923











