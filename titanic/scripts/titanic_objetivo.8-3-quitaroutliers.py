#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 19:45:09 2021

@author: albertomengual
"""


"""
En este script voy a repetir el desarrollo del modelo de clasificación para el 
conjunto de datos de la competición de kaggle del titanic.
La puntuación más alta obtenida hasta ahora ha sido aprximadamente 0.78. El primer
objetivo de este script es llegar a una puntuación de 0.8.
Dentro de los objetivos se encuentran también analizar la tecnicas de ingeniería 
de variables y asentar las habilidades de programción necesarias.
Para conseguir dichos objetivos voy a seguir las tecnicas del blog de Ahmed Besbes.
"""

"""
RESUMEN DE PASOS A SEGUIR
Será necesario ampliar la descripción de los pasos a medida que se desarrolle el 
análisis.

1. Exploratory Data Analysis (DATA EXPLORATION AND VISUALIZATION)
    * Data Extraction
    * Cleaning (DATA CLEANING)
    * Plotting
    * Assumptions
    
2. Ingenieria de Variables (FEATURE ENGINEERING)
    * Append test set
    * Extracting the passengers titles
    * Processing the ages
    * Processing Fare
    * Processing Embarked
    * Processing Cabin
    * Processing Sex
    * Processing Pclass
    * Processing Ticket
    * Processing Family
    
3. Modelling
    * Break de combined data set in train and test set
    * Use the train set to build a predictive model
        + Feature Selection (FEATURE SELECTION)
        + Trying different models
        + Hyperparameters tuning (HYPERPARAMETERS TUNING)
    * Evaluate de model using de train set
    * Generate an output file for the submission (SUBMISSION)
    * Blending different models

"""


# 0.Importar las librerias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
# import pylab as plot
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score





# 1. Analisis Exploratorio de los Datos


## Extraer los datos

train = pd.read_csv('../datasets/train.csv', index_col='PassengerId')
test = pd.read_csv('../datasets/test.csv')


df = train.copy()
dt = test.copy()


### Analisis Básico de los Datasets
print(train.shape)
print(test.shape)

print(train.head)

print(train.describe())

"""
La variable objetivo a predecir es Survived
"""

### Traducir las variables y categorias

df.columns = ['Superviviente', 'clase', 'nombre', 'genero', 'edad', 'SibSp', 'Parch', 
              'tique','tarifa', 'cabina', 'puerto']
dt.columns = ['PassengerId','clase', 'nombre', 'genero', 'edad', 'SibSp', 'Parch', 'tique',
              'tarifa', 'cabina', 'puerto']

df.genero = df.genero.map({'female' : 'mujer',
                           'male' : 'hombre'}, na_action = None)
dt.genero = dt.genero.map({'female' : 'mujer',
                           'male' : 'hombre'}, na_action = None)

dt.set_index("PassengerId", inplace=True)
#dt.drop("PassengerId", axis=1, inplace=True)

### Contar los NAN
print (df.isna().sum())
print (dt.isna().sum())





## Hacer algunos gráficos (Plotting)


### Visualizar los supervivientes basados en el genero
"""
Aqui se pueden sacar las frecuencias relativas con la media de las variables categoricas
porque se trata de una categoria lógica y creando una categoria extra nos basta
para incluirla en la groupby con la media agregada.
"""
df["fallecido"] = 1 - df.Superviviente

#### frecuencias absolutas
df.groupby("genero").agg("sum")[["Superviviente","fallecido"]].plot(kind="bar", 
                                                                    figsize=(25,7),
                                                                    stacked=True, 
                                                                    color= ['g','darkorange'],
                                                                    table = True
                                                                    )
#### frecuencias relativas
df.groupby("genero").agg("mean")[["Superviviente","fallecido"]].plot(kind="bar", 
                                                                    figsize=(25,7),
                                                                    stacked=True, 
                                                                    color= ['g','darkorange'],
                                                                    table = True
                                                                    )

"""
Queda claro que el genero es una categoría discriminatoria. Las mujeres tienen una 
tasa de superviviencia más alta con diferencia.
"""


### Relacionar la supervivencia y el genero con la variable edad

fig = plt.figure(figsize=(15,15))
sb.set_theme(style="whitegrid")
sb.violinplot(x="genero", y="edad",
              hue="Superviviente", data = df,
              split = True,
              palette = {0: "darkorange", 1: "g"},
              scale = "count", scale_hue = False,
              inner="stick",
              cut=0
              )
#plt.yticks()
plt.show()

"""
YA NO ME DISGUSTAN LOS DIAGRAMAS DE VIOLIN: me salen valores negativos
con las escalas mejoran un poco.
TODAVIA NO HE TRATADO LOS NAN DE LA EDAD, es posible que haya supervivientes o 
fallecidos de algún genero que no aparezcan en los gráficos de violin. Da igual 
porque la estrategia de tratamiento de NAN se orienta a no modificar la proporcion
actual.
"""
# =============================================================================
# Se confirma a simple vista que las mujeres tienen una tasa de supervivencia más
# alta que los hombres. Es decir, la mayoría de los supervienvientes son femeninos
# superando a las fallecidas del mismo genero y a los supervivientes masculinos.
# La mayoría de las supervivientes femeninas se encuentran entre los 13 y los 45 años.
# Se mantiene una alta tasa de supervivencia en la mujeres menores de 8 años así 
# como en las mayores de 50.
# No exiten fallecidas menores de un año.

# En el genero masculino destaca una tasa de mortalidad muy elevada entre los 14 
# y los 50 años.
# Teniendo en cuenta la baja tasa de supervivencia del genero masculino en todas 
# las edades se aprecia un incremento relativo de la superviviencia en los niños.
# En particular no fallecen los bebes menores de un año. 

# MUJERES Y NIÑOS PRIMERO

# NOTA: Estaría bien comprobar si las familias que se salvaron tenian bebes.
# Es decir, comprobar los familiares de los menores.
# =============================================================================


# =============================================================================
# familia_Baclini = df[df.nombre.str.contains("Baclini")]
# 
# familia_Allison = df[df.nombre.str.contains("Allison")]
# 
# familia_Hamalainen = df[df.nombre.str.contains("Hamalainen")]
# 
# familia_Caldwell = df[df.nombre.str.contains("Caldwell")]
# 
# familia_Thomas = df[df.tique == "2625"]
# 
# familia_Richards = df[df.tique == "29106"]
# 
# =============================================================================

# =============================================================================
# La mayoría de los familiares de los bebes sobrevivieron. Solo en la familia
# Allison, extrañamente fallecieron dos mujeres de 2 y 25 años.
# Las personas de la misma familia tienen el mismo n de tique.
# Cabría hacer la misma comprobación para los niños que sobrevivieron.
# Y para las mujeres que sobrevivieron.
# =============================================================================
"""
¿Cómo lo hago en modo Big Data? ¿Creando clusters?
"""



### Relacionar la supervivencia con la variable tarifa

figure = plt.figure(figsize=(25,7))
plt.hist([df.tarifa[df.Superviviente == 1],df.tarifa[df.Superviviente == 0]],
         stacked=True, color = ["g", "darkorange"],
         bins = 100, label = ["superviviente", "fallecido"])
plt.xlabel("Tarifa")
plt.ylabel("N de pasajeros")
plt.legend()
plt.show()

# =============================================================================
# La mayoría de los pasajeros se encuentran en unos rangos de tarifa inferiores
# a los 50. Hay es donde están la mayoria de los fallecidos y de los supervivientes.
# En especial se observa una tasa elevada de fallecidos entre las tarifas más bajas,
# por debajo incluso de 25.
# =============================================================================

### La tarifa en función de la edad clasificado por la supervivencia

plt.figure(figsize=(25,15))
ax = plt.subplot()

ax.scatter(df.edad[df.Superviviente == 1], df.tarifa[df.Superviviente == 1],
           c='g', s=df.tarifa[df.Superviviente == 1])
ax.scatter(df.edad[df.Superviviente == 0], df.tarifa[df.Superviviente == 0],
           c='darkorange', s=df.tarifa[df.Superviviente == 0])
plt.title("Tarifa en funcion de la edad ordenado por supervivencia")
plt.xlim(0,85)
plt.ylim(0,550)
plt.show()

### La tarifa en función de la edad clasificado por la genero

plt.figure(figsize=(25,15))
ax = plt.subplot()

ax.scatter(df.edad[df.genero == "mujer"], df.tarifa[df.genero == "mujer"],
           c='fuchsia', s=df.tarifa[df.genero == "mujer"])
ax.scatter(df.edad[df.genero == "hombre"], df.tarifa[df.genero == "hombre"],
           c='b', s=df.tarifa[df.genero == "hombre"])
plt.title("Tarifa en funcion de la edad ordenado por genero")
plt.xlim(0,85)
plt.ylim(0,550)
plt.show()

#### La tarifa en función de la edad clasificado por la supervivencia separado por generos

##### Mujeres
plt.figure(figsize=(25,15))
ax = plt.subplot()

ax.scatter(df.edad[df.Superviviente == 1][df.genero == "mujer"],
           df.tarifa[df.Superviviente == 1][df.genero == "mujer"],
           c='g', s=df.tarifa[df.Superviviente == 1][df.genero == "mujer"])
ax.scatter(df.edad[df.Superviviente == 0][df.genero == "mujer"], 
           df.tarifa[df.Superviviente == 0][df.genero == "mujer"],
           c='darkorange', s=df.tarifa[df.Superviviente == 0][df.genero == "mujer"])
plt.title("Tarifa en funcion de la edad  por supervivencia: FEMENINO")
plt.xlim(0,85)
plt.ylim(0,550)
plt.show()

##### Hombres
plt.figure(figsize=(25,15))
ax = plt.subplot()

ax.scatter(df.edad[df.Superviviente == 1][df.genero == "hombre"],
           df.tarifa[df.Superviviente == 1][df.genero == "hombre"],
           c='g', s=df.tarifa[df.Superviviente == 1][df.genero == "hombre"])
ax.scatter(df.edad[df.Superviviente == 0][df.genero == "hombre"], 
           df.tarifa[df.Superviviente == 0][df.genero == "hombre"],
           c='darkorange', s=df.tarifa[df.Superviviente == 0][df.genero == "hombre"])
plt.title("Tarifa en funcion de la edad  por supervivencia: MASCULINO")
plt.xlim(0,85)
plt.ylim(0,550)
plt.show()

"""
Estos graficos están bien para analizar lo que pasa.
Si busco un cluster para crear una nueva variable habria que evitar la variable 
Supervivencia.
En la ingenieria de variables Habría que hacer un clustering con las variables: 
    edad, tarifa, genero y clase.
"""

### La tarifa en funcion de la edad clasificada por clase

plt.figure(figsize=(25,15))
ax = plt.subplot()
ax.set_facecolor("whitesmoke")
ax.scatter(df.edad[df.clase == 1], df.tarifa[df.clase == 1],
            c='gold', s=df.tarifa[df.clase == 1])
ax.scatter(df.edad[df.clase == 2], df.tarifa[df.clase == 2],
            c='turquoise', s=df.tarifa[df.clase == 2])
ax.scatter(df.edad[df.clase == 3], df.tarifa[df.clase == 3],
            c='maroon', s=df.tarifa[df.clase == 3])
plt.title("Tarifa en funcion de la edad ordenado por clase")
plt.xlim(0,85)
plt.ylim(0,550)
plt.show()


#### La tarifa en funcion de la edad clasificada por agrupado por generos

##### Mujeres
plt.figure(figsize=(25,15), facecolor="lightgrey")
ax = plt.subplot()

ax.scatter(df.edad[df.clase == 1][df.genero == "mujer"], 
            df.tarifa[df.clase == 1][df.genero == "mujer"],
            c='gold', s=df.tarifa[df.clase == 1][df.genero == "mujer"])
ax.scatter(df.edad[df.clase == 2][df.genero == "mujer"], 
            df.tarifa[df.clase == 2][df.genero == "mujer"],
            c='turquoise', s=df.tarifa[df.clase == 2][df.genero == "mujer"])
ax.scatter(df.edad[df.clase == 3][df.genero == "mujer"], 
            df.tarifa[df.clase == 3][df.genero == "mujer"],
            c='maroon', s=df.tarifa[df.clase == 3][df.genero == "mujer"])
plt.title("Tarifa en funcion de la edad ordenado por clase: FEMENINO")
plt.xlim(0,85)
plt.ylim(0,550)
plt.show()


##### Hombres
plt.figure(figsize=(25,15), facecolor="lightgrey")
ax = plt.subplot()

ax.scatter(df.edad[df.clase == 1][df.genero == "hombre"], 
            df.tarifa[df.clase == 1][df.genero == "hombre"],
            c='gold', s=df.tarifa[df.clase == 1][df.genero == "hombre"])
ax.scatter(df.edad[df.clase == 2][df.genero == "hombre"], 
            df.tarifa[df.clase == 2][df.genero == "hombre"],
            c='turquoise', s=df.tarifa[df.clase == 2][df.genero == "hombre"])
ax.scatter(df.edad[df.clase == 3][df.genero == "hombre"], 
            df.tarifa[df.clase == 3][df.genero == "hombre"],
            c='maroon', s=df.tarifa[df.clase == 3][df.genero == "hombre"])
plt.title("Tarifa en funcion de la edad ordenado por clase: MASCULINO")
plt.xlim(0,85)
plt.ylim(0,550)
plt.show()


# =============================================================================
# No se aprecia una relación clara entre la edad y la tarifa.
# Se aprecian varios clusters:
    # El cluster de los niños menores de 14 años.
    # Hay un cluster de mujeres mayores de 14 años que pagan una tarifa superior
    # aproximadamente a los 50 donde se agrupa una tasa muy alta de supervivencia.
    # En este cluster destaca con mayoria la primera clase.
    # De hecho aparece un outlier claro con 25 años y 150 de tarifa aprox.
    # Se puede indicar otro cluster de mujeres con tarifas inferiores  a los 50 y
    # edades superiores a los 13 -14 años donde empieza a aumentar las tasa de 
    # mortalidad y se hace significativo pertenecer a la 3ª clase.
    # En los hombres hay un cluster muy diferenciado en tarifas bajas, inferiores
    # a 50 desde los 14 hasta los 40 años sobre todo (y se extiende hasta los 75
    # años) donde se concentra la mayor tasa de mortalidad.
    # Se podría indicar otro cluster de hombres con edades superiores a los 15 
    # años y tarifas aproximadamente superiores a 50 donde empiezan a observarse
    # ciertas tasas de supervivencia. En este cluster es donde se hace significativa
    # la clase: por debajo de la barrera proxima a los 50 los supervivientes masculinos
    # son de primera clase. Por encima de esa barrera, pocos son de 3ª clase.                                                
# =============================================================================



### Relacionar la tarifa con las clases
plt.figure()
ax = plt.subplot()
ax.set_ylabel("Tarifa Media")
df.groupby("clase").mean()["tarifa"].plot(kind = "bar", ax = ax)
plt.show()
                                          
# =============================================================================
# Obviamente la tarifa y la clase están relacionadas. A mejor clase, mayor tarifa
# =============================================================================





### Relacionar los puertos con la supervivencia

plt.figure(figsize=(15,15))
sb.violinplot(x="puerto", y="tarifa", hue="Superviviente", data = df, split=True,
              scale = "count", scale_hue=False, cut=0, inner= "stick",
              palette = {0:"darkorange", 1:"g"})
plt.title("Tarifa por puerto según supervivencia")
plt.show()


#### Relacionar las tarifas con los puertos y el genero
plt.figure(figsize=(15,15))
sb.violinplot(x="puerto", y="tarifa", hue="genero", 
              data = df,
              split=True,
              scale = "count", scale_hue=False, cut=0, inner= "stick",
              palette = {"hombre":"b", "mujer":"m"})
plt.title("Tarifa por puerto según genero")
plt.show()

"""
Hacer diagrama puerto-clase
"""
# =============================================================================
# La mayoría de los pasajeros embarcaron en el puerto S.
# En este puerto la curva de fallecidos y hombres embarcados son muy semejantes.
# Aqui es donde embarcaron la mayoria de los pasajeros de 3ª clase o con tarifas
# inferiores a 50.
# Se observa un grupo de pasajeros masculinos en el puerto S que no pagan por
# su billete y de los cuales solo sobrevive uno (de tercera). DATOS FALTANTES??.
# Sobreviven aproximadamente tantas personas como mujeres embarcan.

# En el puerto C el rango de tarifas es más alto. Aunque embarcan menos pasajeros,
# se observa que sobreviven más pasajeros que el numero de mujeres embarcadas.
# Comparando la curva de los hombres y los fallecidos, se observa un grupo de 
# hombres supervivientes cuya tarifa es superior a los 50.

# El puerto Q embarcan solo 77 pasajeros. No se observa ningún patron especial.
# La tasa de supervivencia es inferior a la de fallecidos y embarcan aproximadamente
# el mismo numero de mujeres que de hombres.

# =============================================================================


# =============================================================================
# pasajero_gratis = df[df.tarifa == 0]
# 
# pijos_C = df[df.genero == "hombre"][df.tarifa >=50][df.puerto == "C"]
# 
# df[df.puerto == "C"][df.genero == "hombre"].groupby("clase").sum()
# 
# familias = df[df.duplicated("tique")]
# 
# fami_m_1 = familias[df.genero == "mujer"][df.clase == 1]
# 
# capitan = df[df.nombre.str.contains("Capt")]
# =============================================================================

# familia_Fortune = df[df.nombre.str.contains("Fortune")]
# #df[df.tique == "19950"]

# familia_Lurette = df[df.tique == "PC 17569"]

# familia_Ryerson = df[(df.tique == "PC 17608") | (df.nombre.str.contains("Ryerson"))]

# familia_Carter = df[(df.tique == "113760")]
# # hay otros carter de segunda que cascan

# famila_shutes = df[df.tique == "PC 17582"]

# familia_Taussig = df[df.tique == "110413"]

# familia_china = df[df.tique == "1601"]

# =============================================================================
# A excepción de los chinos, no se observa que al ser familiar de una mujer de 
# primera clase tengas que sobrevivir
# =============================================================================

"""
Despues de descubrir algunas relaciones interesantes entre los datos, vamos a 
transformar los datos para que sean manejables por un algoritmo de ML
"""

# 2. INGENIERIA DE VARIABLES


## Cargar los datos
"""
Vamos a unir las matrices de variables del conjunto de entrenamineto y test
"""

y = df.Superviviente

X = df.drop(["Superviviente","fallecido"],axis=1)

x_t = dt.copy()

X_combi = X.append(x_t)




## Extraer y simplificar los titulos de los nombres
"""
Esto se hace para calcular las medias de las edades.
¿Se pueden sacar los apellidos?
"""

tratamiento = set()
# apellido = set()

for name in df.nombre:
    tratamiento.add(name.split(',')[1].split('.')[0].strip())
    # apellido.add(name.split(',')[0].strip())
"""
un set es una lista cuyos elementos no se pueden repetir
split divide la cadena de caracteres por un elemento y devuelve una lista
strip elimina los espacios en blanco
"""

# apellido = sorted(apellido)
tratamiento = sorted(tratamiento)

tratamiento_dict = {
    'Capt' : "Oficial",
    'Col' : "Oficial",
    'Don' : "Noble",
    'Dr' : "Oficial",
    'Jonkheer' : "Noble",
    'Lady' : "Noble",
    'Major' : "Oficial",
    'Master' : "Chavea",
    'Miss' : "Srta",
    'Mlle' : "Srta",
    'Mme' : "Sra",
    'Mr' : "Sr",
    'Mrs' : "Sra",
    'Ms' : "Sra",
    'Rev' : "Oficial",
    'Sir' : "Noble",
    'the Countess' : "Noble"
    }

X_combi["tratamiento"] = X_combi.nombre.map(lambda nombre: nombre.split(',')[1].split('.')[0].strip())

X_combi["tratamiento"] = X_combi.tratamiento.map(tratamiento_dict)

X_combi.tratamiento.isna().sum()
# 0
# 1





## Crear la variable bebé

X_combi["bebe"] = X_combi.edad.map(lambda e: 1 if e <=1 else 0)





## Crear la variable familia de bebe

tiques_bebeDf = set(df.tique[df.edad <= 1])
tiques_bebeDt = set(X_combi.tique[X_combi.edad <= 1])

X_combi["fam_bebe"] = ""

"""
tengo que crear una variable que se llama fam_bebe
si el tique esta en el set de bebe tienes un 1 Y si no es un bebe
"""

def condiFamB_train (fila):
    if (fila["tique"] in tiques_bebeDf) & (fila["bebe"] == 0):
        return True

def condiFamB_test (fila):
    if ((fila["tique"] in tiques_bebeDt) & 
        (fila["bebe"] == 0)) :
        return True

def procesaFam_bebe ():
    global X_combi
    
    X_combi["fam_bebe"].iloc[:len(df)] = X_combi.iloc[:len(df)].apply(lambda fila: 1 if condiFamB_train(fila) else 0, axis = 1)    
#    X_combi["fam_bebe"].iloc[:len(df)] = X.tique.map(lambda t: 1 if t == data.tique[data.bebe == 1])    
    X_combi.fam_bebe.iloc[len(df):] = X_combi.iloc[len(df):].apply(lambda fila: 1 if condiFamB_test(fila) else 0, axis = 1)


procesaFam_bebe()



## Procesar Pijos_C
"""
¿se puede hacer con map?
"""

def condiPijo_C (fila):
    if ((fila["puerto"] == "C") & (fila["genero"] == "hombre") & 
    (fila["tarifa"] > 55)):
        return True

def procesaPijo_C():
    global X_combi
    X_combi["pijo_C"] = X_combi.apply(lambda f: 1 if condiPijo_C(f) else 0, axis = 1)


procesaPijo_C()




## Procesar las edades
df.edad.isna().sum()
# 177
dt.edad.isna().sum()
# 86

ent_ag = X_combi.iloc[:len(df)].groupby(["genero","clase","tratamiento"])
"""
esto crea un objeto pandas
"""
ent_ag_ana = ent_ag.median()
"""
esto ya crea un dataFrame con los valores de las medianas
"""
ent_ag_ana = ent_ag_ana.reset_index()[["genero", "clase", "tratamiento", "edad"]]
"""
Esto ya selecciona las variables que forman parte del DataFrame agrupado.
se acaba de crear un data frame que permite imputar las edades faltantes de acuerdo
con el tratamiento, el genero y la clase.
"""
"""
¿Para que sirve la función reset_index? para seleccionar las variables del
dataFrame agrupado
"""

# =============================================================================
# OJO VIENE LA FUNCION **LAMBDA** PARA RELLENAR LAS NAN DE LAS EDADES 
# =============================================================================

def rellena_edad(x):
    condicion = (
        (ent_ag_ana.genero == x["genero"]) &
        (ent_ag_ana.tratamiento == x["tratamiento"]) &
        (ent_ag_ana.clase == x["clase"])
        )
    return ent_ag_ana[condicion]["edad"].values[0]

"""
¿para que el [0]?
"""

def procesado_edad():
    global X_combi
    # una funcion que rellena los nan de la variable edad
    X_combi.edad = X_combi.apply(lambda x: rellena_edad(x) if np.isnan(x["edad"]) else x["edad"], axis = 1)
    return X_combi

procesado_edad()

"""
ANTES DE ESTO TENGO QUE TENER LAS FAMILIAS DE LOS BEBES (al final lo he hecho con
los tiquets) OJO!!
"""

## Procesar los nombres

X_combi.drop("nombre", axis = 1, inplace=True)    
X_combi = pd.get_dummies(X_combi, prefix="tratamiento", columns=["tratamiento"])
    
   

## Procesar las tarifas

X_combi.tarifa.fillna(X_combi.iloc[:len(df)].tarifa.mean(), inplace = True)



## Procesar puertos

X_combi.puerto.fillna(X_combi.iloc[:len(df)].puerto.mode()[0], inplace = True)
"""
el metodo mode() crea una serie
"""
X_combi = pd.get_dummies(X_combi, prefix = "puerto", columns = ["puerto"])




## Procesar la cabina

### Crear listas con las letras de las cabinas en los conjuntos de entrenamiento 
### y test
cabina_ent, cabina_test = set(), set()

for c in X_combi.cabina.iloc[:len(df)]:
    try:
        cabina_ent.add(c[0])
    except:
        cabina_ent.add("U")
        
for c in X_combi.cabina.iloc[len(df):]:
    try:
        cabina_test.add(c[0])
    except:
        cabina_test.add("U")

cabina_ent = sorted(cabina_ent)
cabina_test = sorted(cabina_test)

"""
No aparece ninguna cabina en el conjunto de test que no aparezca en el de entrenamiento.
Si al reves. Yo creo que es un outlier...
"""

### Rellenar los NAN con el valor U
X_combi.cabina.fillna("U", inplace = True)

### Mapear las cabinas por la primera letra
X_combi.cabina = X_combi.cabina.map(lambda x: x[0])

### Obtener las variables dummy
X_combi = pd.get_dummies(X_combi, prefix="cabina", columns=["cabina"])





## Procesar genero
X_combi.genero = X_combi.genero.map({"mujer":1, "hombre": 0})



## Procesar clase
X_combi = pd.get_dummies(X_combi, prefix="clase", columns=["clase"])



## Procesado de tique
# =============================================================================
# Antes de hacer el procesado de tique ¿hay que sacar el cluster de los familiares
# de bebes? Lo dejo para la siguiente versión, voy a terminar según el ejemplo
# a ver que pasa
# =============================================================================

def limpiatique(tique):
    tique = tique.replace('.','')
    tique = tique.replace('/','')
# =============================================================================
# los reemplaza por nada    
# =============================================================================
    tique = tique.split()
# =============================================================================
# devuelve una lista con las palabras que componen la cadena de caracteres
# =============================================================================
    #tique = tique.strip()
    #error : 'list' object has no attribute 'strip'
    tique = map(lambda t : t.strip(), tique)
# =============================================================================
# devuelve un 'map object'
# =============================================================================
    tique = list(filter(lambda t : not t.isdigit(), tique))
    if len(tique) > 0:
        return tique[0]
    else:
        return 'XXX'
"""
la función filter trabaja con funciones condicionales y devuelve objetos 
booleanos. En este caso le estamos metiendo un map object de una lista.
Para obtener el valor hay que castearlo con la función list:
  + si un elemento de la lista mapeada no es un digito (por completo), 
    lo devuelve a dicha lista de la variable tique.
    Entonce tiene longitud y devuelve el primer elemento de la lista.
  + si es un digito integro (dentro de los strings que pertenecen a la lista 
    mapeada) no lo mete en la lista de la variable tique. Como la lista 
    no tiene longitud devuelve XXX.
"""    
tiques = set()
for t in X_combi.tique:
    tiques.add(limpiatique(t))

print (len(tiques))



X_combi.tique = X_combi.tique.map(limpiatique)
X_combi = pd.get_dummies(X_combi, prefix="tique", columns=["tique"])



## Procesado Familia

df["familia"] = df.Parch + df.SibSp + 1
df.familia = pd.Categorical(df.familia, ordered=True)

plt.figure(figsize=(20,20))
mosaic(df,["familia", "Superviviente"])
plt.show()
# =============================================================================
# Las familias de 2,3 y 4 miembros tienen una tasa de supervivencia mayor
# =============================================================================

X_combi["familia"] = X_combi.Parch + X_combi.SibSp + 1

def procesarFamilia():
    global X_combi
    X_combi["solitario"] = X_combi.familia.map(lambda f: 1 if f == 1 else 0)
    X_combi["familia_peq"] = X_combi.familia.map(lambda f: 1 if 2<= f <= 4 else 0)
    X_combi["familia_num"] = X_combi.familia.map(lambda f: 1 if 5 <= f else 0)

procesarFamilia()




# 3. Modelado

# =============================================================================
# PASOS
# 1. Dividir el dataset combinado
# 2. Usar el conjunto de entrenamiento para construir un modelo predictivo
# 3. Evaluar el modelo con el conjunto de entrenamiento
# 4. Comprobar el modelo usando el conjunto de test y generar el output de envio
# Habra que iterar los puntos 2 y 3 hasta conseguir una evaluación aceptable
# =============================================================================

## Para evaluar el modelo usaremos la cross validation
def procesaResultado (classifier, X_train, y_train, scoring='accuracy'):
    xval = cross_val_score(classifier, X_train, y_train, cv  = 5, scoring = scoring)
    return np.mean(xval)
"""
esto YA lo entiendo. ¿return np.mean(xval)? ¿Qué está devolviendo esta función
para hacerle la media? ¿no se puede pedir la media directamente como argumento 
de la función?
OK
xval es un array con cinco resultados para cada uno de los kfolds
"""

## Dividir X_combi
X = X_combi.iloc[:891]
x_t = X_combi.iloc[891:]



## Selección de variables
# =============================================================================
# BENEFICIOS
# 1. Disminuye la redundancia entre los datos
# 2. Acelera el proceso de entrenamiento
# 3. Reduce el "overfitting"
# Podemos usar el modelo de Random Forest procesar la importancia de las variables.
# Y a su vez descartar variables irrelevantes.
# =============================================================================

classifier = RandomForestClassifier(n_estimators=50, max_features='sqrt')
classifier = classifier.fit(X,y)

### Hacer grafico con las variables y su importancia

variables = pd.DataFrame()
variables["variable"] = X.columns
variables["importancia"] = classifier.feature_importances_
variables.sort_values(by=["importancia"], ascending=True, inplace=True)
variables.set_index("variable", inplace=True)

variables.plot(kind="barh", figsize=(25,25))
plt.show()


### Comprimir los datasets

modelo = SelectFromModel(classifier, prefit=True)
"""
Sigo sin entender que es el prefit
"""
X_redu = modelo.transform(X)

x_t_redu = modelo.transform(x_t)

print(X_redu.shape)
print(x_t_redu.shape)

# Queda por comprobar si usaremos el reducido o el grande


## Probar distintos modelos base

ksvm = SVC(kernel = "rbf", random_state = 0)
logreg = LogisticRegression()
xgb = XGBClassifier(booster = "gbtree", objective='binary:logistic')
rf = RandomForestClassifier()


modelos = [ksvm, logreg, xgb, rf]

for m in modelos:
    print("Cross-validation de: {0}".format(m.__class__))
    score=procesaResultado(classifier = m, 
                           X_train = X_redu, y_train=y, scoring ="accuracy")
    print("resultado CV = {0}".format(score))
    print("****")
    
"""
sigo sin entender los __ 
"""   

# =============================================================================
# Cross-validation de: <class 'sklearn.svm._classes.SVC'>
# resultado CV = 0.6757454020463248
# ****
# Cross-validation de: <class 'sklearn.linear_model._logistic.LogisticRegression'>
# resultado CV = 0.8181721172556651
# ****
# Cross-validation de: <class 'xgboost.sklearn.XGBClassifier'>
# resultado CV = 0.8091833532107211
# ****
# Cross-validation de: <class 'sklearn.ensemble._forest.RandomForestClassifier'>
# resultado CV = 0.8204255853367647
# ****
# 
# =============================================================================




## Tuneado de hiperparametros

# =============================================================================
# Vamos a tunear el modelo de RandomForest con el conjunto sin reducir
# =============================================================================

run_gs = False

if run_gs:
    parameter_grid = {
        'criterion' : ["entropy", "gini"],
        'max_depth' : [4,6,8,10,12],
        'n_estimators' : [10,50,100,500,1000],
        'max_features' : ['sqrt', 'auto', 'log2'],
        'min_samples_split' : [2,3,10],
        'min_samples_leaf' : [1,2,3,10],
        'bootstrap' : [True, False],
        'random_state' : [0]
        }
    bosque = RandomForestClassifier()
    cv = StratifiedKFold(n_splits=5)
    
    grid_search = GridSearchCV(bosque,
                               scoring='accuracy',
                               param_grid = parameter_grid,
                               cv = cv,
                               verbose = 1,
                               n_jobs = -1
                               )

    grid_search.fit(X,y)
    modelo = grid_search
    parametros = grid_search.best_params_
    
    print("Mejor resultado: {}".format(grid_search.best_score_))
    print("Mejores parametros: {}".format(parametros))

# 0.8383842
    
else:
    parametros = {'bootstrap': True,
                  'criterion' : 'gini',
                  'min_samples_leaf' : 1,
                  'n_estimators' : 10,
                  'min_samples_split' : 2,
                  'max_features' : 'log2',
                  'max_depth' : 6,
                  'random_state' : 0}
    
    modelo = RandomForestClassifier(**parametros)
    modelo.fit(X,y)
    print(modelo.get_params())




## Generar el output

output = modelo.predict(x_t).astype(int)
df_output = pd.DataFrame()
df_output["PassengerId"] = test.PassengerId
df_output["Survived"] = output
df_output.set_index("PassengerId", inplace = True)
df_output.to_csv("../datasets/subm_am_py_RF.8+opt_190821.csv")

# 0.78708 el mejor resultado hasta ahora dentro del 3,5% mejor
# 0.77751 al añadir más variables y dejarlo con los mismos parametros lo he estropeado
# 0.77033 con random forest optimizado





# Mezclado de modelos
# =============================================================================
# Hacerlo como una función, no?
# =============================================================================
modelos_entrenados = []
for m in modelos:
    m.fit(X,y)
    modelos_entrenados.append(m)
    
predicciones = []
for m in modelos_entrenados:
    predicciones.append(m.predict_proba(x_t)[:,1])

"""
esto no lo entiendo ¿que es predict_proba? Supongo que no devuelve 0 o 1, si no
su probabilidad.
AttributeError: predict_proba is not available when  probability=False
Hacerlo modelo por modelo
"""

df_predicciones = pd.DataFrame(predicciones).T
"""
tampoco entiendo la .T
"""
df_predicciones["out"] = df_predicciones.mean(axis=1)
"""
no entiendo de que está haciendo la media
"""
df_predicciones["PassengerId"] = test.PassengerId
df_predicciones.out = df_predicciones.out.map(lambda p: 1 if p>= 0.5 else 0)


df_predicciones = df_predicciones[["PassengerId","out"]]
df_predicciones.columns = ["PassengerId", "Survived"]
df_predicciones.set_index("PassengerId", inplace=True)













































