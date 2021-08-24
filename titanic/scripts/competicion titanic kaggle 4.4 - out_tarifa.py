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
#    sb.catplot(x='clase',y ='Superviviente', hue = 'genero',data=df)
#    plt.show()
        
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
plt.title("Boxplot de Edad")
plt.show()


sb.boxplot(y = 'tarifa', data = df)
plt.title("Boxplot de Tarifa")
plt.show()


#### Analisis bidimensional cualitativo
sb.boxplot(x = 'Superviviente', y = 'edad', data = df)
plt.title("Boxplot de Edad agrupada por Supervivencia")
plt.show()

sb.boxplot(x = 'Superviviente', y = 'tarifa', data = df)
plt.title("Boxplot de Tarifa agrupada por Supervivencia")
plt.show()



sb.boxplot(x = 'clase', y = 'tarifa', data = df)
plt.title("Boxplot de tarifa agrupada por clase")
plt.show()



sb.boxplot(x = 'clase', y = 'tarifa', hue = "Superviviente", data = df)
plt.title("Boxplot de tarifa agrupada por clase y supervivencia")
plt.show()

sb.boxplot(x = 'cubierta', y = 'tarifa', hue = "Superviviente", data = df)
plt.title("Boxplot de tarifa agrupada por cubierta y supervivencia")
plt.show()

"""
###### Necesario crear la variable adulto

sb.boxplot(x = 'adulto', y = 'edad', data = df)
plt.title("Boxplot de edad agrupada por adulto")
plt.show()

sb.boxplot(x = 'genero', y = 'edad', hue  = "Superviviente", data = df)
plt.title("Boxplot de edad agrupada por genero y supervivencia")
plt.show()

sb.boxplot(x = 'genero', y = 'edad', hue  = "adulto", data = df)
plt.title("Boxplot de edad agrupada por genero y adulto")
plt.show()

sb.boxplot(x = 'Superviviente', y = 'edad', hue  = "adulto", data = df)
plt.title("Boxplot de edad agrupada por adulto y supervivencia")
plt.show()
"""



#### Analisis Multivariante cualitativo

###### Los diagramas de violin son un poco cutres, buscar otros
"""
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
"""

sb.pairplot(df[cuanti_col])
plt.show()


#### Histogramas

###### Intentar ponerlos más bonitos, meterles colorines y titulos. Intentar añadir el rug
###### Les falta las densidades, concepto que aún no entiendo


###### falta meter alguna pivot.table, no? - 
###### para hacer las pivot table, en concreto los cut, hay que quitar los nan
df.edad.plot.hist(bins=20, edgecolor='black')
plt.title("Histograma Edades")
plt.show()

df[df.Superviviente == 0].edad.plot.hist(bins=20, edgecolor='black')
plt.title("Histograma Edades de los Fallecidos")
plt.show()

#vector_edad = pd.cut(df["edad"], bins=20, include_lowest=True, right=False)
#tab_hist_edad = df.pivot_table(df.edad, index=df.Superviviente, columns=vector_edad)
#print(tab_hist_edad)

df[df.Superviviente == 1].edad.plot.hist(bins=20, edgecolor='black')
plt.title("Histograma Edades de los Supervivientes")
plt.show()

df.tarifa.plot.hist(bins = 100, edgecolor='black')
plt.title("Histograma de tarifas")
plt.show()

df[df.Superviviente == 0].tarifa.plot.hist(bins=50, edgecolor='black')
plt.title("Histograma de tarifas fallecidos")
plt.show()

df[df.Superviviente == 1].tarifa.plot.hist(bins=100, edgecolor='black')
plt.title("Histograma de tarifas de Supervivientes")
plt.show()


#### Conclusiones del Analisis Cualitativo

#### Los niños varones tienen un alto indice de supervivencia.








# PREPARAR LOS DATOS PARA EL MODELO



## Gestión de Variables: Edición de variables a entrenar y separar matriz de caracteristicas
### Crear variable adulto? O añadir niño a genero - creo la variable
df.columns

df.loc[(df.edad <= 15), "adulto"] = "joven"
df.adulto.fillna("adulto", inplace=True)

pd.crosstab(index=[df.Superviviente, df.genero], columns=[df.clase, df.adulto], margins=True)


#### En el conjunto de test
dt.loc[(dt.edad <= 15), "adulto"] = "joven"
dt.adulto.fillna("adulto", inplace=True)


df.adulto = pd.Categorical(df.adulto
#                           , categories=["joven", "adulto", "senior"]
                           )
dt.adulto = pd.Categorical(dt.adulto
#                           , categories=["joven", "adulto", "senior"]
                           )

cual_col.append("adulto")

"""
Haciendolo paso a paso, sin incluir los fillna=mean en edades, parece que no necesito la variable senior
df.loc[(df.edad > 52.5), "adulto"] = "senior"
dt.loc[(df.edad > 52.5), "adulto"] = "senior"
"""




## Ampliacion del analisis bidimensional

sb.boxplot(x = 'adulto', y = 'edad', data = df)
plt.title("Boxplot de edad agrupada por adulto")
plt.show()

sb.boxplot(x = 'genero', y = 'edad', hue  = "Superviviente", data = df)
plt.title("Boxplot de edad agrupada por genero y supervivencia")
plt.show()

sb.boxplot(x = 'genero', y = 'edad', hue  = "adulto", data = df)
plt.title("Boxplot de edad agrupada por genero y adulto")
plt.show()

sb.boxplot(x = 'Superviviente', y = 'edad', hue  = "adulto", data = df)
plt.title("Boxplot de edad agrupada por adulto y supervivencia")
plt.show()






## Detectar y eliminar outliers

###### Intentar buscar outliers de datos agrupados: puerto nan, cubierta T, edad-supervivientes, edad-adultos, tarifas-clase
###### Crear una categoría senior??


### Detectar y eliminar outliers de la categoria cubierta T

df.drop(df[df.cubierta == "T"].index, axis=0, inplace=True)

df.cubierta = pd.Categorical(df.cubierta, categories=["A","B","C","D","E","F","G","Unknown"])


### Detectar y eliminar outliers de la categoria adulto

#### Calculo de estadisticos adultos

Q1_edad_adulto = df.edad[df.adulto == "adulto"].quantile(0.25)

Q3_edad_adulto = df.edad[df.adulto == "adulto"].quantile(0.75)

IQR_edad_adulto = Q3_edad_adulto-Q1_edad_adulto

#### Calculo de los bigotes adultos

BI_edad_adulto = Q1_edad_adulto - 1.5 * IQR_edad_adulto

BS_edad_adulto = Q3_edad_adulto + 1.5 * IQR_edad_adulto

#### Ubicación de outliers adultos

ubi_outliers_adulto = (df.edad[df.adulto == "adulto"] < BI_edad_adulto) | (df.edad[df.adulto == "adulto"] > BS_edad_adulto)
 
outliers_adulto = df[df.adulto == "adulto"][ubi_outliers_adulto]

#### Eliminación de outliers adultos-Supervivientes

df.drop(outliers_adulto.index, axis=0, inplace=True)




### Detectar y eliminar outliers de la variable adultos, agrupada por genero:

#### Calculo de estadisticos adultos-hombres 

Q1_ed_ad_hombre = df.edad[df.adulto == "adulto"][df.genero == "hombre"].quantile(0.25)

Q3_ed_ad_hombre = df.edad[df.adulto == "adulto"][df.genero == "hombre"].quantile(0.75)

IQR_ed_ad_hombre = Q3_ed_ad_hombre-Q1_ed_ad_hombre

#### Calculo de los bigotes adultos-hombres

BI_ed_ad_hombre = Q1_ed_ad_hombre - 1.5 * IQR_ed_ad_hombre

BS_ed_ad_hombre = Q3_ed_ad_hombre + 1.5 * IQR_ed_ad_hombre

#### Ubicación de outliers adultos-Supervivientes

ubi_outliers_adulto_hombre = (df.edad[df.adulto == "adulto"][df.genero == "hombre"] < BI_ed_ad_hombre) | (df.edad[df.adulto == "adulto"][df.genero == "hombre"] > BS_ed_ad_hombre)
 
outliers_adulto_hombre = df[df.adulto == "adulto"][df.genero == "hombre"][ubi_outliers_adulto_hombre]


#### Eliminación de outliers adultos-Supervivientes

df.drop(outliers_adulto_hombre.index, axis=0, inplace=True)




### Detectar y eliminar outliers de la variable adultos, agrupada por supervivientes:

#### Calculo de estadisticos adultos-Supervivientes 

Q1_ed_ad_super = df.edad[df.adulto == "adulto"][df.Superviviente == 1].quantile(0.25)

Q3_ed_ad_super = df.edad[df.adulto == "adulto"][df.Superviviente == 1].quantile(0.75)

IQR_ed_ad_super = Q3_ed_ad_super-Q1_ed_ad_super

#### Calculo de los bigotes adultos-Supervivientes

BI_ed_ad_super = Q1_ed_ad_super - 1.5 * IQR_ed_ad_super

BS_ed_ad_super = Q3_ed_ad_super + 1.5 * IQR_ed_ad_super

#### Ubicación de outliers adultos-Supervivientes

ubi_outliers_adulto_super = (df.edad[df.adulto == "adulto"][df.Superviviente == 1] < BI_ed_ad_super) | (df.edad[df.adulto == "adulto"][df.Superviviente == 1] > BS_ed_ad_super)
 
outliers_adulto_super = df[df.adulto == "adulto"][df.Superviviente == 1][ubi_outliers_adulto_super]


#### Eliminación de outliers adultos-Supervivientes

df.drop(outliers_adulto_super.index, axis=0, inplace=True)



### SEGUNDA ITERACION eliminar outliers de la variable adultos, agrupada por supervivientes:

#### Calculo de estadisticos adultos-Supervivientes 

Q1_ed_ad_super = df.edad[df.adulto == "adulto"][df.Superviviente == 1].quantile(0.25)

Q3_ed_ad_super = df.edad[df.adulto == "adulto"][df.Superviviente == 1].quantile(0.75)

IQR_ed_ad_super = Q3_ed_ad_super-Q1_ed_ad_super

#### Calculo de los bigotes adultos-Supervivientes

BI_ed_ad_super = Q1_ed_ad_super - 1.5 * IQR_ed_ad_super

BS_ed_ad_super = Q3_ed_ad_super + 1.5 * IQR_ed_ad_super

#### Ubicación de outliers adultos-Supervivientes

ubi_outliers_adulto_super = (df.edad[df.adulto == "adulto"][df.Superviviente == 1] < BI_ed_ad_super) | (df.edad[df.adulto == "adulto"][df.Superviviente == 1] > BS_ed_ad_super)
 
outliers_adulto_super = df[df.adulto == "adulto"][df.Superviviente == 1][ubi_outliers_adulto_super]


#### Eliminación de outliers adultos-Supervivientes

df.drop(outliers_adulto_super.index, axis=0, inplace=True)








## Tratamiento de los NA
"""
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0) 
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:,1:3])
"""
### En el conjunto de entrenamiento

edad_master = datatrain.Age[datatrain.Name.str.contains("Master")]
edad_master.mean()

df.loc[(datatrain.Age.isna()) & (datatrain.Name.str.contains("Master")), "edad"] = 4.574



#### Voy a intentar tratar los NA siguiendo una distribucion normal
from scipy.stats import truncnorm

##### Para los hombres adultos
media_na_hombres = df.edad[df.adulto == "adulto"][df.genero == "hombre"].mean()
std_na_hombres = df.edad[df.adulto == "adulto"][df.genero == "hombre"].std()

tam_na_hombres = df.edad[df.adulto == "adulto"][df.genero == "hombre"].isna().sum()


a,b = (15 - media_na_hombres)/std_na_hombres, (BS_edad_adulto - media_na_hombres)/std_na_hombres

#a,b = (15-31.61)/11.32, (65.5-31.61)/11.32

edades_normales_hombres = truncnorm.rvs(a, b, media_na_hombres, std_na_hombres, tam_na_hombres,
                                        random_state=2021)



df.loc[(df.edad.isna()) & (df.adulto == "adulto")
       & (df.genero == "hombre"), "edad"] = edades_normales_hombres


##### Para las mujeres adultas
###### Aqui voy a dejar que me las meta en edades de niñas(porque los masters son solo niños)
media_na_mujer = df.edad[df.adulto == "adulto"][df.genero == "mujer"].mean()
std_na_mujer = df.edad[df.adulto == "adulto"][df.genero == "mujer"].std()

tam_na_mujer = df.edad[df.adulto == "adulto"][df.genero == "mujer"].isna().sum()


a,b = (0 - media_na_mujer)/std_na_mujer, (BS_edad_adulto - media_na_mujer)/std_na_mujer

edades_normales_mujer = truncnorm.rvs(a, b, media_na_mujer, std_na_mujer, tam_na_mujer,
                                        random_state=2021)


df.loc[(df.edad.isna()) & (df.adulto == "adulto")
       & (df.genero == "mujer"), "edad"] = edades_normales_mujer

##### Ahora hay que volver a modificar las categorias de adulto para el genero mujer

df.loc[(df.edad <= 15) & (df.genero == "mujer"), "adulto"] = "joven"


#df.edad.fillna(df.edad.mean(), inplace=True)
###### edades_normales = np.random.normal(31.61,11.32,173)



### En el conjunto de test ¿Hay variable adulto en el conjunto de test? - si,hay

edad_master_test = datatest.Age[datatest.Name.str.contains("Master")]
edad_master_test.mean()
edad_master_test.max()

dt.loc[(datatest.Age.isna()) & (datatest.Name.str.contains("Master")), "edad"] = 7.406


#### NA Distribucion normal conjunto de test
##### Para los hombres adultos
media_na_hombres_test = dt.edad[dt.adulto == "adulto"][dt.genero == "hombre"].mean()
std_na_hombres_test = dt.edad[dt.adulto == "adulto"][dt.genero == "hombre"].std()

tam_na_hombres_test = dt.edad[dt.adulto == "adulto"][dt.genero == "hombre"].isna().sum()


a_t,b_t = (15 - media_na_hombres_test)/std_na_hombres_test, (BS_edad_adulto - media_na_hombres_test)/std_na_hombres_test

#a,b = (15-31.61)/11.32, (65.5-31.61)/11.32

edades_normales_hombres_test = truncnorm.rvs(a_t, b_t, media_na_hombres_test, std_na_hombres_test, 
                                        tam_na_hombres_test, random_state=2021)



dt.loc[(dt.edad.isna()) & (dt.adulto == "adulto")
       & (dt.genero == "hombre"), "edad"] = edades_normales_hombres_test


##### Para las mujeres adultas
###### Aqui voy a dejar que me las meta en edades de niñas(porque los masters son solo niños)
media_na_mujer_t = dt.edad[dt.adulto == "adulto"][dt.genero == "mujer"].mean()
std_na_mujer_t = dt.edad[dt.adulto == "adulto"][dt.genero == "mujer"].std()

tam_na_mujer_t = dt.edad[dt.adulto == "adulto"][dt.genero == "mujer"].isna().sum()


a_t,b_t = (0 - media_na_mujer_t)/std_na_mujer_t, (BS_edad_adulto - media_na_mujer_t)/std_na_mujer_t

edades_normales_mujer_t = truncnorm.rvs(a_t, b_t, media_na_mujer_t, std_na_mujer_t, tam_na_mujer_t,
                                        random_state=2021)


dt.loc[(dt.edad.isna()) & (dt.adulto == "adulto")
       & (dt.genero == "mujer"), "edad"] = edades_normales_mujer_t

##### Ahora hay que volver a modificar las categorias de adulto para el genero mujer

dt.loc[(dt.edad <= 15) & (dt.genero == "mujer"), "adulto"] = "joven"


dt.edad[dt.adulto == "adulto"].mean()
dt.edad[dt.adulto == "adulto"].std()

dt.edad.isna().sum()

edades_normales_test = np.random.normal(32.74,12.51,86)

dt.loc[(dt.edad.isna()) & (dt.adulto == "adulto"), "edad"] = abs(edades_normales_test)


#dt.edad.fillna(dt.edad.mean(), inplace=True)

#### Hay un NAN en tarifa

dt.tarifa.fillna(dt.tarifa.mean(), inplace=True)



###### Volver a definir la variable adulto de la categoria adulto





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

X = x0.iloc[0:873,:]
x_t = x0.iloc[873:,:]

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

###### 0.777
###### 0.777


# CREAR EL OUTPUT

## Output RandomForests

classiRF_out = classiRF.fit(X,y)

y_out_RF = classiRF_out.predict(x_t)

y_out_RF = pd.DataFrame(y_out_RF, columns=["Survived"])

outRF = pd.concat([datatest.PassengerId, y_out_RF], axis=1)
outRF.set_index('PassengerId', inplace=True)

outRF.to_csv('../datasets/subm_am_py_rf_070821.csv')
##### 0.73923
##### 0.744 (Después de limpiar outliers edades)











