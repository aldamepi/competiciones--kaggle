#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 23:05:01 2021

@author: albertomengual
"""

## Detectar y eliminar outliers

###### Intentar buscar outliers de datos agrupados: puerto nan, cubierta T, edad-supervivientes, edad-adultos, tarifas-clase
###### Crear una categoría senior??

### Detectar y eliminar outliers de la variable edad, sin agrupar:
    
#### Calculo de estadisticos
"""
Q1_edad = df.edad.quantile(0.25)

Q3_edad = df.edad.quantile(0.75)

IQR_edad = Q3_edad-Q1_edad

#### Calculo de los bigotes

BI_edad = Q1_edad - 1.5 * IQR_edad

BS_edad = Q3_edad + 1.5 * IQR_edad

#### Ubicación de outliers

ubicacion_outliers = (df.edad < BI_edad) | (df.edad > BS_edad)
 
outliers = df[ubicacion_outliers]
"""

### Detectar y eliminar outliers de la variable edad, agrupada por adultos:
    
#### Calculo de estadisticos jovenes
"""
Q1_edad_joven = df.edad[df.adulto == "joven"].quantile(0.25)

Q3_edad_joven = df.edad[df.adulto == "joven"].quantile(0.75)

IQR_edad_joven = Q3_edad_joven-Q1_edad_joven

#### Calculo de los bigotes jovenes

BI_edad_joven = Q1_edad_joven - 1.5 * IQR_edad_joven

BS_edad_joven = Q3_edad_joven + 1.5 * IQR_edad_joven

#### Ubicación de outliers jovenes

#ubicacion_outliers = (df.edad < BI_edad) | (df.edad > BS_edad)
 
#outliers = df[ubicacion_outliers]
"""

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
 
outliers_adulto = df[ubi_outliers_adulto]



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




### Detectar y eliminar outliers de la variable adultos, agrupada por fallecidos:

#### Calculo de estadisticos adultos-Fallecidos 

Q1_ed_ad_falle = df.edad[df.adulto == "adulto"][df.Superviviente == 0].quantile(0.25)

Q3_ed_ad_falle = df.edad[df.adulto == "adulto"][df.Superviviente == 0].quantile(0.75)

IQR_ed_ad_falle = Q3_ed_ad_falle-Q1_ed_ad_falle

#### Calculo de los bigotes adultos-Fallecidos

BI_ed_ad_falle = Q1_ed_ad_falle - 1.5 * IQR_ed_ad_falle

BS_ed_ad_falle = Q3_ed_ad_falle + 1.5 * IQR_ed_ad_falle

#### Ubicación de outliers adultos-Fallecidos

ubi_outliers_adulto_falle = (df.edad[df.adulto == "adulto"][df.Superviviente == 0] < BI_ed_ad_falle) | (df.edad[df.adulto == "adulto"][df.Superviviente == 0] > BS_ed_ad_falle)
 
outliers_adulto_falle = df[df.adulto == "adulto"][df.Superviviente == 0][ubi_outliers_adulto_falle]


#### Eliminación de outliers adultos-Fallecidos

df.drop(outliers_adulto_falle.index, axis=0, inplace=True)



### Detectar y eliminar outliers de la categoria senior


#### Calculo de estadisticos adultos

Q1_edad_senior = df.edad[df.adulto == "senior"].quantile(0.25)

Q3_edad_senior = df.edad[df.adulto == "senior"].quantile(0.75)

IQR_edad_senior = Q3_edad_senior-Q1_edad_senior


#### Calculo de los bigotes adultos

BI_edad_senior = Q1_edad_senior - 1.5 * IQR_edad_senior

BS_edad_senior = Q3_edad_senior + 1.5 * IQR_edad_senior


#### Ubicación de outliers adultos

ubi_outliers_senior = (df.edad[df.adulto == "senior"] < BI_edad_senior) | (df.edad[df.adulto == "senior"] > BS_edad_senior)
 
outliers_senior = df[df.adulto == "senior"][ubi_outliers_senior]

#### Eliminación de outliers senior

df.drop(outliers_senior.index, axis=0, inplace=True)


### Detectar y eliminar outliers de la variable adultos, agrupada por fallecidos; segunda iteracion
###### NO me los voy a cargar
"""
#### Calculo de estadisticos adultos-Fallecidos 

Q1_ed_ad_falle = df.edad[df.adulto == "adulto"][df.Superviviente == 0].quantile(0.25)

Q3_ed_ad_falle = df.edad[df.adulto == "adulto"][df.Superviviente == 0].quantile(0.75)

IQR_ed_ad_falle = Q3_ed_ad_falle-Q1_ed_ad_falle

#### Calculo de los bigotes adultos-Fallecidos

BI_ed_ad_falle = Q1_ed_ad_falle - 1.5 * IQR_ed_ad_falle

BS_ed_ad_falle = Q3_ed_ad_falle + 1.5 * IQR_ed_ad_falle

#### Ubicación de outliers adultos-Fallecidos

ubi_outliers_adulto_falle = (df.edad[df.adulto == "adulto"][df.Superviviente == 0] < BI_ed_ad_falle) | (df.edad[df.adulto == "adulto"][df.Superviviente == 0] > BS_ed_ad_falle)
 
outliers_adulto_falle = df[df.adulto == "adulto"][df.Superviviente == 0][ubi_outliers_adulto_falle]


#### Eliminación de outliers adultos-Fallecidos

df.drop(outliers_adulto_falle.index, axis=0, inplace=True)
"""

### Detectar y eliminar outliers de la categoria cubierta T

df.drop(df[df.cubierta == "T"].index, axis=0, inplace=True)




### Detectar y eliminar outliers de la categoria tarifa agrupada por clases

#### Calculo de estadisticos clase 3

Q1_tarifa_3 = df.tarifa[df.clase == 3].quantile(0.25)

Q3_tarifa_3 = df.tarifa[df.clase == 3].quantile(0.75)

IQR_tarifa_3 = Q3_tarifa_3-Q1_tarifa_3

#### Calculo de los bigotes clase 3

BI_tarifa_3 = Q1_tarifa_3 - 1.5 * IQR_tarifa_3

BS_tarifa_3 = Q3_tarifa_3 + 1.5 * IQR_tarifa_3

#### Ubicación de outliers clase 3

ubi_outliers_tarifa_3 = (df.tarifa[df.clase == 3] < BI_tarifa_3) | (df.tarifa[df.clase == 3] > BS_tarifa_3)
 
outliers_tarifa_3 = df[df.clase == 3][ubi_outliers_tarifa_3]

#### Eliminación de outliers clase 3

df.drop(outliers_adulto_super.index, axis=0, inplace=True)



