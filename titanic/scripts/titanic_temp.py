# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

print("Hello World")

etiqueta = ["Master"]

datatrain.Age[datatrain.Name.str.contains("Master")].mean()

datatrain.Name[datatrain.Age <= 12]

datatrain.Age[datatrain.Name.str.contains("Miss")][datatrain.Age <= 12].mean()
                                                 

df.edad[df.SibSp == 0][df.genero == "hombre"][df.clase == 3][df.Superviviente == 1].mean()
                                            
df.edad[df.SibSp == 0][df.genero == "hombre"][df.clase == 1][df.Superviviente == 1].mean()



datatrain.Survived[datatrain.Fare == datatrain.Fare.max()]         



edad_master = datatrain.Age[datatrain.Name.str.contains("Master")]
edad_master = pd.DataFrame(edad_master)



df.loc[(datatrain.Age.isna()) & (datatrain.Name.str.contains("Master")), "edad"] = 4.574


df.edad[datatrain.Age.isna()][datatrain.Name.str.contains("Master")].fillna(edad_master.mean(), inplace=True)
= 5                                 


datatest.Age[datatest.Name.str.contains("Master")]
edad_master_test = datatest.Age[datatest.Name.str.contains("Master")]
edad_master_test.mean()
edad_master_test.max()

dt.loc[(datatest.Age.isna()) & (datatest.Name.str.contains("Master")), "edad"] = 7.406


X[X.cubierta_T == 1]
