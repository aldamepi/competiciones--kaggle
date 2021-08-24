#SCRIPT DE PRUEBAS

datasocial = read.csv('Social_Network_Ads.csv')

pred_prueba = data.frame(dataset$Superviviente)
test_prueba = data.frame(dataset$Superviviente)

# eam = abs(as.numeric(pred_prueba)-as.numeric(test_prueba))/dim(test_prueba)[1]

errordata = cbind(pred_prueba,test_prueba)

errores = sapply(errordata,
       function(f) (abs(as.numeric(errordata[f,1])-as.numeric(errordata[f,2])
                        )))

eam = sum(errores[,1])/dim(errores)[1]



test_prueba[1:10,] = ifelse(test_prueba[1:10,] == "1", "0","1")




