#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importar librerias pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover

#crear sesion spark
spark = SparkSession.builder.appName("Analisis_Opinion_PeruARusia")            .config("spark.some.config.option","some-value")            .getOrCreate()


# In[2]:


#cargar dataset en DataFrame
tuits_peruArusia = spark.read.csv("data/PeruARusia2018.csv", inferSchema = True, header = True)
tuits_peruArusia.show(2)


# In[3]:


#seleccionar los datos necesarios para el modelo de aprendizaje
tuitsData = tuits_peruArusia.select(col("polaridad").cast("Int").alias("label"),"texto")
tuitsData.show(2)


# In[23]:


tuitsData.show(10, False)


# In[4]:


#1.-limpiando texto, remplazando vocales atildadas por vocales simples
dataSinTilde = tuitsData.select(translate(col("texto"),"áéíóú","aeiou")                                .alias("textoSinTilde")                                ,col("label"))

#2 y 3.-limpiando texto, quitando @usuarios y #etiqueta
patron = "@[A-Za-z0-9]+|#[A-Za-z0-9]+"
dataSinUserEtiqueta = dataSinTilde.select(regexp_replace(col("textoSinTilde"),patron,"")                                          .alias("textoSinUsEt")                                          ,col("label"))

#4.-limpiando texto, quitando pic.twitter.com/... goo.gl/... 
# twitter.com/../.. /status/... fb.me/...
patron="pic.twitter.com/[A-Za-z0-9]+|goo.gl/[A-Za-z0-9]+|twitter.com/[A-Za-z0-9]+/[A-Za-z0-9]+|twitter.com/[A-Za-z0-9]+|/status/[A-Za-z0-9]+|fb.me/[A-Za-z0-9]+"
dataSinURL = dataSinUserEtiqueta.select(regexp_replace(col("textoSinUsEt"),patron,"")                                        .alias("textoSinURL")                                        ,col("label"))

#5.-limpiando texto, quitando tatus/... word.word.com/ http:// https://
patron="http://|https://|[A-Za-z0-9]+/[A-Za-z0-9]+|[A-Za-z0-9]+.[A-Za-z0-9]+.[A-Za-z0-9]+/[A-Za-z0-9]+"
dataSinURLs = dataSinURL.select(regexp_replace(col("textoSinURL"),patron,"")                                .alias("textoSinURLs")                                ,col("label"))

#6.-limpiando texto, quitando simbolos, signos de puntuacion
patron="\"|\.|,|;|:|¿|\?|¡|!|=|-|/|…"
dataSinPunt = dataSinURLs.select(regexp_replace(col("textoSinURLs"),patron,"")                                 .alias("textoSinPunt")                                 ,col("label"))

#7.-limpiando texto, quitando espacios en blanco demas y remplazarlo por un espacio
patron=" +"
dataLimpio = dataSinPunt.select(regexp_replace(col("textoSinPunt"),patron," ")                                .alias("text")                                ,col("label"))
dataLimpio.show(2, False)


# In[5]:


dataLimpio.show(10, False)


# In[6]:


#Dividir el dataset en entrenamiento=70% y prueba=30%
tuitsDividido = dataLimpio.randomSplit([0.7, 0.3])
tuitsEntrenamiento = tuitsDividido[0]
tuitsPrueba = tuitsDividido[1]
nroTuitsEntrenamiento = tuitsEntrenamiento.count()
nroTuitsPrueba = tuitsPrueba.count()
print("Tuits de Entrenamiento :",nroTuitsEntrenamiento, "Tuits de Prueba :",nroTuitsPrueba)


# In[7]:


#Preparar Datos de Entrenamiento: Tokenizacion
tokenizer = Tokenizer(inputCol = "text", outputCol="palabras")
tokenizedData = tokenizer.transform(tuitsEntrenamiento)
tokenizedData.select("palabras","label").show(5, False)


# In[8]:


tokenizedData.select("palabras","label").show(11)


# In[9]:


#Preparar Datos: quitar palabras sin importancia de spanish
spanishStopWords = StopWordsRemover.loadDefaultStopWords("spanish")
swr = StopWordsRemover(stopWords = spanishStopWords, inputCol = tokenizer.getOutputCol(), outputCol="palabrasUtil")
swrTuitsData = swr.transform(tokenizedData)


# In[10]:


swrTuitsData.select("palabrasUtil").show(6, False)


# In[11]:


#Convertir las palabras Utiles a caracteristicas numericas
hashTF = HashingTF(inputCol=swr.getOutputCol(),outputCol="features")
numericTrainData = hashTF.transform(swrTuitsData).select("label","palabrasUtil","features")


# In[12]:


numericTrainData.show(truncate=False, n=3)


# In[13]:


#Entrenar el modelo con los datos de entrenamiento
lr = LogisticRegression(labelCol="label",featuresCol="features",maxIter=10, regParam=0.01)
model = lr.fit(numericTrainData)
print("El modelo esta entrenado!")


# In[14]:


#Preparar datos de entrenamiento
tokenizedTest = tokenizer.transform(tuitsPrueba)
swrTuitsDataTest = swr.transform(tokenizedTest)
numericTest = hashTF.transform(swrTuitsDataTest).select("label","palabrasUtil","features")
numericTest.show(truncate = False, n = 3)


# In[15]:


#Predecir los datos de prueba y calcular la presicion del modelo
prediction = model.transform(numericTest)
predictionFinal = prediction.select("palabrasUtil","prediction","label")
predictionFinal.show(n=5)
correctPrediction = predictionFinal.filter(predictionFinal["label"]==predictionFinal["prediction"]).count()
totalData = predictionFinal.count()
accuracy = correctPrediction/totalData
print("Predicciones correctas :",correctPrediction, "Total Datos :", totalData, "Accuracy :",accuracy)


# In[16]:


true_positives = predictionFinal.filter('label==1 and prediction==1.0').count()
print(true_positives)


# In[17]:


true_negatives = predictionFinal.filter('label==0 and prediction==0.0').count()
print(true_negatives)


# In[18]:


false_positives = predictionFinal.filter('label==0 and prediction==1.0').count()
print(false_positives)
false_negatives = predictionFinal.filter('label==1 and prediction==0.0').count()
print(false_negatives)


# In[19]:


recall = float(true_positives)/(true_positives + false_negatives)
print("Recall :", recall)
precision = float(true_positives)/(true_positives + false_positives)
print("Precision :", precision)
accuracy = float(true_positives + true_negatives)/(totalData)
print("Accuracy :", accuracy)


# In[22]:


F1Score = 2*float(precision*recall)/(precision + recall)
print("F1 Score :", F1Score)


# In[ ]:




