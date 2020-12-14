/*
Evaluacion Unidad_2

Authors:
-Alvarez Yanez Jose Alonso 
-Quiroz Montes Yim Yetzhael

-Big Data


Desarrollar las siguientes instrucciones en Spark con el leguaje de programación Scala, 
utilizando solo la documentacion de la librería de Machine Learning  Mllib de Spark y Google.
    1. Cargar en un dataframe Iris.csv que se encuentra en https://github.com/jcromerohdz/iris, elaborar la liempieza de datos
    necesaria para ser procesado por el siguiente algoritmo (Importante, esta limpieza debe ser por medio de un script de Scala en Spark) .
       a. Utilice la librería Mllib de Spark el algoritmo de Machine Learning correspondiente a multilayer perceptron
    2. ¿Cuáles son los nombres de las columnas?
    3. ¿Cómo es el esquema?
    4. Imprime las primeras 5 columnas.
    5. Usa el metodo describe () para aprender mas sobre los datos del  DataFrame.
    6. Haga la transformación pertinente para los datos categoricos los cuales seran nuestras etiquetas a clasificar.
    7. Construya el modelo de clasificación y explique su arquitectura.
    8. Imprima los resultados del modelo


*/

// Import the libraries that we are going to use

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors

// 1- Load the dataframe in a variable

val df = spark.read.option("header","true").option("inferSchema","true").csv("C:/Users/alons/OneDrive/Escritorio/Universidad/Datos Masivos/2_Big_Data/Unit_2/Evaluations/Iris.csv")

// YIM's path "C:/Users/ASUS S510U/OneDrive/Documentos/8vo Semestre/Datos_Masivos/2_Big_Data/Unit_2/Evaluations/Iris.csv"
// ALONSO's path  "C:/Users/alons/OneDrive/Escritorio/Universidad/Datos Masivos/2_Big_Data/Unit_2/Evaluations/Iris.csv"

// 2- Showing the name of the columns

df.show(5) 

// 3- Showing the schema

df.printSchema()

// 4- Showing the first 5 columns

df.columns

// 5- Using the "describe()" metod for know more about the dataframe

df.describe().show()


///////////////////////////////////////////////////////////////////////////////////////

// Clean the data deleting the null fields and adding it to  a new dataframe called "cleanData"

val cleanData = df.na.drop()



// We are going to add a new headers
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))
val features = vectorFeatures.transform(cleanData)
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
val dataIndexed = speciesIndexer.fit(features).transform(features)


val splits = features.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = splits(0)
val test = splits(1)

val layers = Array[Int](4, 5, 4, 3)

val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)  

// We train the model 

val model = trainer.fit(train)

//Transform the model with transform.(test)

val result = model.transform(test)

val predictionAndLabels = result.select("prediction", "label")

val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

