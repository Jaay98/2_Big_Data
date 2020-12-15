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

// 2- Showing the the dataframe

df.show(5) 

// 3- Showing the schema

df.printSchema()

// 4- Showing the  columns

df.columns

// 5- Using the "describe()" metod for know more about the dataframe

df.describe().show()

 
///////////////////////////////////////////////////////////////////////////////////////

// Clean the data deleting the null fields and adding it to  a new dataframe called "cleanData"

val cleanData = df.na.drop()


// VectorAssembler is a transformer that combines a given list of columns into a single vector column
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features"))

//Transform fetures into a dataframe
val features = vectorFeatures.transform(cleanData)
//Example
features.show()

//StringIndexer encodes a string column of labels to a column of label indices 
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
// Fit the indexed species with the features of the vector
val dataIndexed = speciesIndexer.fit(features).transform(features)

// Pull apart the training data of the test data
//0.7 Training
//0.3 Test
val splits = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = splits(0)
val test = splits(1)

train.show()
test.show()


//Set the layer settings
val layers = Array[Int](4, 5, 4, 3)

//Set up the MultilayerPerceptronClassifier
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)  

// We train the model 

val model = trainer.fit(train)

//Transform the model with transform.(test)

//Run the model and assing to a "result" variable
val result = model.transform(test)

//Select the prediction colums
val predictionAndLabels = result.select("prediction", "label")

//Evaluate reliability
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")


//Show the result
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

