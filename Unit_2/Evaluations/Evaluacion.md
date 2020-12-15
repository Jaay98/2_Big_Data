<h1>Tecnológico Nacional de México</h1>
<h6> Instituto Tecnológico de Tijuana 

Subdirección Académica 
Departamento de Sistemas y Computación 

Semestre: Septiembre - Enero 2020-2021

Materia:
Datos Masivos

Profesor: 
Jose Christian Romero Hernandez

Alumno: 
17210526 Alvarez Yanez Jose Alonso
17210623 Quiroz Montes Yim Yetzhael


Fecha:
15/12/2020 </h6>

1. The first step is to import the libraries
```scala
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
```
2. Load the dataframe in a variable
```scala
val df = spark.read.option("header","true").option("inferSchema","true").csv("C:/Users/alons/OneDrive/Escritorio/Universidad/Datos Masivos/2_Big_Data/Unit_2/Evaluations/Iris.csv")
```
// YIM's path "C:/Users/ASUS S510U/OneDrive/Documentos/8vo Semestre/Datos_Masivos/2_Big_Data/Unit_2/Evaluations/Iris.csv"
// ALONSO's path  "C:/Users/alons/OneDrive/Escritorio/Universidad/Datos Masivos/2_Big_Data/Unit_2/Evaluations/Iris.csv"

// 2- Showing the dataframe
```scala
df.show(5) 
```
// 3- Showing the schema
```scala
df.printSchema()
```
// 4- Showing the  columns
```scala
df.columns
```
// 5- Using the "describe()" metod for know more about the dataframe
```scala
df.describe().show()
```
 
///////////////////////////////////////////////////////////////////////////////////////

// Clean the data deleting the null fields and adding it to  a new dataframe called "cleanData"
```scala
val cleanData = df.na.drop()
cleanData.show()
```
// VectorAssembler is a transformer that combines a given list of columns into a single vector column
```scala
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features"))
```
//Transform fetures into a dataframe
```scala
val features = vectorFeatures.transform(cleanData)
```
//Example
```scala
features.show()
```
//StringIndexer encodes a string column of labels to a column of label indices 
```scala
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
```
// Fit the indexed species with the features of the vector
```scala
val dataIndexed = speciesIndexer.fit(features).transform(features)
dataIndexed.show(200)
```
// Pull apart the training data of the test data
//0.7 Training
//0.3 Test
```scala
val splits = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = splits(0)
val test = splits(1)

train.show()
```

//Set the layer settings
```scala
val layers = Array[Int](4, 5, 4, 3)
```
//Set up the MultilayerPerceptronClassifier
```scala
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)  
```
// We train the model 
```scala
val model = trainer.fit(train)
```
//Transform the model with transform.(test)

//Run the model and assing to a "result" variable
```scala
val result = model.transform(test)
result.show()
```
//Select the prediction colums
```scala
val predictionAndLabels = result.select("prediction", "label")
predictionAndLabels.show()
```
//Evaluate reliability
```scala
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
```

//Show the result
```scala
println("Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```