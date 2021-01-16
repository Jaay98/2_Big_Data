//Authors: Alonso Alvarez, Yim Quiroz
//Instituto Tecnologico de Tijuana
//Materia: Datos Masivos
//Proyecto

//Import libraries
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics

//Minimize errors 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Define val on runtime and startTimeMillis
val runtime = Runtime.getRuntime
val startTimeMillis = System.currentTimeMillis()

// Start a session in spark
val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

// The data is loaded into the variable "data" in the format "bank-full.csv""
val training = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(training)
val indexed = labelIndexer.transform(training).drop("y").withColumnRenamed("indexedLabel", "label")

// Generate a vector with the names of the columns to evaluate
val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))
//Define the vectorFeatures object to transform vectorFeatures.transform(newDF)
val features = vectorFeatures.transform(indexed)
// Rename the column ("y" , Label)
val featuresLabel = features.withColumnRenamed("y", "label")
//index the label and features columns
val dataIndexed = featuresLabel.select("label","features")
// index the labels
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed) // features with > 4 distinct values are treated as continuous.

//divide the data, 70% training and 30% testing
val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 12345)

val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
    //Define model trainig data 70%
val lsvcModel = lsvc.fit(training)

//Define results trainig data 30%
val results = lsvcModel.transform(test)



val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)
//Print confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)
//Print confusion Accurancy
println("Accurancy: " + metrics.accuracy) 
println(s"Test Error = ${(1.0 - metrics.accuracy)}")

// 1mb -> 1e6 bytes
val mb = 0.000001
println("Used Memory: " + (runtime.totalMemory - runtime.freeMemory) * mb)
println("Free Memory: " + runtime.freeMemory * mb)
println("Total Memory: " + runtime.totalMemory * mb)
println("Max Memory: " + runtime.maxMemory * mb)

val endTimeMillis = System.currentTimeMillis()
val durationSeconds = (endTimeMillis - startTimeMillis) / 1000
//Print Coefficients and intercept
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")