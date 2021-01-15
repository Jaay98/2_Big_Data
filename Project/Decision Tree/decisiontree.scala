//Authors: Alonso Alvarez, Yim Quiroz
//Instituto Tecnologico de Tijuana
//Materia: Datos Masivos
//Proyecto


//Import libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer 
import org.apache.spark.ml.feature.VectorAssembler

//Minize errors 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Define val on runtime and system
val runtime = Runtime.getRuntime
val startTimeMillis = System.currentTimeMillis()

//Start Spark session

val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()
    
//Define dataframe ("df") 
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

//Print schema
data.printSchema()

//Define vectorFeatures with the values "balance","day","duration","pdays","previous"
val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))
//Define  9. Use the vectorFeatures object to transform feature_data
val features = vectorFeatures.transform(indexed)

// 
val featuresLabel = features.withColumnRenamed("y", "label")


val dataIndexed = featuresLabel.select("label","features")


val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed) // features with > 4 distinct values are treated as continuous.


val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))


val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")


val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))


val model = pipeline.fit(trainingData)


val predictions = model.transform(testData)


predictions.select("predictedLabel", "label", "features").show(5)


val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)

println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]

println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

val mb = 0.000001
println("Used Memory: " + (runtime.totalMemory - runtime.freeMemory) * mb)
println("Free Memory: " + runtime.freeMemory * mb)
println("Total Memory: " + runtime.totalMemory * mb)
println("Max Memory: " + runtime.maxMemory * mb)


val endTimeMillis = System.currentTimeMillis()
val durationSeconds = (endTimeMillis - startTimeMillis) / 1000