## Proyect
##### -Big Data

#### Authors:
-Alvarez Yanez Jose Alonso 
-Quiroz Montes Yim Yetzhael 



#### Desicion Tree

1. Import libraries
~~~
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer 
import org.apache.spark.ml.feature.VectorAssembler
~~~

2. Minimize errors 
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~ 

3. Define val on runtime and system
~~~
val runtime = Runtime.getRuntime
val startTimeMillis = System.currentTimeMillis()
~~~

4. Start Spark session
~~~
val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()
~~~

5. Define dataframe ("df") 
~~~
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")
~~~
6. Print schema
~~~
data.printSchema()
~~~
7. Define vectorFeatures with the values "balance","day","duration","pdays","previous"
~~~
val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))
~~~
8. Use the vectorFeatures object to transform feature_data
~~~
val features = vectorFeatures.transform(indexed)
~~~

9. Define featuresLabelwithColumnsRenamed
~~~
val featuresLabel = features.withColumnRenamed("y", "label")
~~~

10. Index the label and features columns
~~~
val dataIndexed = featuresLabel.select("label","features")
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed) // features with > 4 distinct values are treated as continuous.
val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))
~~~

11. Define dt in a DecisionTree
~~~
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
~~~

12. Define pipeline and save everything in pipeline
~~~
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
~~~

13. Model trainigData
~~~
val model = pipeline.fit(trainingData)
~~~

14. Make a prediction
~~~
val predictions = model.transform(testData)
predictions.select("predictedLabel", "label", "features").show(5)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
~~~

15. Define accuracy and evaluator predictions
~~~
val accuracy = evaluator.evaluate(predictions)
~~~

16. Print test error
~~~
println(s"Test Error = ${(1.0 - accuracy)}")
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
~~~

17. Print Learned classification tree model:
~~~
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

val mb = 0.000001
println("Used Memory: " + (runtime.totalMemory - runtime.freeMemory) * mb)
println("Free Memory: " + runtime.freeMemory * mb)
println("Total Memory: " + runtime.totalMemory * mb)
println("Max Memory: " + runtime.maxMemory * mb)


val endTimeMillis = System.currentTimeMillis()
val durationSeconds = (endTimeMillis - startTimeMillis) / 1000
~~~

#### Logistic Regression


1. Import libraries
~~~
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
~~~ 

2. Minimize errors
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~

3. Create a spark session
~~~
val spark = SparkSession.builder().getOrCreate()
~~~

4. Load  CSV 
~~~
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("C:/Users/alons/OneDrive/Escritorio/Universidad/Datos Masivos/2_Big_Data/Project/Logistic Regression/bank-full.csv")
~~~

5. Print schema
~~~
df.printSchema()
~~~

6. Show Dataframe
~~~
df.show()
~~~

7. Modify the column of strings to numeric data
~~~
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
~~~
8. Show the new column
~~~
newcolumn.show()
~~~
9. Generate the features table
~~~
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
~~~
10. Show the new column
~~~
fea.show()
~~~
11. Change the column y to the label column
~~~
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)
~~~
12. Logistic Regression algorithm
~~~
val logistic = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
~~~
13. Fit of the model
~~~
val logisticModel = logistic.fit(feat)
~~~
14. Impression of coefficients and interception
~~~
println(s"Coefficients: ${logisticModel.coefficients} Intercept: ${logisticModel.intercept}")
val logisticMult = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val logisticMultModel = logisticMult.fit(feat)
println(s"Multinomial coefficients: ${logisticMultModel.coefficientMatrix}")
println(s"Multinomial intercepts: ${logisticMultModel.interceptVector}")
~~~

#### MultiLayer Perceptron

















