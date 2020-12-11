/*
Practice 5

Authors:
-Alvarez Yanez Jose Aloso 
-Quiroz Montes Yim Yetzahel

-Big Data

*/


// 1- Import libraries.
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel,RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer,VectorIndexer} 

// 2- Import a Spark Session.
import org.apache.spark.sql.SparkSession

// 3- Create a Spark session.
def main(): Unit = {
     val spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()

// 4- Load the data and create a Dataframe.
val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")

// 5- Index labels
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// 6- Identify categorical features, and index them.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// 7- Split the data
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// 8- Train the RandomForest mode
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees

// 9- Convert indexed labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// 10- Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer,featureIndexer, rf, labelConverter))

// 11- Train model
val model = pipeline.fit(trainingData)

// 12- Make predictions
val predictions = model.transform(testData)

// 13- Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// 14- Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

// 15- Print the trees obtained from the model.
val rfModel =
model.stages(2).asInstanceOf[RandomForestClassificationModel]
println(s"Learned classification forest model:\n${rfModel.toDebugString}")
}