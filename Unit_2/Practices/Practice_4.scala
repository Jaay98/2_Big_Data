/*
Practice 4

Authors:
-Alvarez Yanez Jose Aloso 
-Quiroz Montes Yim Yetzahel

-Big Data

*/

// 1- Import Pipeline
import org.apache.spark.ml.Pipeline

// 2- Import DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassificationModel

// 3- Import DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassifier

// 4- Import MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// 5- Import IndexToString, StringIndexer, VectorIndexer
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
 
// 6- Import a SparkSession
import org.apache.spark.sql.SparkSession

// 7- Create a SparkSession
def main(): Unit = {
val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

// 8- Create a dataframe and add the info of sample_libsvm_data.txt in LIBSVM format
val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")

// 9- Adding metadata to the label column 
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// 10- Identify categorical features, and index them
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// 11- Split the data into training and test sets
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// 12- Train a DecisionTree model
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
 
// 13- Convert indexed labels back to original labels
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// 14- Chain indexers and tree in a Pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
 
// 15- Train the model
val model = pipeline.fit(trainingData)
 
// 16- Make the predictions
val predictions = model.transform(testData)
 
// 17- Select an example rows to display
predictions.select("predictedLabel", "label", "features").show(5)
 
// 18- Select prediction and true label and then compute test error
   val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
   val accuracy = evaluator.evaluate(predictions)
   println(s"Test Error = ${(1.0 - accuracy)}")

// 19- Print the tree obtained from the model
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
   println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

