/*

Practice 1

Practice 6


Authors:
-Alvarez Yanez Jose Aloso 
-Quiroz Montes Yim Yetzahel

-Big Data

*/

// 1- Import libraries 
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
// Here the file is loaded and analyzed, turning it into a dataframe "sample_libsvm_data.txt"
val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")

//We adjust the entire data set to include the labels in the index
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
//we add our val = new vector followed by maxCategories with> 4
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// We divide the data in 2 to be able to do the tes
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

//we get the max number of iterations
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")

//Convert indexed tags to original tags
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

//Label Converter
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
//we can visualize the rows
predictions.select("predictedLabel", "label", "features").show(20)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
//Print accuracy
println(s"Test Error = ${1.0 - accuracy}")

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
//Print Learned Classification Model
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")

