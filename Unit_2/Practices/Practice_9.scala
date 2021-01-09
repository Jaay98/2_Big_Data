/*
Practice 9

Authors:
-Alvarez Yanez Jose Alonso 
-Quiroz Montes Yim Yetzhael
-Big Data

*/
// 1- Import libraries and package
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
// 2- Define the function  main as parameter 
def main(): Unit = {
// 3- Define the object spark=sparksession - MultilayerPerceptronClassifierEvaluator
   val spark = SparkSession.builder.appName("MulticlassClassificationEvaluator").getOrCreate()
// 4- the data is loaded into the dataframe
val inputData = spark.read.format("libsvm")load("sample_multiclass_classification_data.txt")
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))
val classifier = new LogisticRegression()
.setMaxIter(10)
.setTol(1E-6)
.setFitIntercept(true)

val ovr = new OneVsRest().setClassifier(classifier)

val ovrModel = ovr.fit(train)
  
val predictions = ovrModel.transform(test)

val evaluator = new MulticlassClassificationEvaluator()
.setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)

println(s"Test Error = ${1 - accuracy}")}

println(s"Test Error = ${1 - accuracy}")


