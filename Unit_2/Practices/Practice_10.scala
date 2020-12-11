/*
Practice 10

Authors:
-Alvarez Yanez Jose Alonso 
-Quiroz Montes Yim Yetzhael

-Big Data

*/
package spark.ml.cookbook.chapter6
//Import libraries spark
import org.apache.spark.mllib.linalg.{Vector, Vectors} 
import org.apache.spark.mllib.regression.LabeledPoint 
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics, MultilabelMetrics, binary}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.log4j.Logger
import org.apache.log4j.Level

//Import Data "iris-data-prepared.txt"
val data = sc.textFile("iris-data-prepared.txt")

//Transform in the data set of these columns into vectors
val NaiveBayesDataSet = data.map { line => val 
columns = line.split(',')
LabeledPoint(columns(4).toDouble,
Vectors.dense(
columns(0).toDouble,
columns(1).toDouble,
columns(2).toDouble,
columns(3).toDouble))
}

//We get the total of data vector
println(" Total number of data vectors =", 
NaiveBayesDataSet.count())

val distinctNaiveBayesData = NaiveBayesDataSet.distinct() 
println("Distinct number of data vectors = ", 
distinctNaiveBayesData.count())

// Print the collect  taking 10
distinctNaiveBayesData.collect().take(10).foreach(println(_))

//divide data random for create dataset of training  and one to release test of (70% and 30%)
val allDistinctData =
distinctNaiveBayesData.randomSplit(Array(.80,.20),10L)
val trainingDataSet = allDistinctData(0)
val testingDataSet = allDistinctData(1)

//Print the number of training and test of data
println("number of training data =",trainingDataSet.count())
println("number of test data =",testingDataSet.count())

//Create the model NaiveBayes
val myNaiveBayesModel = NaiveBayes.train(trainingDataSet)

//  test dataset is read by each one of its values and it will try to predict and compare them.
val predictedClassification = testingDataSet.map( x => 
 (myNaiveBayesModel.predict(x.features), x.label))

// val Metrics
val metrics = new MulticlassMetrics(predictedClassification)

//Create confusion matrix and print
 val confusionMatrix = metrics.confusionMatrix 
 println("Confusion Matrix= n",confusionMatrix)

//Create and print precision metrics
 val myModelStat=Seq(metrics.precision)
 myModelStat.foreach(println(_))