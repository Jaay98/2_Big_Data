/*
Practice 10

Authors:
-Alvarez Yanez Jose Alonso 
-Quiroz Montes Yim Yetzhael

-Big Data

*/
// 1- Import package spark.ml.cookbook.chapter6
package spark.ml.cookbook.chapter6
// 2- Import libraries spark
import org.apache.spark.mllib.linalg.{Vector, Vectors} 
import org.apache.spark.mllib.regression.LabeledPoint 
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics, MultilabelMetrics, binary}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.log4j.Logger
import org.apache.log4j.Level

// 3- Import Data "iris-data-prepared.txt"
val data = sc.textFile("iris-data-prepared.txt")

// 4- Transform in the data set of these columns into vectors
val NaiveBayesDataSet = data.map { line => val 
columns = line.split(',')
LabeledPoint(columns(4).toDouble,
Vectors.dense(
columns(0).toDouble,
columns(1).toDouble,
columns(2).toDouble,
columns(3).toDouble))
}

// 5- We get the total of data vector
println(" Total number of data vectors =", 
NaiveBayesDataSet.count())

val distinctNaiveBayesData = NaiveBayesDataSet.distinct() 
println("Distinct number of data vectors = ", 
distinctNaiveBayesData.count())

// 6-Print the collect  taking 10
distinctNaiveBayesData.collect().take(10).foreach(println(_))

// 7- Divide data random for create dataset of training  and one to release test of (70% and 30%)
val allDistinctData = distinctNaiveBayesData.randomSplit(Array(.80,.20),10L)
val trainingDataSet = allDistinctData(0)
val testingDataSet = allDistinctData(1)

// 8- Print the number of training and test of data
println("number of training data =",trainingDataSet.count())
println("number of test data =",testingDataSet.count())

// 9- Create the model NaiveBayes
val myNaiveBayesModel = NaiveBayes.train(trainingDataSet)

// 10- test dataset is read by each one of its values and it will try to predict and compare them.
val predictedClassification = testingDataSet.map( x => 
 (myNaiveBayesModel.predict(x.features), x.label))

// 11- Create val Metrics
val metrics = new MulticlassMetrics(predictedClassification)

// 12- Create confusion matrix and print
 val confusionMatrix = metrics.confusionMatrix 
 println("Confusion Matrix= n",confusionMatrix)

// 13- Create and print precision metrics
 val myModelStat=Seq(metrics.precision)
 myModelStat.foreach(println(_))