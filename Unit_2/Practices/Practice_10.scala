/*
Practice 10

Authors:
-Alvarez Yanez Jose Aloso 
-Quiroz Montes Yim Yetzahel

-Big Data

*/
package spark.ml.cookbook.chapter6
//Libraries
//Import spark linalg
//Import Regresion
//Import Classification
//Import Naive Bayes
//Import sql
import org.apache.spark.mllib.linalg.{Vector, Vectors} 
import org.apache.spark.mllib.regression.LabeledPoint 
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics, MultilabelMetrics, binary}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.log4j.Logger
import org.apache.log4j.Level

//Import Data
val data = sc.textFile("iris-data-prepared.txt")

// Trasform in to dataset in we gonna do take the column species with label of reference 
//sepal_length,sepal_width,petal_length,petal_width, transforms this columns in vectores
val NaiveBayesDataSet = data.map { line => val 
columns = line.split(',')
LabeledPoint(columns(4).toDouble,
Vectors.dense(
columns(0).toDouble,
columns(1).toDouble,
columns(2).toDouble,
columns(3).toDouble))
}

//We clean our dataset to eliminate duplicates 
println(" Total number of data vectors =", 
NaiveBayesDataSet.count())

val distinctNaiveBayesData = NaiveBayesDataSet.distinct() 
println("Distinct number of data vectors = ", 
distinctNaiveBayesData.count())

// print data to see who are wa gonna print them
distinctNaiveBayesData.collect().take(10).foreach(println(_))

//divide data random for create dataset of training  and one to release test of (70% and 30%)
val allDistinctData =
distinctNaiveBayesData.randomSplit(Array(.80,.20),10L)
val trainingDataSet = allDistinctData(0)
val testingDataSet = allDistinctData(1)

println("number of training data =",trainingDataSet.count())
println("number of test data =",testingDataSet.count())

//Create model with the functions of naive bayes what oferr packaje of scala and will train to be our data set of training 
val myNaiveBayesModel = NaiveBayes.train(trainingDataSet)

//  test dataset is read by each one of its values and it will try to predict and compare them.
val predictedClassification = testingDataSet.map( x => 
 (myNaiveBayesModel.predict(x.features), x.label))

//Metrics
val metrics = new MulticlassMetrics(predictedClassification)

//Confusion matrix
 val confusionMatrix = metrics.confusionMatrix 
 println("Confusion Matrix= n",confusionMatrix)

//Precision Metrics
 val myModelStat=Seq(metrics.precision)
 myModelStat.foreach(println(_))