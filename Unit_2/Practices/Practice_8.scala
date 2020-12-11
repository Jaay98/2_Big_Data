/*
Practice 8

Authors:
-Alvarez Yanez Jose Aloso 
-Quiroz Montes Yim Yetzahel

-Big Data

*/

//1. Import libraries and package
package org.apache.spark.examples.ml
import org.apache.spark.ml.classification.LinearSVC
// Import a Spark Session. 
import org.apache.spark.sql.SparkSession
//Load the data stored in LIBSVM format as a DataFrame.
val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

val training = spark.read.format("libsvm").load("/usr/local/spark-2.3.4-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
 val lsvcModel = lsvc.fit(training)
 //print coefficients intercept
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
