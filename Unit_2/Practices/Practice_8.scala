/*
Practice 8

Authors:
-Alvarez Yanez Jose Alonso 
-Quiroz Montes Yim Yetzhael

-Big Data

*/

// 1- Import libraries and package
package org.apache.spark.examples.ml
import org.apache.spark.ml.classification.LinearSVC
// 2- Import a Spark Session. 
import org.apache.spark.sql.SparkSession
// 3- Load the data stored in LIBSVM format as a DataFrame.
val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

val training = spark.read.format("libsvm").load("sample_libsvm_data.txt")
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
val lsvcModel = lsvc.fit(training)

 // 4- print coefficients intercept
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
