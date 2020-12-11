/*
Practice 2

Authors:
-Alvarez Yanez Jose Alonso 
-Quiroz Montes Yim Yetzhael

-Big Data

*/

//Libraries

// 1- Import LogisticRegression
import org.apache.spark.ml.classification.LogisticRegression

// 2- Import SparkSession
import org.apache.spark.sql.SparkSession


import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// 3- Construct SparkSession
val spark = SparkSession.builder().getOrCreate()

// 4- Access to file advertising.csv
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")

// 5- Print Data
data.printSchema()

// 6- Print 1 row
data.head(1)

val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}


// 7- Create a new column Hour
val timedata = data.withColumn("Hour",hour(data("Timestamp")))

// 8- Rename Column Clicked on ad to Label

val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")


// 9- Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// 10- Create new object VectorAsssembler
val assembler = (new VectorAssembler()
.setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"))
.setOutputCol("features"))

// 11- Random split
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

// 12- Import Pipeline
import org.apache.spark.ml.Pipeline

// 13- Create LogisticRegression object
val lr = new LogisticRegression()

// 14- Construct Pipeline
val pipeline = new Pipeline().setStages(Array(assembler, lr))

val model = pipeline.fit(training)

val results = model.transform(test)

//15- Import MulticlassMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)
// 16- Print confussion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)
metrics.accuracy