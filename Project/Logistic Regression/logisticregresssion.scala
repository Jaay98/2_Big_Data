//Authors: Alonso Alvarez, Yim Quiroz
//Instituto Tecnologico de Tijuana
//Materia: Datos Masivos
//Proyecto

// Import libraries
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression

//Minimize errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Create a spark session
val spark = SparkSession.builder().getOrCreate()

//Load  CSV 

val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("C:/Users/alons/OneDrive/Escritorio/Universidad/Datos Masivos/2_Big_Data/Project/Logistic Regression/bank-full.csv")

// Print schema
df.printSchema()

// Show Dataframe
df.show()


// Modify the column of strings to numeric data
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//Show the new column
newcolumn.show()

//Generate the features table
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Show the new column
fea.show()
//Change the column y to the label column
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)
//Logistic Regression algorithm
val logistic = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
//Fit of the model
val logisticModel = logistic.fit(feat)
//Impression of coefficients and interception
println(s"Coefficients: ${logisticModel.coefficients} Intercept: ${logisticModel.intercept}")
val logisticMult = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val logisticMultModel = logisticMult.fit(feat)
println(s"Multinomial coefficients: ${logisticMultModel.coefficientMatrix}")
println(s"Multinomial intercepts: ${logisticMultModel.interceptVector}")
