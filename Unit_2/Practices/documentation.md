##<p align="center">INSTITUTO TECNOLOGICO DE TIJUANA </p>

##<p align="center"> Datos Masivos </p>
##<p align="center"> Unidad 2 </p>





<p align="center">
 Quiroz Montes Yim Yetzhael
</p>
<p align="center">
 Alvarez Yanez Jose Alonso
</p>



---
#### Practice 1

1- Import LinearRegression
~~~
import org.apache.spark.ml.regression.LinearRegression
~~~
2- Use the following code to configure errors   	 

~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~

3- Start a simple Spark Session
~~~
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
~~~
4- Use Spark for the Clean-Ecommerce csv file 
~~~
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")
~~~
5- Print the schema on the DataFrame
~~~
data.printSchema
~~~
6- Print an example row from the DataFrame 
~~~
data.head(1)
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(0, colnames.length)){
   println(colnames(ind))
   println(firstrow(ind))
   println("\n")
}
~~~
7- Transform the data frame so that it takes the form of ("label", "features")
~~~
Import VectorAssembler and Vectors:
 
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~	
8- Rename the Yearly Amount Spent column as "label"
~~~
val df = data.select(data("Yearly Amount Spent").as("label"), $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership", $"Yearly Amount Spent")
~~~
9- The VectorAssembler Object 
~~~
val new_assembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership", "Yearly Amount Spent")).setOutputCol("features")
~~~
10- Use the assembler to transform our DataFrame to two columns: label and features 	
~~~
val output = new_assembler.transform(df).select($"label",$"features")
~~~
11- Create an object for line regression model 
~~~
val lr = new LinearRegression()
~~~
12- Fit the model for the data and call this model lrModel
~~~
val lrModel = lr.fit(output)
~~~
13- Print the coefficients and intercept for the linear regression
~~~
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
~~~
14- Summarize the model on the training set and print the output of some metrics
~~~
val trainingSummary = lrModel.summary
~~~  
15- Show the residuals values, the RMSE, the MSE, and also the R^2
~~~
trainingSummary.residuals.show()
val RMSE = trainingSummary.rootMeanSquaredError
val MSE = scala.math.pow(RMSE, 2.0)
val R2 = trainingSummary.r2 
~~~
---
#### Practice 2

1- Import LogisticRegression
~~~
import org.apache.spark.ml.classification.LogisticRegression
~~~
2- Import SparkSession
~~~
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~
3- Construct SparkSession
~~~
val spark = SparkSession.builder().getOrCreate()
~~~
4- Access to file advertising.csv
~~~
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
~~~
5- Print Data
~~~
data.printSchema()
~~~
6- Print 1 row
~~~
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
~~~

7- Create a new column Hour
~~~
val timedata = data.withColumn("Hour",hour(data("Timestamp")))
~~~
8- Rename Column Clicked on ad to Label
~~~
val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")
~~~

9- Import VectorAssembler and Vectors
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~
10- Create new object VectorAsssembler
~~~
val assembler = (new VectorAssembler()
.setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"))
.setOutputCol("features"))
~~~
11- Random split
~~~
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)
~~~
12- Import Pipeline
~~~
import org.apache.spark.ml.Pipeline
~~~

13- Create LogisticRegression object
~~~
val lr = new LogisticRegression()
~~~
14- Construct Pipeline
~~~
val pipeline = new Pipeline().setStages(Array(assembler, lr))

val model = pipeline.fit(training)

val results = model.transform(test)
~~~
15- Import MulticlassMetrics
~~~
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)
~~~
16- Print confussion matrix
~~~
println("Confusion matrix:")
println(metrics.confusionMatrix)
metrics.accuracy
~~~
---
#### Practice 3

1- Import libraries
~~~
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession


object CorrelationExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("CorrelationExample")
      .getOrCreate()
    import spark.implicits._
~~~
2- Create "data" and assign a secuenci of vectors 
~~~   
    val data = Seq(
      Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))), 
      Vectors.dense(4.0, 5.0, 0.0, 3.0),
      Vectors.dense(6.0, 7.0, 0.0, 8.0),
      Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
    )
~~~
3- Create a dataframe an assign the value of a tuple called "Tuple1"
The dataframe contains a colum called features
~~~
val df = data.map(Tuple1.apply).toDF("features")
~~~
4- To value type row called coefficiente1 of a matrix, assings value of the Pearson Correlation used in the dataframe aplied to the colum features
~~~
val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
println(s"Pearson correlation matrix:\n $coeff1")
~~~
5- Add a value type row named coefficiente2 of a matrix and assing the value of the Spearman Correlation, using the dataframe and use it 
to the  features colum
~~~
val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
println(s"Spearman correlation matrix:\n $coeff2") 
    

    spark.stop()
  }
}
~~~

---
### Practice 4
1- Import Pipeline
~~~
import org.apache.spark.ml.Pipeline
~~~
2- Import DecisionTreeClassificationModel
~~~
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
~~~
3- Import DecisionTreeClassifier
~~~
import org.apache.spark.ml.classification.DecisionTreeClassifier
~~~
4- Import MulticlassClassificationEvaluator
~~~
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~
5- Import IndexToString, StringIndexer, VectorIndexer
~~~
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
 ~~~
6- Import a SparkSession
~~~
import org.apache.spark.sql.SparkSession
~~~
7- Create a SparkSession
~~~
def main(): Unit = {
val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()
~~~
8- Create a dataframe and add the info of sample_libsvm_data.txt in LIBSVM format
~~~
val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
~~~
9- Adding metadata to the label column 
~~~
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
~~~
10- Identify categorical features, and index them
~~~
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
~~~
11- Split the data into training and test sets
~~~
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
~~~
12- Train a DecisionTree model
~~~
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
~~~ 
13- Convert indexed labels back to original labels
~~~
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
 ~~~
14- Chain indexers and tree in a Pipeline
~~~
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
 ~~~
15- Train the model
~~~
val model = pipeline.fit(trainingData)
 ~~~
16- Make the predictions
~~~
val predictions = model.transform(testData)
 ~~~
17- Select an example rows to display
~~~
predictions.select("predictedLabel", "label", "features").show(5)
 ~~~
18- Select prediction and true label and then compute test error
~~~
   val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
   val accuracy = evaluator.evaluate(predictions)
   println(s"Test Error = ${(1.0 - accuracy)}")
~~~
19- Print the tree obtained from the model
~~~
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
~~~
---
### Practice 6

1- Import libraries 
~~~
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
~~~
2- Here the file is loaded and analyzed, turning it into a dataframe "sample_libsvm_data.txt"
~~~
val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
~~~
3- We adjust the entire data set to include the labels in the index
~~~
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
~~~
4- we add our val = new vector followed by maxCategories with> 4
~~~
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
~~~
5- We divide the data in 2 to be able to do the tes
~~~
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
~~~
6- we get the max number of iterations
~~~
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")
~~~
7- Convert indexed tags to original tags
~~~
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
~~~
8- Label Converter
~~~
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
~~~
9- Runs the indexers
~~~
val model = pipeline.fit(trainingData)
~~~
10- Make predictions
~~~
val predictions = model.transform(testData)
~~~
11- Select example rows to display
We can visualize the rows
~~~
predictions.select("predictedLabel", "label", "features").show(20)
~~~
12- Select (prediction, true label) and compute test error
~~~
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
~~~
13- Print accuracy
~~~
println(s"Test Error = ${1.0 - accuracy}")

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
~~~
14- Print Learned Classification Model
~~~
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
~~~
---
#### Practica 7

1- Import MultilayerPerceptronClassifier y MulticlassClassificationEvaluator
~~~
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~
2- Import spark
~~~
import org.apache.spark.sql.SparkSession
~~~

3- Create the object  MultilayerPerceptronClassifier
~~~
object MultilayerPerceptronClassifierExample {
~~~
4- Define the function  main as parameter  
~~~
  def main(): Unit = {
~~~   
5- Define the object spark=sparksession 
~~~  
    val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()
~~~
6- The data is loaded into the dataframe
~~~   
    val data = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")
~~~
7- Data is divided into training and testing
~~~  
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)
~~~
8- The layers of the neural network are specified: in an array of size [4,5,4,3]
~~~   
    val layers = Array[Int](4, 5, 4, 3)
~~~
   
9- Define the parameters of training
   ~~~
    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
~~~
10- The model is trained
   ~~~
    val model = trainer.fit(train)
~~~
11- Calculate the precision of the dates of test
   ~~~
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
~~~
12- Print pattern accuracy
   ~~~
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")


    spark.stop()
  }
}
~~~

---

### Practice 8

1- Import libraries and package
~~~
package org.apache.spark.examples.ml
import org.apache.spark.ml.classification.LinearSVC
~~~
2- Import a Spark Session. 
~~~
import org.apache.spark.sql.SparkSession
~~~
3- Load the data stored in LIBSVM format as a DataFrame.
~~~
val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

val training = spark.read.format("libsvm").load("sample_libsvm_data.txt")
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
val lsvcModel = lsvc.fit(training)
~~~
4- Print coefficients intercept
~~~
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
~~~
---


### Practice 9

1- Import libraries and package
~~~
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
~~~
2- Define the function  main as parameter 
~~~
def main(): Unit = {
~~~
3- Define the object spark=sparksession - MultilayerPerceptronClassifierEvaluator
~~~
   val spark = SparkSession.builder.appName("MulticlassClassificationEvaluator").getOrCreate()
~~~
4- The data is loaded into the dataframe
~~~
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
~~~

---
### Practice 10

1- Import package spark.ml.cookbook.chapter6
~~~
package spark.ml.cookbook.chapter6
~~~
2- Import libraries spark
~~~
import org.apache.spark.mllib.linalg.{Vector, Vectors} 
import org.apache.spark.mllib.regression.LabeledPoint 
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics, MultilabelMetrics, binary}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.log4j.Logger
import org.apache.log4j.Level
~~~
3-Import Data "iris-data-prepared.txt"
~~~
val data = sc.textFile("iris-data-prepared.txt")
~~~
4-Transform in the data set of these columns into vectors
~~~
val NaiveBayesDataSet = data.map { line => val 
columns = line.split(',')
LabeledPoint(columns(4).toDouble,
Vectors.dense(
columns(0).toDouble,
columns(1).toDouble,
columns(2).toDouble,
columns(3).toDouble))
}
~~~
5-We get the total of data vector
~~~
println(" Total number of data vectors =", 
NaiveBayesDataSet.count())

val distinctNaiveBayesData = NaiveBayesDataSet.distinct() 
println("Distinct number of data vectors = ", 
distinctNaiveBayesData.count())
~~~
6-Print the collect  taking 10
~~~
distinctNaiveBayesData.collect().take(10).foreach(println(_))
~~~
7-divide data random for create dataset of training  and one to release test of (70% and 
30%)
~~~
val allDistinctData = distinctNaiveBayesData.randomSplit(Array(.80,.20),10L)
val trainingDataSet = allDistinctData(0)
val testingDataSet = allDistinctData(1)
~~~
8-Print the number of training and test of data
~~~
println("number of training data =",trainingDataSet.count())
println("number of test data =",testingDataSet.count())
~~~
9-Create the model NaiveBayes
~~~
val myNaiveBayesModel = NaiveBayes.train(trainingDataSet)
~~~
10-Test dataset is read by each one of its values and it will try to predict and compare them.
~~~
val predictedClassification = testingDataSet.map( x => 
 (myNaiveBayesModel.predict(x.features), x.label))
~~~
11-Create val Metrics
~~~
val metrics = new MulticlassMetrics(predictedClassification)
~~~
12-Create confusion matrix and print
 ~~~
 val confusionMatrix = metrics.confusionMatrix 
 println("Confusion Matrix= n",confusionMatrix)
~~~
13-Create and print precision metrics
 ~~~
 val myModelStat=Seq(metrics.precision)
 myModelStat.foreach(println(_))
 ~~~