/*
Practice 7

Authors:
-Alvarez Yanez Jose Alonso 
-Quiroz Montes Yim Yetzhael
-Big Data

*/

// 1- Import MultilayerPerceptronClassifier y MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// 2- Import spark
import org.apache.spark.sql.SparkSession


// An example for Multilayer Perceptron Classification.


 // 3- Create the object  MultilayerPerceptronClassifier
object MultilayerPerceptronClassifierExample {

// 4- Define the function  main as parameter  
  def main(): Unit = {
    // 5- Define the object spark=sparksession 
  
    val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()

    // 6- The data is loaded into the dataframe
    val data = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")

    // 7- Data is divided into training and testing
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // 8- The layers of the neural network are specified: in an array of size [4,5,4,3]
    val layers = Array[Int](4, 5, 4, 3)

    // 9- Define the parameters of training
    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

    // 10- The model is trained
    val model = trainer.fit(train)

    // 11- Calculate the precision of the dates of test
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    // 12- Print pattern accuracy
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")


    spark.stop()
  }
}
