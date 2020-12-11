/*
Practice 1

Authors:
-Alvarez Yanez Jose Alonso 
-Quiroz Montes Yim Yetzhael

-Big Data

*/

// 1- Import LinearRegression
import org.apache.spark.ml.regression.LinearRegression

// 2- Use the following code to configure errors   	 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// 3- Start a simple Spark Session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

// 4- Use Spark for the Clean-Ecommerce csv file 
 val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")

// 5- Print the schema on the DataFrame
data.printSchema

// 6- Print an example row from the DataFrame 
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

// 7- Transform the data frame so that it takes the form of ("label", "features")
Import VectorAssembler and Vectors:
 
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
	
// 8- Rename the Yearly Amount Spent column as "label"
val df = data.select(data("Yearly Amount Spent").as("label"), $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership", $"Yearly Amount Spent")

// 9- The VectorAssembler Object 
val new_assembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership", "Yearly Amount Spent")).setOutputCol("features")

// 10- Use the assembler to transform our DataFrame to two columns: label and features 	
val output = new_assembler.transform(df).select($"label",$"features")

// 11- Create an object for line regression model 
val lr = new LinearRegression()

// 12- Fit the model for the data and call this model lrModel
val lrModel = lr.fit(output)

// 13- Print the coefficients and intercept for the linear regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// 14- Summarize the model on the training set and print the output of some metrics
val trainingSummary = lrModel.summary
  
// 15- Show the residuals values, the RMSE, the MSE, and also the R^2
trainingSummary.residuals.show()
val RMSE = trainingSummary.rootMeanSquaredError
val MSE = scala.math.pow(RMSE, 2.0)
val R2 = trainingSummary.r2 