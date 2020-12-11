/*
Practice 3

Authors:
-Alvarez Yanez Jose Aloso 
-Quiroz Montes Yim Yetzahel

-Big Data

*/

//Import libraries
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
//Create "data" and assign a secuenci of vectors 
    val data = Seq(
      Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))), // (0,3,1,-2)
      Vectors.dense(4.0, 5.0, 0.0, 3.0),
      Vectors.dense(6.0, 7.0, 0.0, 8.0),
      Vectors.sparse(4, Seq((0, 9.0), (3, 1.0))) // (0,3,9,1)
    )

//Create a dataframe an assign the value of a tuple called "Tuple1"
//The dataframe contains a colum called features

val df = data.map(Tuple1.apply).toDF("features")

//To value type row called coefficiente1 of a matrix, assings value of the Pearson Correlation used in the dataframe aplied to the colum features
val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
println(s"Pearson correlation matrix:\n $coeff1")

//Add a value called "" 
//A un valor tipo Fila llamado coefficiente2 de una matriz se le asigna el valor de la correlacion de spearman aplicada en el dataframe
//Aplicada a su columna features
val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
println(s"Spearman correlation matrix:\n $coeff2") //Se imprime
    

    spark.stop()
  }
}