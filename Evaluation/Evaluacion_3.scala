/*
Evaluacion Unidad_3


Authors:
-Alvarez Yanez Jose Alonso 
-Quiroz Montes Yim Yetzhael 

-Big Data

*/

   // 1. Importar una simple sesión Spark.
   // 1. Import SparkSession 

   import org.apache.spark.sql.SparkSession

   // 2. Utilizamos log4j para poder mostrar mensajes de error
   // 2.We use log4j to see the error message

   import org.apache.log4j._ 
   Logger.getLogger("org").setLevel(Level.ERROR)

   // 3. Cree una instancia de la sesión Spark
   // 3. Create a SparkSession

   val spark = SparkSession.builder().getOrCreate()

   // 4. Importar la librería de Kmeans para el algoritmo de agrupamiento.
   // 4. Import Kmeans library for the agroupment algoritm 
   import org.apache.spark.ml.clustering.KMeans

    //5. Carga el dataset de Wholesale Customers Data
    //5. Load the dataset

    val df = spark.read.option("header","true").option("inferSchema","true").csv("../Evaluation/Wholesale_customers_data.csv")
   //Show the data set / mostramos el dataset
    df.show()

   // 6. Seleccione las siguientes columnas: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen y llamar a este conjunto feature_data
   // 6. Select colums: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and name to this group "feature_data"

   val  feature_data  = df.select("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")
   //show feature_data // mostrar feature_data 
   feature_data.show()

    // 7. Importar Vector Assembler y Vector
    // 7. Import VectorAssembler and Vector

    import org.apache.spark.ml.feature.VectorAssembler

   // 8. Crea un nuevo objeto VectorAssembler para las columnas de caracteristicas como un conjunto de entrada, recordando que no hay etiquetas
   // 8. Make an object type VectorAssembler for the colums of the caracteristics as a group of entrance

   val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")

   // 9. Utilice el objeto assembler para transformar feature_data
   // 9. Use the assembler object to transform feature_data

    val  features = assembler.transform(feature_data)
   // show features / mostrar features
    features.show

    // 10. Crear un modelo Kmeans con K=3
    // 10. Make a Kmeans model whit K=3

    val kmeans = new KMeans().setK(3).setSeed(1L)
    val model = kmeans.fit(features)
    
   // 11. Evalúe  los grupos utilizando "Within Set Sum of Squared Errors (WSSSE)" e imprima los centroides.
   // 11. Evaluate the groups using "Within Set Sum of Squared Errors WSSSE" and print the centroids

    val WSSSE = model.computeCost(features)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

   // 12. Print the results
   // 12. Imprimir los resultados

    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)