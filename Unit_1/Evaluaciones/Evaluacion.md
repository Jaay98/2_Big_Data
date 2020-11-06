<h1>Tecnológico Nacional de México</h1>
<h6> Instituto Tecnológico de Tijuana 

Subdirección Académica 
Departamento de Sistemas y Computación 

Semestre: Septiembre - Enero 2020-2021

Materia:
Datos Masivos

Profesor: 
Jose Christian Romero Hernandez

Alumno: 
17210526 Alvarez Yanez Jose Alonso


Fecha:
20/10/2020 </h6>


1. The first step is to import the library to create an entry point 
to sqlspark and create a session in spark
~~~
import org.apache.spark.sql.SparkSession 
val spark = SparkSession.builder().getOrCreate()
~~~

2. The next step is to load the Netflix CSV file, with the dataframe df and spark infer the data types
~~~
val df = spark.read.option("header","true").option("inferSchema","true").csv("Netflix_2011_2016.csv")
~~~

3. In the following instruction it is in charge of showing us the columns with our df followed by the word column
~~~
df.columns
~~~

4. It shows us the scheme that are the characteristics of the columns
~~~
df.printSchema() 
~~~

5. It takes care of printing the first 5 columns
~~~
df.head(5) 
~~~

6. The describes ()returns the basic metadata information of the df
~~~
df.describe().show() 
~~~
7. Create a new dataframe whit a new column named "HV_Ratio" who is the relation of the price of "high" columns
and "volume" of action negotiated by day
~~~
val df_hv = df.withColumn("HV Ratio",df("High")/df("Volume"))
df_hv.columns
~~~
8. What day had the highest peak in the “Close” column?
~~~
df.orderBy($"High".desc).show(1)
~~~
9. Who is the mean of the column "close" ?
 Answer: This column is the average between the columns "open", "high" and "low"
~~~
df.select(mean("Close")).show()
~~~

10. What is the maximum and minimum of the “Volume” column?
~~~
df.select(max("Volume")).show()
df.select(min("Volume")).show()
~~~

11. Sintaxis Scala/Spark
~~~
import spark.implicits._
~~~

a. How many days the column "close" was lowe tha $600?
~~~
df.filter($"Close"<600).count()
~~~

b. Who percentage of the time was the column "High" higher than $500?
~~~
(df.filter($"High" > 500).count() 
~~~

c. Find the Pearson correlation of the colums "high" and "volume"
~~~
df.select(corr("High","Volume")).show()
~~~

d. Who was the maximun in the column "high" by year?
~~~
val yeardataframe = df.withColumn("Year",year(df("Date")))
val yearmaxs = yeardataframe.select($"Year",$"High").groupBy("Year").max()
val res = yearmaxs.select($"Year",$"max(High)")
res.orderBy("Year").show()
~~~

e. Find the average of the column "close" for each month of the year

~~~
val monthdataframe = df.withColumn("Month",month(df("Date")))
val monthavgs = monthdataframe.select($"Month",$"Close").groupBy("Month").mean()
monthavgs.select($"Month",$"avg(Close)").orderBy("Month").show()
~~~