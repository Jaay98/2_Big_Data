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
23/10/2020 </h6>
~~~
//Practice 5 dataframe functions for spark

//First we import the library

import spark.implicits._ 

val simpleData = Seq (
    ("Montserrat","Sales","BC",29483,34,10000),
    ("Aaron","Sales","CDMX",45000,56,20000),
    ("Sebastian","Sales","JAL",90000,30,23000),
    ("Alfredo","Finance","COL",90000,24,23000),
    ("Ramon","Finance","BCS",99000,40,24000),
    ("Judith","Finance","SON",83000,36,19000),
    ("Ivanue","Finance","BC",79000,53,15000),
    ("Yim","Marketing","SD",80000,25,18000),
    ("Alonso","Marketing","NY",91000,50,21000)
)

val df= simpleData.toDF("employee","department","state","salary","age","bonus")

// 1. Show the info in the DataFrame

df.show()

// 2. Order by department and state
df.orderBy("department","state").show(false)

// 3. Return the first element 

df.select(first("employee")).show() 

// 4. Return the number of elements 

df.select(approx_count_distinct("state")).show()

// 5. Return the max value
df.select(max("salary")).show() 

// 6. Return the avg
df.select(avg("salary")).show() 

// 7. Return the last element

df.select(last("employee")).show()

// 8. Return the pearson correlation coefficient

df.select(corr("salary","bonus")).show()

// 9. Filter a range

df.filter("age > 50").show

// 10. Return  an  specific number of lines in a list

df.takeAsList(6)

// 11. Return the first row

df.first()

// 12. Return the summary of the bonus

df.select(sum("bonus")).show()

// 13. Return the sum of the different values

df.select(sumDistinct("salary")).show()

// 14. Show the existent columns

df.printSchema()

// 15. Show the minimun value 

df.select(min("salary")).show()
~~~