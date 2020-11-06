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
17210623 Quiroz Montes Yim Yetzhael

Fecha:
20/10/2020 </h6>


###Práctica 2.

1. Create a list named "list" with the elements "red", "white", "black"
~~~
val  lista =  scala.collection.mutable.MutableList ( "rojo" , "blanco" , "negro" )
~~~

2. Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"


~~~
lista +="verde"
lista +="amarillo"
lista +="azul"
lista +="naranja"
lista +="perla"
~~~

3. Bring the items from "list" "green", "yellow", "blue"

~~~
lista slice (3,6)
~~~

4. Creates an integer array ranging from 1-1000 in steps of 5 by 5

~~~
Array.range(0, 1000, 5)
~~~

5. What are the unique elements of the list List (1,3,3,4,6,7,3,7) use conversion to sets

~~~
val lista_2 = Set(1,3,3,4,6,7,3,7)
~~~


6. Create a mutable map named names that contains the following
~~~
"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
val map = Map(("Jose",20),("Luis",24),("Ana",23),("Susana",27))
~~~

6. Print all keys on the map

~~~
println (map)
~~~


7. Add the following value to the map ("Miguel", 23)

~~~
va mutmap = collection.mutable.Map(("Jose",20),("Luis",24),("Ana",23),("Susana",27))
mutmap += ("Miguel" -> 23)
~~~
