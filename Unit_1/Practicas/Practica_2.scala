//Practice 2 
//Author Alvarez Yanez Jose Alonso,Quiroz Montes Yim Yetzhael
//17210526,17210623

// 1. Crea una lista llamada "lista" con los elementos "rojo", "blanco", "negro"

val  lista =  scala.collection.mutable.MutableList ( "rojo" , "blanco" , "negro" )


// 2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla"

lista +="verde"
lista +="amarillo"
lista +="azul"
lista +="naranja"
lista +="perla"

// 3. Traer los elementos de "lista" "verde", "amarillo", "azul"

lista slice (3,6)

// 4. Crea un arreglo de número en rango del 1-1000 en pasos de 5 en 5

Array.range(0, 1000, 5)

// 5. Cuales son los elementos únicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversión a conjuntos

val lista_2 = Set(1,3,3,4,6,7,3,7)

 // 6. Crea una mapa mutable llamado nombres que contenga los siguiente 
// "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"

val map = Map(("Jose",20),("Luis",24),("Ana",23),("Susana",27))

//6. Imprime todas la llaves del mapa

println (map)

//7. Agrega el siguiente valor al mapa("Miguel", 23)

val mutmap = collection.mutable.Map(("Jose",20),("Luis",24),("Ana",23),("Susana",27))
mutmap += ("Miguel" -> 23)