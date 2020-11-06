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


~~~
//1.Calcular el radio  de un circulo


val area=98
val radio= area/(2*3.1416)

//2.Definir si un numero es primo 

def numero_primo(n :Int) : Boolean = 
{
    if (n <= 1)
    false
    else (n==2)
    true
   !(2 to (n-1)).exists(x=> n % x==0)
}

//3.Dada la variable bird = "tweet", utiliza interpolación de string para imprimir "Estoy escribiendo un tweet"
var bird= "tweet"
println ("Estoy escribiendo un" + bird)

// 4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la // secuencia "Luke"
var mensaje = "Hola Luke soy  tu padre!"
mensaje.slice(5,9)

//5.¿Cual es la diferencia entre value y una variable en scala?
//Un valor es inmutable, en cambio la variable si puede cambiar de valor

//6. Dada la tupla (2,4,5,1,2,3,3.1416,23) regresa el número 3.1416 

val tupla = (2,4,5,1,2,3,3.1416,23)
println (tupla._7)
~~~
