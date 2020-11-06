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
//Make a code who runs a Fibbonacci Series

//Algorithm 1, Recursive version

def fibbo1(numero:Int) : Int ={

    if(numero<2)
        {
            return numero
        }
        else 
        {
            return fibbo1 (numero-1) + fibbo1 (numero-2)
        }
}
//Algorithm 2 explicit formula

def fibbo2(numero:Double) : Double ={

    if(numero<2) {
        return numero
    }
    else {
        
    }

    return ((1/math.sqrt(5))* math.pow(((1+math.sqrt(5))/2),numero)+(1/2))

}

//Algorithm 3 Iterative with two variables

def fibbo3( numero : Int ) : Int = {
  var a = 0
  var b = 1
  var c = 0   
  var i = 0
 
  while( i < numero ) {
    val c = a + b
    a = b
    b = c
    i = i + 1
  } 
  return a
}


//Algorithm 4 Iterative version 2 variables

def fibbo4( numero : Int ) : Int = {
  var a = 0
  var b = 1
  var i = 0

 
  while( i < numero ) {
     b = b + a 
     a = b - a
    
    i=i+1
  } 
  return a
}


//Algorithm 5  Iterative version vector


def fibbo5(numero:Int) : Int = {
	if(numero < 2){
		numero
	}
	else{
		var z = new Array[Int](numero+1)
		z(0) = 0
		z(1) = 1
		for(x <- 2 to (numero))
		{
			z(x) = (z(x-1) + z(x-2))
		}
		z(numero)
	}
}
~~~