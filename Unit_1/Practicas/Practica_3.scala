//Practice 3
//Author Alvarez Yanez Jose Alonso
//17210526

// Practice 3, analyse the following code with your own words

def listEvens(list:List[Int]): String ={
    for(n <- list){
        if(n%2==0){
            println(s"$n is even")
        }else{
            println(s"$n is odd")
        }
    }
    return "Done"
}

// We define a method named listEvens, this method receives a list of integers and return a string value, this method determines if a 
// number es even or odd

val l = List(1,2,3,4,5,6,7,8) //here create a list of numbers
val l2 = List(4,3,22,55,7,8) //  //here create a list of numbers
listEvens(l)
listEvens(l2)

//3 7 afortunado

def afortunado(list:List[Int]): Int={
    var res=0
    for(n <- list){
        if(n==7){
            res = res + 14
        }else{
            res = res + n
        }
    }
    return res
}


val af= List(1,7,7)
println(afortunado(af))

//This method named "afortunado" receives a list of integers and make a summation of the elements but if the list includes de number "7"
// in place of that number adds the number "14"

def balance(list:List[Int]): Boolean={
    var primera = 0
    var segunda = 0

    segunda = list.sum

    for(i <- Range(0,list.length)){
        primera = primera + list(i)
        segunda = segunda - list(i)

        if(primera == segunda){
            return true
        }
    }
    return false 
}

val bl = List(3,2,1)
val bl2 = List(2,3,3,2)
val bl3 = List(10,30,90)

balance(bl)
balance(bl2)
balance(bl3)

//This method named "balance" recives a list of integers and return a Boolean value, returns a "True" if de sum of the first until  
//middle element is the same as the sum of last until the middle element

def palindromo(palabra:String):Boolean ={
    return (palabra == palabra.reverse)
}

val palabra = "OSO"
val palabra2 = "ANNA"
val palabra3 = "JUAN"

println(palindromo(palabra))
println(palindromo(palabra2))
println(palindromo(palabra3))

//This method named "palindromo" receive a string and return a boolean value,  is going to return  a "True" if the word writes 
//equal to the right or the reverse