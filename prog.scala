/* Sample Code to print Statement */

object ExampleString {
    def main(args: Array[String]) {

        //declare and assign string variable "text"
        val text : String = "You are reading SCALA programming language.";
        //print the value of string variable "text"
        println("Value of text is: " + text);
    }
}


/**Scala program to find a number is positive, negative or positive.*/

object ExCheckNumber {
    def main(args: Array[String]) {

        /**declare a variable*/
        var number= (-100);

        if(number==0){
            println("number is zero");
        }
        else if(number>0){
            println("number is positive");
        }
        else{
            println("number is negative");
        }
    }
}


/*Scala program to print your name*/

object ExPrintName {
    def main(args: Array[String]) {
        println("My name is Mike!")
    }
}


/**Scala Program to find largest number among two numbers.*/

object ExFindLargest {
    def main(args: Array[String]) {
        var number1=20;
        var number2=30;
        var x = 10;
        if( number1>number2){
            println("Largest number is:" + number1);
        }
        else{
            println("Largest number is:" + number2);
        }
    }
}