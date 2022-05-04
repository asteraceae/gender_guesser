//const python = require("python-bridge");
//let py = python();

var input = document.getElementById("txtbox");

let guesser = {
    predict: function(text){
        $.ajax({
            type: "POST",
            url: "./test.py",
            data: {param: text},
            success: this.displayResult
        });
    },
    displayResult: function(response){
        console.log(response);
        document.querySelector(".result").innerText = response;
    }
};

//document.querySelector(".button-reset").addEventListener("click", );
document.querySelector(".button-submit").addEventListener("click", function(){
    var text = input.value
    guesser.predict(text);
});
document.querySelector(".button-reset").addEventListener("click", function(){
    input.value = "";
});