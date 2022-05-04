var input = document.getElementById("txtbox");

let guesser = {
    predict: function(text){
        //var json = {"text": text}
        $.ajax({
            type: "POST",
            url: "/model",
            data: text,
            success: this.displayResult
        });
    },
    displayResult: function(response){
        document.querySelector(".result").innerText = "The author is likely to be: " + response;
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