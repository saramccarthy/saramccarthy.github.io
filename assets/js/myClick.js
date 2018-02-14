function myClick(c) {
    var x = document.getElementsByClassName(c);
    for(var i =0, il = selects.length;i<il;i++){
	movement.animate(
	    20,
	    function(){
	    	x[i].style.background-color: rgba(255,255,255,0.1);

	      	if (x[i].style.display === "none") {
		        x[i].style.display = "flex";
		    } else {
		        x[i].style.display = "none";
		    }
	    });

	}
}

