function myClick() {
    var x = document.getElementsByClassName("post");
    for(var i =0, il = x.length;i<il;i++){
    	x[i].style.color = "blue";
	    if (x[i].style.display === "none") {
	        x[i].style.display = "flex";
	    } else {
	        x[i].style.display = "none";
	    }
	}
}


