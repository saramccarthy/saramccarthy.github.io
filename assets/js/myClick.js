function myClick(c) {
    var x = document.getElementsByClassName(c);
    for(var i =0, il = selects.length;i<il;i++){

	    if (x[i].style.display === "none") {
	        x[i].style.display = "flex";
	    } else {
	        x[i].style.display = "none";
	    }
	}
}


