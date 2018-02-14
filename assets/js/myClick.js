function myClick(type) {
    var x = document.getElementsByClassName("post");

    for(var i =0, il = x.length;i<il;i++){
	    if (x[i].classList.contains(type)) {
	        x[i].style.display = "flex";
	    } else {
	        x[i].style.display = "none";
	    }
	}
}


