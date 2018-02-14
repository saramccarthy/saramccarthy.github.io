function myClick(i) {
    var x = document.getElementsByClassName("post");
    var type;
	if (i == 1){
		type = "sticky"
	}
    for(var i =0, il = x.length;i<il;i++){
	    if (x[i].classList.contains(type)) {
	        x[i].style.display = "flex";
	    } else {
	        x[i].style.display = "none";
	    }
	}
}
