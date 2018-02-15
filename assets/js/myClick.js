function myClick(i) {
    var x = document.getElementsByClassName("post");
    var type;
	if (i == 1){
		type = "sticky"
	} else if ( i == 2){
		type = "project"
	} else if (i == 3){
		type = "education"
	}
    for(var i =0, il = x.length;i<il;i++){
	    if (x[i].classList.contains(type)) {
	    	console.log(x[i].classList)
	        x[i].style.display = "flex";
	    } else {
	        x[i].style.display = "none";
	    }
	}
}

function changeNav(element){
	var x = document.getElementsByClassName("active");
	for(var i =0, il = x.length;i<il;i++){
		x[i].classList.remove("active");

	}
	element.classList.add("active");
}