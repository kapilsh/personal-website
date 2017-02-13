var image = new Array ();
image[0] = "../assets/images/bp1.png";
image[1] = "../assets/images/bp2.png";
image[2] = "../assets/images/bp3.png";
image[3] = "../assets/images/bp4.png";
image[4] = "../assets/images/bp5.png";
var size = image.length

var i = 0;
setInterval(fadeDivs, 3000);

function fadeDivs() {
    $('.random-img img').fadeOut(100, function(){
        var x = Math.floor(size*Math.random())
        $(this).attr('src', image[x]).fadeIn(100);
    })
    i++;
}
