// Add your javascript here
// Don't forget to add it into respective layouts where this js file is needed

var startLat="";
var startLon="";

var endLat="";
var endLon="";

function selectStartLatitude(e)
{
    startLat = e.target.innerText;
    $("#lat-start").html(startLat).attr('style', "color: black; background-color: white !important;" )
    displaySatelliteImage();
}
function selectEndLatitude(e)
{
    endLat = e.target.innerText;
    $("#lat-end").html(endLat).attr('style', "color: black; background-color: white !important;" );
    displaySatelliteImage();
}


function selectStartLongitude(e)
{
    startLon = e.target.innerText;
    $("#lon-start").html(startLon).attr('style', "color: black; background-color: white !important;" );
    displaySatelliteImage();
}
function selectEndLongitude(e)
{
    endLon = e.target.innerText;
    $("#lon-end").html(endLon).attr('style', "color: black; background-color: white !important;" );
    displaySatelliteImage();
}
function selectPlaceOptions(e)
{
    startLat = e.target.getAttribute('data-latstart')
    startLon = e.target.getAttribute('data-latend')
    endLat = e.target.getAttribute('data-lonstart')
    endLon = e.target.getAttribute('data-lonend')
    placeName = e.target.getAttribute('data-name')
    $("#select-place-button").html(placeName).attr('style', "color: black; background-color: white !important;" )

    $("#lat-start").html(startLat).attr('style', "color: black; background-color: white !important;" )
    $("#lat-end").html(endLat).attr('style', "color: black; background-color: white !important;" );
    $("#lon-start").html(startLon).attr('style', "color: black; background-color: white !important;" );
    $("#lon-end").html(endLon).attr('style', "color: black; background-color: white !important;" );

    displaySatelliteImage();
}
function switchToPlaces()
{
    $(".select-place").removeClass("dn");
    $(".select-co-ordinates").addClass("dn");
}
function switchToCordinates()
{
    $(".select-place").addClass("dn");
    $(".select-co-ordinates").removeClass("dn");
}
function displaySatelliteImage()
{
    if(startLat!=""&&startLon!="" && endLat!=""&&endLon!="")
    {
        $(".selected-area").removeClass("dn");
        $(".selected-placeholder").addClass("dn");
    }
}
function startAnalysis(e)
{   
    if(startLat!=""&&startLon!="" && endLat!=""&&endLon!="")
    {
        e.target.innerHTML = '<i class="fa fa-spinner fa-spin fa-fw"></i> Loading...';
        showTimeSeries();
    }
    else
        alert("Please select latitude and longitude")
}
function showTimeSeries()
{
    setTimeout(()=>{
        $(".ndmi-section").removeClass("dn")
        $([document.documentElement, document.body]).animate({
            scrollTop: $("#ndmi-section").offset().top - 100
        }, 10);
            $(".time-series-section").removeClass("dn")
            $(".forecast-section").removeClass("dn")
            $(".start-analysis").html("Start Analysis")
            
        //showForecast();
    }, 1000)
}
function scrollToForecast()
{
    
    /* setTimeout(()=>{ */
        $([document.documentElement, document.body]).animate({
            scrollTop: $("#forecast-section").offset().top
        }, 10);
    /* }, 2000) */
}
function startTransition()
{
    $('.main-content').hide()
    let i = 1;
    $('#main-content'+i).show()
    setInterval(()=>{
        if(i >= $('.main-content').length)
            i=1;
        else
            i++;
        $('.main-content').hide();
        //$('#main-content'+i).removeClass("dn");
        fadeRight('#main-content'+i)
    }, 4000);

}
$(".start-lat-options").click(selectStartLatitude);
$(".end-lat-options").click(selectEndLatitude);
$(".start-lon-options").click(selectStartLongitude);
$(".end-lon-options").click(selectEndLongitude);
$(".select-place-options").click(selectPlaceOptions);
$(".start-analysis").click(startAnalysis);
$(".forecast-button").click(scrollToForecast)
$("#switch-co-ordinates").click(switchToCordinates)
$("#switch-place-button").click(switchToPlaces)

let i = 1;
setInterval(()=>{
    $('#main-content'+i).animate({
        opacity: 0, // animate slideUp
        marginLeft: '800px'
    }, 'slow', 'linear', function() {
        $('.main-content').hide();
        $('.main-content').css({ marginLeft:'-500px'});
        if(i>=$('.main-content').length)
            i=1;
        else
            i++;
        $('#main-content'+i).show().animate({ marginLeft:'0px', opacity: 1, display: 'block'})
    })

}, 5000);