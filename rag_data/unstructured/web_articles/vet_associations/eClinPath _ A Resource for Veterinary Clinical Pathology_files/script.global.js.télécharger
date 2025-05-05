var $j=jQuery.noConflict();
// Use jQuery via $j(...)

$j(document).ready(rdp_gcs_global_onReady);
function rdp_gcs_global_onReady(){
    $j("div.rdp-gcs-search-scope-button").on('click', function() {
        var guid = $j(this).data('guid');
        var oMenu = $j("#rdp-gcs-sub-wrapper-"+guid);
        if(oMenu.hasClass('hidden')){
            oMenu.addClass('visible').removeClass('hidden');
            var pos = $j.PositionCalculator( {
                target: this,
                targetAt: "bottom left",
                item: oMenu,
                itemAt: "top left",
                flip: "both"
            }).calculate();

            oMenu.css({
                top: parseInt(oMenu.css('top')) + pos.moveBy.y + "px",
                left: parseInt(oMenu.css('left')) + pos.moveBy.x + "px"
            });
            var formID = $j(this).data('formId');
            oMenu.data('formId',formID);
        }else{
            oMenu.addClass('hidden').removeClass('visible');
        }
    });
    
    $j(".rdp-gcs-sub-wrapper").on('mouseleave',function(){$j(this).addClass('hidden').removeClass('visible');})
    $j(".rdp-gcs-search-term").on('focus',rdp_gcs_placeholder_handler);
    $j(".rdp-gcs-search-term").on('blur',rdp_gcs_placeholder_handler);
    $j('body').on( "keydown", '.rdp-gcs-search-term' ,function(event){
        if ( event.which == 13 ) {
            var txtSearchTerm = $j('#'+this.form.id+' .rdp-gcs-search-term');
            var sPlaceholder = 'Search '+ $j('#'+this.form.id+' #txtRDPGCSMenuTitle').val();
            if($j('#'+this.form.id+' #txtRDPGCSCSEID').val() == 0){
                handle_search_and_highlight($j(this).val());
            }else{
                $j("body").unhighlight();
                if(txtSearchTerm == sPlaceholder)$j('#'+this.form.id+' #txtRDPGCSSearchTerm').val('');
                this.form.submit();                
            }

        }        
    });
    
    
    $j( ".rdp-gcs-search-form" ).submit(function( event ) {
        if($j("#txtRDPGCSCSEID",this).val() == 0)event.preventDefault();
      
    });    
    
    $j(".rdp-gcs-search-term").trigger('blur');

    $j('a.rdp-gcs-cse').on('click', function() {
        $j("body").unhighlight();
        var guid = $j(this).data('guid');
        $j('#rdp-gcs-sub-wrapper-'+guid+' p').removeClass('selected highlighted');
        $j(this).parent().addClass('selected highlighted');
        $j('#frmRDPGCSearch-'+guid+' #txtRDPGCSPage').val('1');
        $j('#frmRDPGCSearch-'+guid+' #txtRDPGCSMenuTitle').val($j(this).text());
        $j('#frmRDPGCSearch-'+guid+' #txtRDPGCSCSEID').val($j(this).data('id'));
        
        var txtSearchTerm = $j('#frmRDPGCSearch-'+guid+' .rdp-gcs-search-term');
       
        if(txtSearchTerm.hasClass('placeholder')){
            txtSearchTerm.val('').removeClass('placeholder').trigger('blur');
        }
        
        $j('#rdp-gcs-search-scope-button-'+guid).html($j(this).text() + " &#9662;");
        $j('#rdp-gcs-sub-wrapper-'+guid).addClass('hidden').removeClass('visible');
        if(!txtSearchTerm.hasClass('placeholder') && $j(this).data('id') == 0)handle_search_and_highlight(txtSearchTerm.val());
    });
    
    $j('div.gsc-cursor-page').on('click', function() {
        $j('#txtRDPGCSPage').val($j(this).data('page'));
        $j('form.rdp-gcs-search-form').submit();
    });
    
    $j('.rdp-gcs-search-button').on('click',function(){
        var txtSearchTerm = $j('#'+this.form.id+' .rdp-gcs-search-term');
        var sPlaceholder = 'Search '+ $j('#'+this.form.id+' #txtRDPGCSMenuTitle').val();
        if($j('#'+this.form.id+' #txtRDPGCSCSEID').val() == 0){
            handle_search_and_highlight(txtSearchTerm.val());            
        }else{
            $j("body").unhighlight();
            if(txtSearchTerm == sPlaceholder)$j('#'+this.form.id+' #txtRDPGCSSearchTerm').val('');
            this.form.submit();            
        }

    })
}//rdp_gcs_global_onReady


function handle_search_and_highlight(sTerm){
    var res = sTerm.split(" ");
    $j("body").unhighlight();
    $j("body").highlight(res); 
    
}//handle_search_and_highlight


function rdp_gcs_get_source_element(e){
    if( !e ) e = window.event;
    var target;
    if(e.target||e.srcElement){
        target = e.target||e.srcElement;
    }else target = e;  
    return target;    
}//rdp_gcs_get_source_element



function rdp_gcs_placeholder_handler(e){
    var target = rdp_gcs_get_source_element(e);
    var form = target.form;
    var sPlaceholder = 'Search '+ $j('#'+form.id+' #txtRDPGCSMenuTitle').val();

    if($j(target).hasClass('placeholder')){
        $j(target).val('').removeClass('placeholder');
    }else{
        if($j(target).val() == '')$j(target).val(sPlaceholder);
        if($j(target).val() == sPlaceholder)$j(target).addClass('placeholder');
    }
}//rdp_gcs_placeholder_handler

