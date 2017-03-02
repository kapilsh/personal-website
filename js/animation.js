
$(document).ready(function() {
    $(window).load(function() {
      $('table.dataframe').addClass('table').addClass('table-striped').addClass('table-hover');
      $(".input_prompt,.output_prompt").remove();
      $('.bk-root').css({'height' : 400});
      $('h1').addClass("page-header");
      $('pre.highlight code').addClass('syntax')
    });
});
