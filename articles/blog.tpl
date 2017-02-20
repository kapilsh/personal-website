{%- extends 'basic.tpl' -%}
{% from 'mathjax.tpl' import mathjax %}

{%- block header -%}
{%- block html_head -%}
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>{{resources['metadata']['name']}}</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>

<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css"/>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.3.0/css/font-awesome.min.css"/>
<link rel="stylesheet" href="../assets/css/styles.min.css"/>
<link rel="stylesheet" href="../assets/css/pygments.css"/>

{%- endblock html_head -%}
{%- endblock header -%}

{% block body %}
  <div id="fb-root"></div>
  <script>(function(d, s, id) {
    var js, fjs = d.getElementsByTagName(s)[0];
    if (d.getElementById(id)) return;
    js = d.createElement(s); js.id = id;
    js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&version=v2.8";
    fjs.parentNode.insertBefore(js, fjs);
  }(document, 'script', 'facebook-jssdk'));
  </script>
    <div class="cover">
      <header class="navbar navbar-static-top">
        <nav class="container">
          <div class="navbar-header"><a href="../index.html" class="navbar-brand">{K:S}</a>
            <button data-toggle="collapse" data-target="#nav-header-collapse" aria-expanded="false" class="navbar-toggle collapsed"><span class="icon-bar"></span><span class="icon-bar"></span><span class="icon-bar">    </span></button>
          </div>
          <div id="nav-header-collapse" class="collapse navbar-collapse">
            <ul class="nav navbar-nav">
              <li><a href="../index.html"><span class="fa fa-home">&nbsp;</span><span class="capitalized">HOME</span></a></li>
              <li><a href="../html/about.html"><span class="fa fa-user">&nbsp;</span><span class="capitalized">ABOUT</span></a></li>
              <li class="dropdown"><a class="dropdown-toggle"  data-toggle="dropdown" href="html/blog.html"><span class="fa fa-book">&nbsp;</span><span class="capitalized">BLOG</span>
                <span class="caret"></span></a>
                <ul class="dropdown-menu">
                  <li><a href="../html/blog.html">All</a></li>
                  <li><hr></li>
                  <li><a href="../html/blog.html">Mathematics</a></li>
                  <li><a href="../html/blog.html">Programming</a></li>
                  <li><a href="../html/blog.html">Quantitative Trading</a></li>
                </ul>
              </li>
              <li><a href="../html/resources.html"><span class="fa fa-external-link">&nbsp;</span><span class="capitalized">RESOURCES</span></a></li>
            </ul>
          </div>
        </nav>
      </header>
    </div>
    <div class="section section-primary">
        <div class="container" id="notebook-container">
            <div class="row">
              <div class="col-md-1">
                <a href="https://twitter.com/share"
                class="twitter-share-button"
                data-text="Kapil Sharma Blog Post"
                data-via="kapilthequant"
                data-show-count="true">
              </a>
              <script async src="//platform.twitter.com/widgets.js" charset="utf-8">
              </script>
              <div class="fb-share-button"
              data-href="https://developers.facebook.com/docs/plugins/"
              data-layout="button" data-mobile-iframe="true"><a class="fb-xfbml-parse-ignore" target="_blank" href="https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fdevelopers.facebook.com%2Fdocs%2Fplugins%2F&amp;src=sdkpreparse"></a>
              </div>
              <div>
                <script src="//platform.linkedin.com/in.js" type="text/javascript"> lang: en_US</script>
                <script type="IN/Share" data-counter="right"></script>
              </div>
              </div>
                <div class="col-md-11">
                    {{ super() }}
                </div>
            </div>
        </div>
    </div>
    <footer class="section section-primary">
      <div class="container">
        <div class="row">
          <div class="col-md-12">
            <h6 id="copyright">2017 &copy; Copyright Kapil Sharma</h6>
          </div>
        </div>
      </div>
    </footer>
    <script>
    $('h1').addClass("page-header");
    $(".input_prompt,.output_prompt").remove();
    $('table.dataframe').addClass('table').addClass('table-striped').addClass('table-hover');
    </script>
    <script>
    (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
      (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
      m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
    })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

    ga('create', 'UA-92270634-1', 'auto');
    ga('send', 'pageview');
    </script>
{%- endblock body %}
