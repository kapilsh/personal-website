(window.webpackJsonp=window.webpackJsonp||[]).push([[128,81],{375:function(e,t,n){"use strict";function a(e){function t(e,t){return"___"+e.toUpperCase()+t+"___"}var n;n=e,Object.defineProperties(n.languages["markup-templating"]={},{buildPlaceholders:{value:function(e,a,o,r){var i;e.language===a&&(i=e.tokenStack=[],e.code=e.code.replace(o,(function(n){if("function"==typeof r&&!r(n))return n;for(var o,s=i.length;-1!==e.code.indexOf(o=t(a,s));)++s;return i[s]=n,o})),e.grammar=n.languages.markup)}},tokenizePlaceholders:{value:function(e,a){var o,r;e.language===a&&e.tokenStack&&(e.grammar=n.languages[a],o=0,r=Object.keys(e.tokenStack),function i(s){for(var l=0;l<s.length&&!(o>=r.length);l++){var u,p,d,c,g,f,k,m,b,h=s[l];"string"==typeof h||h.content&&"string"==typeof h.content?(u=r[o],p=e.tokenStack[u],d="string"==typeof h?h:h.content,c=t(a,u),-1<(g=d.indexOf(c))&&(++o,f=d.substring(0,g),k=new n.Token(a,n.tokenize(p,e.grammar),"language-"+a,p),m=d.substring(g+c.length),b=[],f&&b.push.apply(b,i([f])),b.push(k),m&&b.push.apply(b,i([m])),"string"==typeof h?s.splice.apply(s,[l,1].concat(b)):h.content=b)):h.content&&i(h.content)}return s}(e.tokens))}}})}(e.exports=a).displayName="markupTemplating",a.aliases=[]},537:function(e,t,n){"use strict";var a=n(375);function o(e){var t;e.register(a),(t=e).languages.smarty={comment:/\{\*[\s\S]*?\*\}/,delimiter:{pattern:/^\{|\}$/i,alias:"punctuation"},string:/(["'])(?:\\.|(?!\1)[^\\\r\n])*\1/,number:/\b0x[\dA-Fa-f]+|(?:\b\d+\.?\d*|\B\.\d+)(?:[Ee][-+]?\d+)?/,variable:[/\$(?!\d)\w+/,/#(?!\d)\w+#/,{pattern:/(\.|->)(?!\d)\w+/,lookbehind:!0},{pattern:/(\[)(?!\d)\w+(?=\])/,lookbehind:!0}],function:[{pattern:/(\|\s*)@?(?!\d)\w+/,lookbehind:!0},/^\/?(?!\d)\w+/,/(?!\d)\w+(?=\()/],"attr-name":{pattern:/\w+\s*=\s*(?:(?!\d)\w+)?/,inside:{variable:{pattern:/(=\s*)(?!\d)\w+/,lookbehind:!0},operator:/=/}},punctuation:[/[\[\]().,:`]|->/],operator:[/[+\-*\/%]|==?=?|[!<>]=?|&&|\|\|?/,/\bis\s+(?:not\s+)?(?:div|even|odd)(?:\s+by)?\b/,/\b(?:eq|neq?|gt|lt|gt?e|lt?e|not|mod|or|and)\b/],keyword:/\b(?:false|off|on|no|true|yes)\b/},t.hooks.add("before-tokenize",(function(e){var n=!1;t.languages["markup-templating"].buildPlaceholders(e,"smarty",/\{\*[\s\S]*?\*\}|\{[\s\S]+?\}/g,(function(e){return"{/literal}"===e&&(n=!1),!n&&("{literal}"===e&&(n=!0),!0)}))})),t.hooks.add("after-tokenize",(function(e){t.languages["markup-templating"].tokenizePlaceholders(e,"smarty")}))}(e.exports=o).displayName="smarty",o.aliases=[]}}]);