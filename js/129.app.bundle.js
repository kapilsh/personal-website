(window.webpackJsonp=window.webpackJsonp||[]).push([[129,81],{391:function(e,t,a){"use strict";function n(e){function t(e,t){return"___"+e.toUpperCase()+t+"___"}var a;a=e,Object.defineProperties(a.languages["markup-templating"]={},{buildPlaceholders:{value:function(e,n,r,o){var l;e.language===n&&(l=e.tokenStack=[],e.code=e.code.replace(r,(function(a){if("function"==typeof o&&!o(a))return a;for(var r,i=l.length;-1!==e.code.indexOf(r=t(n,i));)++i;return l[i]=a,r})),e.grammar=a.languages.markup)}},tokenizePlaceholders:{value:function(e,n){var r,o;e.language===n&&e.tokenStack&&(e.grammar=a.languages[n],r=0,o=Object.keys(e.tokenStack),function l(i){for(var s=0;s<i.length&&!(r>=o.length);s++){var p,c,u,g,d,m,b,f,k,h=i[s];"string"==typeof h||h.content&&"string"==typeof h.content?(p=o[r],c=e.tokenStack[p],u="string"==typeof h?h:h.content,g=t(n,p),-1<(d=u.indexOf(g))&&(++r,m=u.substring(0,d),b=new a.Token(n,a.tokenize(c,e.grammar),"language-"+n,c),f=u.substring(d+g.length),k=[],m&&k.push.apply(k,l([m])),k.push(b),f&&k.push.apply(k,l([f])),"string"==typeof h?i.splice.apply(i,[s,1].concat(k)):h.content=k)):h.content&&l(h.content)}return i}(e.tokens))}}})}(e.exports=n).displayName="markupTemplating",n.aliases=[]},554:function(e,t,a){"use strict";var n=a(391);function r(e){var t,a,r;e.register(n),a=/(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,r=/\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b|\b0x[\dA-F]+\b/,(t=e).languages.soy={comment:[/\/\*[\s\S]*?\*\//,{pattern:/(\s)\/\/.*/,lookbehind:!0,greedy:!0}],"command-arg":{pattern:/({+\/?\s*(?:alias|call|delcall|delpackage|deltemplate|namespace|template)\s+)\.?[\w.]+/,lookbehind:!0,alias:"string",inside:{punctuation:/\./}},parameter:{pattern:/({+\/?\s*@?param\??\s+)\.?[\w.]+/,lookbehind:!0,alias:"variable"},keyword:[{pattern:/({+\/?[^\S\r\n]*)(?:\\[nrt]|alias|call|case|css|default|delcall|delpackage|deltemplate|else(?:if)?|fallbackmsg|for(?:each)?|if(?:empty)?|lb|let|literal|msg|namespace|nil|@?param\??|rb|sp|switch|template|xid)/,lookbehind:!0},/\b(?:any|as|attributes|bool|css|float|in|int|js|html|list|map|null|number|string|uri)\b/],delimiter:{pattern:/^{+\/?|\/?}+$/,alias:"punctuation"},property:/\w+(?==)/,variable:{pattern:/\$[^\W\d]\w*(?:\??(?:\.\w+|\[[^\]]+]))*/,inside:{string:{pattern:a,greedy:!0},number:r,punctuation:/[\[\].?]/}},string:{pattern:a,greedy:!0},function:[/\w+(?=\()/,{pattern:/(\|[^\S\r\n]*)\w+/,lookbehind:!0}],boolean:/\b(?:true|false)\b/,number:r,operator:/\?:?|<=?|>=?|==?|!=|[+*/%-]|\b(?:and|not|or)\b/,punctuation:/[{}()\[\]|.,:]/},t.hooks.add("before-tokenize",(function(e){var a=!1;t.languages["markup-templating"].buildPlaceholders(e,"soy",/{{.+?}}|{.+?}|\s\/\/.*|\/\*[\s\S]*?\*\//g,(function(e){return"{/literal}"===e&&(a=!1),!a&&("{literal}"===e&&(a=!0),!0)}))})),t.hooks.add("after-tokenize",(function(e){t.languages["markup-templating"].tokenizePlaceholders(e,"soy")}))}(e.exports=r).displayName="soy",r.aliases=[]}}]);