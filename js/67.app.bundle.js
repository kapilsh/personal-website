(window.webpackJsonp=window.webpackJsonp||[]).push([[67],{407:function(t,n,e){"use strict";function a(t){!function(t){var n=t.util.clone(t.languages.javascript);t.languages.jsx=t.languages.extend("markup",n),t.languages.jsx.tag.pattern=/<\/?(?:[\w.:-]+\s*(?:\s+(?:[\w.:-]+(?:=(?:("|')(?:\\[\s\S]|(?!\1)[^\\])*\1|[^\s{'">=]+|\{(?:\{(?:\{[^}]*\}|[^{}])*\}|[^{}])+\}))?|\{\.{3}[a-z_$][\w$]*(?:\.[a-z_$][\w$]*)*\}))*\s*\/?)?>/i,t.languages.jsx.tag.inside.tag.pattern=/^<\/?[^\s>\/]*/i,t.languages.jsx.tag.inside["attr-value"].pattern=/=(?!\{)(?:("|')(?:\\[\s\S]|(?!\1)[^\\])*\1|[^\s'">]+)/i,t.languages.jsx.tag.inside.tag.inside["class-name"]=/^[A-Z]\w*(?:\.[A-Z]\w*)*$/,t.languages.insertBefore("inside","attr-name",{spread:{pattern:/\{\.{3}[a-z_$][\w$]*(?:\.[a-z_$][\w$]*)*\}/,inside:{punctuation:/\.{3}|[{}.]/,"attr-value":/\w+/}}},t.languages.jsx.tag),t.languages.insertBefore("inside","attr-value",{script:{pattern:/=(\{(?:\{(?:\{[^}]*\}|[^}])*\}|[^}])+\})/i,inside:{"script-punctuation":{pattern:/^=(?={)/,alias:"punctuation"},rest:t.languages.jsx},alias:"language-javascript"}},t.languages.jsx.tag);var e=function(t){return t?"string"==typeof t?t:"string"==typeof t.content?t.content:t.content.map(e).join(""):""},a=function(n){for(var s=[],g=0;g<n.length;g++){var i,o=n[g],p=!1;"string"!=typeof o&&("tag"===o.type&&o.content[0]&&"tag"===o.content[0].type?"</"===o.content[0].content[0].content?0<s.length&&s[s.length-1].tagName===e(o.content[0].content[1])&&s.pop():"/>"===o.content[o.content.length-1].content||s.push({tagName:e(o.content[0].content[1]),openedBraces:0}):0<s.length&&"punctuation"===o.type&&"{"===o.content?s[s.length-1].openedBraces++:0<s.length&&0<s[s.length-1].openedBraces&&"punctuation"===o.type&&"}"===o.content?s[s.length-1].openedBraces--:p=!0),(p||"string"==typeof o)&&0<s.length&&0===s[s.length-1].openedBraces&&(i=e(o),g<n.length-1&&("string"==typeof n[g+1]||"plain-text"===n[g+1].type)&&(i+=e(n[g+1]),n.splice(g+1,1)),0<g&&("string"==typeof n[g-1]||"plain-text"===n[g-1].type)&&(i=e(n[g-1])+i,n.splice(g-1,1),g--),n[g]=new t.Token("plain-text",i,null,i)),o.content&&"string"!=typeof o.content&&a(o.content)}};t.hooks.add("after-tokenize",(function(t){"jsx"!==t.language&&"tsx"!==t.language||a(t.tokens)}))}(t)}(t.exports=a).displayName="jsx",a.aliases=[]}}]);