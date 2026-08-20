/* Minimal D3-compatible runtime used by rtichoke summary reports.
 * Implements only the selection, linear-scale, line, axis, grouping, max and
 * pointer primitives used by the report renderers. Keeping this tiny runtime
 * inline makes generated reports self-contained without shipping the full D3
 * distribution or requiring a CDN at viewing time.
 */
(function(global){
  const SVG='http://www.w3.org/2000/svg';
  const svgTags=new Set(['svg','g','path','rect','circle','line','text','defs','clipPath']);
  const create=(parent,tag)=>svgTags.has(tag)||parent.namespaceURI===SVG?document.createElementNS(SVG,tag):document.createElement(tag);
  class Selection{
    constructor(nodes,parents=null){this.nodes=(nodes||[]).filter(Boolean);this.parents=parents||[];this._data=null;}
    node(){return this.nodes[0]||null}
    append(tag){const out=[];this.nodes.forEach(n=>{const e=create(n,tag);e.__data__=n.__data__;n.appendChild(e);out.push(e)});return new Selection(out,this.nodes)}
    select(q){const out=this.nodes.map(n=>typeof q==='function'?q.call(n,n.__data__):n.querySelector(q)).filter(Boolean);return new Selection(out,this.nodes)}
    selectAll(q){let out=[];this.nodes.forEach(n=>{out.push(...(typeof q==='function'?q.call(n,n.__data__):n.querySelectorAll(q)))});return new Selection(out,this.nodes)}
    remove(){this.nodes.forEach(n=>n.remove());return this}
    attr(k,v){if(arguments.length===1)return this.node()?.getAttribute(k);this.nodes.forEach((n,i)=>{const x=typeof v==='function'?v.call(n,n.__data__,i):v;x==null?n.removeAttribute(k):n.setAttribute(k,x)});return this}
    style(k,v){if(arguments.length===1)return getComputedStyle(this.node()).getPropertyValue(k);this.nodes.forEach((n,i)=>n.style.setProperty(k,typeof v==='function'?v.call(n,n.__data__,i):v));return this}
    text(v){this.nodes.forEach((n,i)=>n.textContent=typeof v==='function'?v.call(n,n.__data__,i):v);return this}
    html(v){this.nodes.forEach((n,i)=>n.innerHTML=typeof v==='function'?v.call(n,n.__data__,i):v);return this}
    classed(k,v){this.nodes.forEach((n,i)=>n.classList.toggle(k,!!(typeof v==='function'?v.call(n,n.__data__,i):v)));return this}
    datum(v){if(!arguments.length)return this.node()?.__data__;this.nodes.forEach(n=>n.__data__=v);return this}
    data(v){this._data=Array.from(v||[]);return this}
    enter(){return new EnterSelection(this.parents.length?this.parents:(this.nodes[0]?.parentNode?[this.nodes[0].parentNode]:[]),this._data||[])}
    on(type,fn){this.nodes.forEach(n=>n.addEventListener(type,e=>fn.call(n,e,n.__data__)));return this}
    call(fn,...args){fn(this,...args);return this}
    each(fn){this.nodes.forEach((n,i)=>fn.call(n,n.__data__,i,this.nodes));return this}
  }
  class EnterSelection{
    constructor(parents,data){this.parents=parents;this.dataValues=data}
    append(tag){const out=[];const p=this.parents[0];if(!p)return new Selection([]);this.dataValues.forEach(d=>{const e=create(p,tag);e.__data__=d;p.appendChild(e);out.push(e)});return new Selection(out,[p])}
  }
  function select(q){return new Selection([typeof q==='string'?document.querySelector(q):q])}
  function scaleLinear(){let dom=[0,1],ran=[0,1];const s=x=>ran[0]+(x-dom[0])/(dom[1]-dom[0]||1)*(ran[1]-ran[0]);s.domain=x=>arguments.length?(dom=Array.from(x,Number),s):dom.slice();s.range=x=>arguments.length?(ran=Array.from(x,Number),s):ran.slice();s.copy=()=>scaleLinear().domain(dom).range(ran);s.ticks=(count=10)=>ticks(dom[0],dom[1],count);s.nice=()=>{const ts=ticks(dom[0],dom[1],10);if(ts.length){dom=[Math.min(dom[0],ts[0]),Math.max(dom[1],ts[ts.length-1])]}return s};return s}
  function ticks(a,b,count=10){if(!isFinite(a)||!isFinite(b)||a===b)return[a];const span=Math.abs(b-a),raw=span/Math.max(1,count),pow=10**Math.floor(Math.log10(raw)),err=raw/pow,step=(err>=7.5?10:err>=3.5?5:err>=1.5?2:1)*pow;const lo=Math.ceil(Math.min(a,b)/step)*step,hi=Math.floor(Math.max(a,b)/step)*step,out=[];for(let x=lo;x<=hi+step*1e-9;x+=step)out.push(+x.toPrecision(12));return a>b?out.reverse():out}
  function fmt(x){if(Math.abs(x)>=1000||Math.abs(x)>0&&Math.abs(x)<1e-4)return x.toExponential(0);return String(+x.toFixed(6))}
  function axis(scale,orient){let count=10;const fn=sel=>{const root=sel.node();if(!root)return;while(root.firstChild)root.removeChild(root.firstChild);const r=scale.range(),vals=scale.ticks(count),horizontal=orient==='bottom';const domain=document.createElementNS(SVG,'path');domain.setAttribute('class','domain');domain.setAttribute('fill','none');domain.setAttribute('stroke','currentColor');domain.setAttribute('d',horizontal?`M${r[0]},0H${r[1]}`:`M0,${r[0]}V${r[1]}`);root.appendChild(domain);vals.forEach(v=>{const g=document.createElementNS(SVG,'g');g.setAttribute('class','tick');g.setAttribute('transform',horizontal?`translate(${scale(v)},0)`:`translate(0,${scale(v)})`);const l=document.createElementNS(SVG,'line');l.setAttribute('stroke','currentColor');horizontal?l.setAttribute('y2','6'):l.setAttribute('x2','-6');const t=document.createElementNS(SVG,'text');t.setAttribute('fill','currentColor');t.setAttribute('font-size','10');t.setAttribute('font-family','sans-serif');if(horizontal){t.setAttribute('y','9');t.setAttribute('dy','0.71em');t.setAttribute('text-anchor','middle')}else{t.setAttribute('x','-9');t.setAttribute('dy','0.32em');t.setAttribute('text-anchor','end')}t.textContent=fmt(v);g.append(l,t);root.appendChild(g)})};fn.ticks=n=>(count=n,fn);return fn}
  function line(){let fx=d=>d[0],fy=d=>d[1],defined=()=>true;const gen=data=>{let out='',started=false;for(const d of data||[]){if(!defined(d)){started=false;continue}const x=fx(d),y=fy(d);if(!isFinite(x)||!isFinite(y)){started=false;continue}out+=(started?'L':'M')+x+','+y;started=true}return out};gen.x=f=>(fx=f,gen);gen.y=f=>(fy=f,gen);gen.defined=f=>(defined=f,gen);return gen}
  function group(values,key){const m=new Map;for(const v of values||[]){const k=key(v);if(!m.has(k))m.set(k,[]);m.get(k).push(v)}return m}
  function max(values,accessor=x=>x){let out=-Infinity;for(const v of values||[]){const x=+accessor(v);if(isFinite(x)&&x>out)out=x}return out===-Infinity?undefined:out}
  function pointer(ev,node=ev.currentTarget){const r=node.getBoundingClientRect();const vb=node.viewBox?.baseVal;if(vb&&r.width&&r.height)return[(ev.clientX-r.left)*vb.width/r.width+vb.x,(ev.clientY-r.top)*vb.height/r.height+vb.y];return[ev.clientX-r.left,ev.clientY-r.top]}
  global.d3={select,scaleLinear,line,axisBottom:s=>axis(s,'bottom'),axisLeft:s=>axis(s,'left'),group,max,pointer};
})(globalThis);
