#!/usr/bin/env python3
# Página de comparação para exame humano: um par por tela, cada digital recortada no conteúdo,
# ampliada para ocupar metade da tela, com contraste normalizado (CLAHE) e controles de
# contraste/brilho/inversão/rotação por lado. Navegação: ← → (ou A/D) entre pares.
#
# Uso:
#   scripts/comparar.py pares.csv            # CSV com colunas img1,img2[,rotulo]
#   scripts/comparar.py --dedos data/fvc/FVC2002_DB3_B/108 data/fvc/FVC2004_DB3_B/102
#        (todas as 8x8 combinações de impressões dos dois dedos, picos dos matchers primeiro)
#   scripts/comparar.py --fila 15            # candidatos: top-15 pares de dedos reais de cada matcher
#   scripts/comparar.py --figura pontos.json # figura com os pontos marcados (exportados da página)
# Marcação: tecla M liga/desliga; clique na esquerda e depois o correspondente na direita; Z desfaz.
# Saída: out/comparar/index.html — sirva com: python3 -m http.server -d out/comparar 8765
import os, sys, csv, glob, json, html
import numpy as np
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out", "comparar")


def preparar(caminho, destino):
    """Recorta no conteúdo da digital e aplica CLAHE. Retorna o nome do PNG gerado."""
    nome = os.path.relpath(caminho, ROOT).replace("/", "__").rsplit(".", 1)[0] + ".png"
    alvo = os.path.join(destino, nome)
    if os.path.exists(alvo):
        return nome
    g = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    # Máscara de conteúdo: desvio padrão local alto = crista; fundo e borda de sensor são lisos.
    f = g.astype(np.float32)
    k = max(9, (min(g.shape) // 30) | 1)
    media = cv2.blur(f, (k, k))
    desvio = np.sqrt(np.maximum(cv2.blur(f * f, (k, k)) - media * media, 0))
    mascara = (desvio > np.percentile(desvio, 90) * 0.5).astype(np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, rot, stats, _ = cv2.connectedComponentsWithStats(mascara)
    if n > 1:
        maior = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h = stats[maior, :4]
        m = max(4, int(0.03 * max(w, h)))
        x0, y0 = max(0, x - m), max(0, y - m)
        g = g[y0:min(g.shape[0], y + h + m), x0:min(g.shape[1], x + w + m)]
    # Ampliação real (não só CSS) para o CLAHE trabalhar na escala final e a imagem não borrar.
    escala = max(1, int(round(900 / max(g.shape))))
    if escala > 1:
        g = cv2.resize(g, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    g = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(g)
    cv2.imwrite(alvo, g)
    return nome


def pares_de_dedos(dedo_a, dedo_b):
    """Todas as combinações de impressões; ordena pelos escores dos dois matchers, se existirem."""
    ia = sorted(glob.glob(dedo_a + "_*.*"))
    ib = sorted(glob.glob(dedo_b + "_*.*"))
    escore = {}
    try:
        idx = [l.split(",") for l in open(os.path.join(ROOT, "out", "index.csv")).read().split("\n")[1:] if l]
        n = len(idx)
        pos = {f"data/fvc/{r[1]}/{r[2]}_{r[3]}": i for i, r in enumerate(idx)}
        nb = np.fromfile(os.path.join(ROOT, "out", "nbis", "matrix.i32"), "<i4").reshape(n, n)
        sa = np.fromfile(os.path.join(ROOT, "out", "sourceafis", "matrix.f32"), "<f4").reshape(n, n)
        for a in ia:
            for b in ib:
                i = pos[os.path.relpath(a, ROOT).rsplit(".", 1)[0]]
                j = pos[os.path.relpath(b, ROOT).rsplit(".", 1)[0]]
                escore[(a, b)] = (int(nb[i, j]), float(max(sa[i, j], sa[j, i])))
    except (OSError, KeyError, ValueError):
        pass
    pares = [(a, b) for a in ia for b in ib]
    if escore:
        pares.sort(key=lambda p: -(escore[p][0] / 59 + escore[p][1] / 54))
    rot = lambda p: (f"NBIS {escore[p][0]} · SourceAFIS {escore[p][1]:.1f}" if p in escore else "")
    return [(a, b, rot((a, b))) for a, b in pares]


def fila_candidatos(n_top):
    """Pares de dedos REAIS (sem DB4 sintética, sem artefato de mesmo índice) mais próximos em cada
    matcher, pela mediana das 64 comparações; uma tela por par de dedos, com o pico combinado."""
    idx = [l.split(",") for l in open(os.path.join(ROOT, "out", "index.csv")).read().split("\n")[1:] if l]
    n = len(idx)
    nb = np.fromfile(os.path.join(ROOT, "out", "nbis", "matrix.i32"), "<i4").reshape(n, n).astype(float)
    sa = np.fromfile(os.path.join(ROOT, "out", "sourceafis", "matrix.f32"), "<f4").reshape(n, n).astype(float)
    sa = np.maximum(sa, sa.T)
    dedos = {}
    for i, r in enumerate(idx):
        if "DB4" not in r[1]:
            dedos.setdefault((r[1], r[2]), []).append(i)
    chaves = sorted(dedos)
    linhas = []
    for a in range(len(chaves)):
        for b in range(a + 1, len(chaves)):
            if chaves[a][1] == chaves[b][1]:
                continue  # mesmo índice em bases diferentes = artefato já conhecido
            ia, ib = dedos[chaves[a]], dedos[chaves[b]]
            bn, bs = nb[np.ix_(ia, ib)], sa[np.ix_(ia, ib)]
            comb = bn / 59 + bs / 54
            u, v = np.unravel_index(comb.argmax(), comb.shape)
            linhas.append((chaves[a], chaves[b], np.median(bn), np.median(bs), ia[u], ib[v], nb[ia[u], ib[v]], sa[ia[u], ib[v]]))
    mn = np.array([l[2] for l in linhas]); ms = np.array([l[3] for l in linhas])
    rn = np.argsort(np.argsort(-mn)); rs = np.argsort(np.argsort(-ms))
    escolhidos = sorted({int(x) for x in np.flatnonzero((rn < n_top) | (rs < n_top))}, key=lambda x: min(rn[x], rs[x]))
    cam = lambda i: os.path.join(ROOT, "data", "fvc", idx[i][1], f"{idx[i][2]}_{idx[i][3]}.tif")
    out = []
    for x in escolhidos:
        l = linhas[x]
        out.append((cam(l[4]), cam(l[5]),
                    f"{l[0][0]}/{l[0][1]} × {l[1][0]}/{l[1][1]} · mediana NBIS {l[2]:.0f} (#{rn[x]+1}) · SA {l[3]:.1f} (#{rs[x]+1}) · pico NBIS {l[6]:.0f} / SA {l[7]:.1f}"))
    print(f"{len(linhas)} pares de dedos reais; {len(out)} no top-{n_top} de algum matcher")
    return out


def figura(json_path):
    """Gera PNG lado a lado com os pontos correspondentes numerados (exportados da página)."""
    d = json.load(open(json_path))
    a = cv2.cvtColor(cv2.imread(os.path.join(OUT, d["a"]), 0), cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(cv2.imread(os.path.join(OUT, d["b"]), 0), cv2.COLOR_GRAY2BGR)
    h = max(a.shape[0], b.shape[0])
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h)); b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    for img, lado in ((a, "L"), (b, "R")):
        for num, pt in enumerate(d["pontos"], 1):
            if pt.get(lado):
                x, y = int(pt[lado][0] * img.shape[1]), int(pt[lado][1] * img.shape[0])
                cv2.circle(img, (x, y), 14, (0, 0, 230), 3)
                cv2.putText(img, str(num), (x + 16, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 230), 3)
    sep = np.full((h, 12, 3), 255, np.uint8)
    alvo = os.path.join(ROOT, "results", os.path.basename(json_path).rsplit(".", 1)[0] + ".png")
    cv2.imwrite(alvo, np.hstack([a, sep, b]))
    print(f"{len(d['pontos'])} pontos -> {alvo}")


def main():
    args = sys.argv[1:]
    if args[:1] == ["--figura"]:
        return figura(args[1])
    if args[:1] == ["--fila"]:
        pares = fila_candidatos(int(args[1]) if len(args) > 1 else 15)
    elif args[:1] == ["--dedos"]:
        pares = pares_de_dedos(os.path.abspath(args[1]), os.path.abspath(args[2]))
    else:
        with open(args[0]) as f:
            pares = [(os.path.abspath(r["img1"]), os.path.abspath(r["img2"]), r.get("rotulo", "")) for r in csv.DictReader(f)]
    os.makedirs(OUT, exist_ok=True)
    dados = [{"a": preparar(a, OUT), "b": preparar(b, OUT),
              "na": os.path.relpath(a, ROOT), "nb": os.path.relpath(b, ROOT), "rot": r} for a, b, r in pares]
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(PAGINA.replace("__DADOS__", json.dumps(dados)))
    print(f"{len(dados)} pares -> {os.path.join(OUT, 'index.html')}")


PAGINA = r"""<!doctype html><html lang=pt-BR><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Comparação de digitais</title>
<style>
:root{--bg:#111;--fg:#eee;--dim:#999;--bar:#1c1c1c}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
#tela{display:flex;height:calc(100vh - 44px)}
.lado{flex:1;position:relative;overflow:hidden;display:flex;align-items:center;justify-content:center;border-right:1px solid #333}
.lado:last-child{border-right:0}
.pan{position:relative;height:100%;display:inline-block;transform-origin:center;cursor:grab;user-select:none}
.pan img{height:100%;width:auto;display:block;pointer-events:none}
.mk{position:absolute;width:22px;height:22px;margin:-11px 0 0 -11px;border:3px solid #ff3b3b;border-radius:50%;pointer-events:none}
.mk b{position:absolute;left:18px;top:-16px;color:#ff3b3b;font:bold 15px system-ui;text-shadow:0 0 3px #000}
body.marcar .pan{cursor:crosshair}#modo{color:#ff6b6b}
.nome{position:absolute;top:6px;left:8px;font-size:12px;color:var(--dim);background:#000a;padding:2px 6px;border-radius:4px}
.ctl{position:absolute;bottom:6px;left:50%;transform:translateX(-50%);display:flex;gap:10px;align-items:center;background:#000c;padding:4px 10px;border-radius:6px;font-size:12px;opacity:.25;transition:opacity .2s}
.lado:hover .ctl{opacity:1}
.ctl input[type=range]{width:90px}
#barra{height:44px;display:flex;align-items:center;gap:14px;padding:0 12px;background:var(--bar)}
button{background:#333;color:var(--fg);border:0;padding:6px 12px;border-radius:4px;cursor:pointer}
#rot{color:#ffcf6b}#pos{color:var(--dim)}
</style>
<div id=tela>
 <div class=lado id=L><span class=nome></span><div class=pan><img draggable=false></div><div class=ctl>
  contraste <input type=range min=50 max=300 value=100 data-k=c> brilho <input type=range min=50 max=200 value=100 data-k=b>
  girar <input type=range min=-45 max=45 value=0 data-k=r> <label><input type=checkbox data-k=i> inverter</label></div></div>
 <div class=lado id=R><span class=nome></span><div class=pan><img draggable=false></div><div class=ctl>
  contraste <input type=range min=50 max=300 value=100 data-k=c> brilho <input type=range min=50 max=200 value=100 data-k=b>
  girar <input type=range min=-45 max=45 value=0 data-k=r> <label><input type=checkbox data-k=i> inverter</label></div></div>
</div>
<div id=barra><button id=ant>← anterior</button><button id=prox>próximo →</button><span id=pos></span><span id=rot></span><span id=modo></span><button id=exp>exportar pontos</button>
<span style="margin-left:auto;color:var(--dim)">M = marcar pontos · Z = desfazer · roda = zoom · arrastar = mover · duplo clique = reset</span></div>
<script>
const P=__DADOS__;let k=0;
const st={L:{c:100,b:100,r:0,i:false,z:1,x:0,y:0},R:{c:100,b:100,r:0,i:false,z:1,x:0,y:0}};
let marcar=false;const chave=p=>'pontos:'+p.na+'|'+p.nb;
const pts=()=>{try{return JSON.parse(localStorage.getItem(chave(P[k]))||'[]')}catch(e){return[]}};
const salva=a=>{try{localStorage.setItem(chave(P[k]),JSON.stringify(a))}catch(e){}};
function desenha(){const a=pts();for(const id of['L','R']){const pan=document.querySelector('#'+id+' .pan');
 pan.querySelectorAll('.mk').forEach(m=>m.remove());
 a.forEach((p,n)=>{if(!p[id])return;const m=document.createElement('div');m.className='mk';
  m.style.left=p[id][0]*100+'%';m.style.top=p[id][1]*100+'%';m.innerHTML='<b>'+(n+1)+'</b>';pan.appendChild(m)})}
 const f=a.filter(p=>p.L&&p.R).length;document.getElementById('modo').textContent=(marcar?'MARCANDO · ':'')+(a.length?f+' pontos pareados':'')}
function aplica(id){const s=st[id],pan=document.querySelector('#'+id+' .pan');
 pan.querySelector('img').style.filter=`contrast(${s.c}%) brightness(${s.b}%)${s.i?' invert(1)':''}`;
 pan.style.transform=`translate(${s.x}px,${s.y}px) rotate(${s.r}deg) scale(${s.z})`}
function mostra(){const p=P[k];
 for(const[id,src,n]of[['L',p.a,p.na],['R',p.b,p.nb]]){const e=document.getElementById(id);e.querySelector('.pan img').src=src;e.querySelector('.nome').textContent=n;
  Object.assign(st[id],{z:1,x:0,y:0});aplica(id)}
 document.getElementById('pos').textContent=`par ${k+1} de ${P.length}`;document.getElementById('rot').textContent=p.rot||'';desenha()}
for(const id of['L','R']){const e=document.getElementById(id),im=e.querySelector('.pan');
 e.querySelectorAll('.ctl input').forEach(inp=>inp.addEventListener('input',()=>{const s=st[id];
  s[inp.dataset.k]=inp.type=='checkbox'?inp.checked:+inp.value;aplica(id)}));
 e.addEventListener('wheel',ev=>{ev.preventDefault();const s=st[id];s.z=Math.min(8,Math.max(1,s.z*(ev.deltaY<0?1.15:1/1.15)));aplica(id)},{passive:false});
 let arr=null,ini=null;im.addEventListener('pointerdown',ev=>{arr=[ev.clientX-st[id].x,ev.clientY-st[id].y];ini=[ev.clientX,ev.clientY];im.setPointerCapture(ev.pointerId)});
 im.addEventListener('pointermove',ev=>{if(arr&&!marcar){st[id].x=ev.clientX-arr[0];st[id].y=ev.clientY-arr[1];aplica(id)}});
 im.addEventListener('pointerup',ev=>{arr=null;if(!marcar||Math.hypot(ev.clientX-ini[0],ev.clientY-ini[1])>4)return;
  // posição no sistema da imagem (desfaz zoom/rotação): usa o retângulo do img sem transform via offset
  const r=im.getBoundingClientRect(),cx=r.left+r.width/2,cy=r.top+r.height/2,s=st[id],t=-s.r*Math.PI/180;
  let dx=(ev.clientX-cx)/s.z,dy=(ev.clientY-cy)/s.z;[dx,dy]=[dx*Math.cos(t)-dy*Math.sin(t),dx*Math.sin(t)+dy*Math.cos(t)];
  const w=im.offsetWidth,h=im.offsetHeight,pt=[.5+dx/w,.5+dy/h];
  const a=pts();let alvo=a.find(p=>!p[id]);if(!alvo){alvo={};a.push(alvo)}alvo[id]=pt;salva(a);desenha()});
 im.addEventListener('dblclick',()=>{Object.assign(st[id],{z:1,x:0,y:0});aplica(id)})}
const vai=d=>{k=(k+d+P.length)%P.length;mostra()};
document.getElementById('ant').onclick=()=>vai(-1);document.getElementById('prox').onclick=()=>vai(1);
document.getElementById('exp').onclick=()=>{const p=P[k],j=JSON.stringify({a:p.a,b:p.b,na:p.na,nb:p.nb,pontos:pts()},null,1);
 const u=URL.createObjectURL(new Blob([j],{type:'application/json'})),l=document.createElement('a');
 l.href=u;l.download='pontos_'+p.na.split('/').slice(-2).join('_')+'__'+p.nb.split('/').slice(-2).join('_')+'.json';l.click()};
addEventListener('keydown',e=>{if(e.key=='m'||e.key=='M'){marcar=!marcar;document.body.classList.toggle('marcar',marcar);desenha()}
 if(e.key=='z'||e.key=='Z'){const a=pts();const u=a[a.length-1];if(u){if(u.R)delete u.R;else a.pop()}salva(a);desenha()}if(e.key=='ArrowRight'||e.key=='d')vai(1);if(e.key=='ArrowLeft'||e.key=='a')vai(-1)});
mostra();
</script></html>"""

if __name__ == "__main__":
    main()
