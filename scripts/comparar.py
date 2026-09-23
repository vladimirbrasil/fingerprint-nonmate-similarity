#!/usr/bin/env python3
# Página de comparação para exame humano: um par por tela, cada digital recortada no conteúdo,
# ampliada para ocupar metade da tela, com contraste normalizado (CLAHE) e controles de
# contraste/brilho/inversão/rotação por lado. Navegação: ← → (ou A/D) entre pares.
#
# Uso:
#   scripts/comparar.py pares.csv            # CSV com colunas img1,img2[,rotulo]
#   scripts/comparar.py --dedos data/fvc/FVC2002_DB3_B/108 data/fvc/FVC2004_DB3_B/102
#        (todas as 8x8 combinações de impressões dos dois dedos, picos dos matchers primeiro)
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


def main():
    args = sys.argv[1:]
    if args[:1] == ["--dedos"]:
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
.lado img{max-width:100%;max-height:100%;width:auto;height:100%;object-fit:contain;transform-origin:center;cursor:grab;user-select:none}
.nome{position:absolute;top:6px;left:8px;font-size:12px;color:var(--dim);background:#000a;padding:2px 6px;border-radius:4px}
.ctl{position:absolute;bottom:6px;left:50%;transform:translateX(-50%);display:flex;gap:10px;align-items:center;background:#000c;padding:4px 10px;border-radius:6px;font-size:12px;opacity:.25;transition:opacity .2s}
.lado:hover .ctl{opacity:1}
.ctl input[type=range]{width:90px}
#barra{height:44px;display:flex;align-items:center;gap:14px;padding:0 12px;background:var(--bar)}
button{background:#333;color:var(--fg);border:0;padding:6px 12px;border-radius:4px;cursor:pointer}
#rot{color:#ffcf6b}#pos{color:var(--dim)}
</style>
<div id=tela>
 <div class=lado id=L><span class=nome></span><img draggable=false><div class=ctl>
  contraste <input type=range min=50 max=300 value=100 data-k=c> brilho <input type=range min=50 max=200 value=100 data-k=b>
  girar <input type=range min=-45 max=45 value=0 data-k=r> <label><input type=checkbox data-k=i> inverter</label></div></div>
 <div class=lado id=R><span class=nome></span><img draggable=false><div class=ctl>
  contraste <input type=range min=50 max=300 value=100 data-k=c> brilho <input type=range min=50 max=200 value=100 data-k=b>
  girar <input type=range min=-45 max=45 value=0 data-k=r> <label><input type=checkbox data-k=i> inverter</label></div></div>
</div>
<div id=barra><button id=ant>← anterior</button><button id=prox>próximo →</button><span id=pos></span><span id=rot></span>
<span style="margin-left:auto;color:var(--dim)">roda do mouse = zoom · arrastar = mover · duplo clique = reset</span></div>
<script>
const P=__DADOS__;let k=0;
const st={L:{c:100,b:100,r:0,i:false,z:1,x:0,y:0},R:{c:100,b:100,r:0,i:false,z:1,x:0,y:0}};
function aplica(id){const s=st[id],im=document.querySelector('#'+id+' img');
 im.style.filter=`contrast(${s.c}%) brightness(${s.b}%)${s.i?' invert(1)':''}`;
 im.style.transform=`translate(${s.x}px,${s.y}px) rotate(${s.r}deg) scale(${s.z})`}
function mostra(){const p=P[k];
 for(const[id,src,n]of[['L',p.a,p.na],['R',p.b,p.nb]]){const e=document.getElementById(id);e.querySelector('img').src=src;e.querySelector('.nome').textContent=n;
  Object.assign(st[id],{z:1,x:0,y:0});aplica(id)}
 document.getElementById('pos').textContent=`par ${k+1} de ${P.length}`;document.getElementById('rot').textContent=p.rot||''}
for(const id of['L','R']){const e=document.getElementById(id),im=e.querySelector('img');
 e.querySelectorAll('.ctl input').forEach(inp=>inp.addEventListener('input',()=>{const s=st[id];
  s[inp.dataset.k]=inp.type=='checkbox'?inp.checked:+inp.value;aplica(id)}));
 e.addEventListener('wheel',ev=>{ev.preventDefault();const s=st[id];s.z=Math.min(8,Math.max(1,s.z*(ev.deltaY<0?1.15:1/1.15)));aplica(id)},{passive:false});
 let arr=null;im.addEventListener('pointerdown',ev=>{arr=[ev.clientX-st[id].x,ev.clientY-st[id].y];im.setPointerCapture(ev.pointerId)});
 im.addEventListener('pointermove',ev=>{if(arr){st[id].x=ev.clientX-arr[0];st[id].y=ev.clientY-arr[1];aplica(id)}});
 im.addEventListener('pointerup',()=>arr=null);
 im.addEventListener('dblclick',()=>{Object.assign(st[id],{z:1,x:0,y:0});aplica(id)})}
const vai=d=>{k=(k+d+P.length)%P.length;mostra()};
document.getElementById('ant').onclick=()=>vai(-1);document.getElementById('prox').onclick=()=>vai(1);
addEventListener('keydown',e=>{if(e.key=='ArrowRight'||e.key=='d')vai(1);if(e.key=='ArrowLeft'||e.key=='a')vai(-1)});
mostra();
</script></html>"""

if __name__ == "__main__":
    main()
