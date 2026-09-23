#!/usr/bin/env python3
# Análise da rodada completa SOCOFing (6.000 dedos, 1 impressão cada => todo par é de dedos diferentes).
# 1) Mesmo corpo x corpos diferentes, e o subgrupo "dedo homólogo da outra mão" (ex.: indicador E x D).
# 2) Crescimento do máximo com a população (subamostras de sujeitos).
# 3) Fila de candidatos (topo de cada matcher) para exame humano em scripts/comparar.py.
# Saídas: results/socofing_analise.json, out/socofing/completo/fila.csv
import os, csv, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(ROOT, "out", "socofing", "completo")

rec = list(csv.DictReader(open(os.path.join(ROOT, "out", "socofing", "index.csv"))))
n = len(rec)
suj = np.array([int(r["sujeito"]) for r in rec])
mao = np.array([r["mao"] for r in rec])
dedo = np.array([r["dedo"] for r in rec])

iu, ju = np.triu_indices(n, 1)
nb = np.fromfile(os.path.join(D, "nb_real.i32"), "<i4").reshape(n, n)[iu, ju].astype(np.float32)
sa_m = np.fromfile(os.path.join(D, "sa_real.f32"), "<f4").reshape(n, n)
sa = np.maximum(sa_m[iu, ju], sa_m[ju, iu])  # SourceAFIS não é simétrico; vale o maior sentido
del sa_m
assert (nb >= 0).all(), "matriz NBIS incompleta"

mesmo = suj[iu] == suj[ju]
homologo = mesmo & (dedo[iu] == dedo[ju]) & (mao[iu] != mao[ju])
grupos = {"corpos_diferentes": ~mesmo, "mesmo_corpo": mesmo, "mesmo_corpo_homologo": homologo,
          "mesmo_corpo_nao_homologo": mesmo & ~homologo}


def cauda(s, limiares=(30, 40, 60)):
    q = [50, 90, 99, 99.9, 99.99]
    r = {f"p{p}": round(float(np.percentile(s, p)), 1) for p in q}
    r["max"] = round(float(s.max()), 1)
    r["n"] = int(s.size)
    for t in limiares:
        k = int((s >= t).sum())
        r[f"frac_ge_{t}"] = k / s.size
        r[f"n_ge_{t}"] = k
    return r


res = {"n_dedos": n, "n_sujeitos": int(len(set(suj))), "n_pares": int(iu.size)}
for nome, s in (("nbis", nb), ("sourceafis", sa)):
    res[nome] = {g: cauda(s[m]) for g, m in grupos.items()}

# Crescimento do máximo com a população: média do máximo em 20 subamostras de k sujeitos.
rng = np.random.default_rng(0)
sujeitos = np.unique(suj)
cresc = []
for k in (30, 60, 120, 240, 480, 600):
    mx = {"nbis": [], "sourceafis": []}
    for _ in range(1 if k == len(sujeitos) else 20):
        sel = np.isin(suj, rng.choice(sujeitos, k, replace=False))
        m = sel[iu] & sel[ju] & ~mesmo
        mx["nbis"].append(float(nb[m].max()))
        mx["sourceafis"].append(float(sa[m].max()))
    cresc.append({"sujeitos": k, "pares": int(k * 10 * (k - 1) * 10 / 2),
                  "max_nbis": round(float(np.mean(mx["nbis"])), 1), "max_sa": round(float(np.mean(mx["sourceafis"])), 1)})
res["crescimento_do_maximo"] = cresc

# Posto (FMR empírica em corpos diferentes) de cada par, para cruzar os matchers.
dif = ~mesmo
ord_nb = np.sort(nb[dif]); ord_sa = np.sort(sa[dif])
fmr = lambda ordenado, x: 1 - np.searchsorted(ordenado, x, side="left") / ordenado.size

nome = lambda i: f'{rec[i]["sujeito"]}_{rec[i]["mao"]}_{rec[i]["dedo"]}'
fila = []
for quem, s in (("nbis", nb), ("sourceafis", sa)):
    for p in np.argsort(-s, kind="stable")[:25]:
        fila.append((p, quem))
vistos, linhas = set(), []
for p, quem in fila:
    if p in vistos:
        continue
    vistos.add(p)
    i, j = iu[p], ju[p]
    linhas.append({"img1": rec[i]["caminho_png"], "img2": rec[j]["caminho_png"],
                   "rotulo": f'{nome(i)} × {nome(j)} · NBIS {nb[p]:.0f} (FMR {fmr(ord_nb, nb[p]):.1e}) · SA {sa[p]:.1f} (FMR {fmr(ord_sa, sa[p]):.1e})'
                             + (" · MESMO CORPO" if mesmo[p] else ""),
                   "ordenado_por": quem})
with open(os.path.join(D, "fila.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["img1", "img2", "rotulo", "ordenado_por"])
    w.writeheader(); w.writerows(linhas)

top = lambda s, k: set(np.argsort(-s)[:k].tolist())
res["sobreposicao_top"] = {k: len(top(nb, k) & top(sa, k)) for k in (25, 100, 1000)}
ambos = np.flatnonzero((nb >= np.percentile(nb[dif], 99.999)) & (sa >= np.percentile(sa[dif], 99.999)))
res["pares_no_top_0.001pct_dos_dois"] = [[nome(iu[p]), nome(ju[p]), float(nb[p]), round(float(sa[p]), 1)] for p in ambos]

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "results", "socofing_analise.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
print(f"fila: {len(linhas)} pares -> {os.path.join(D, 'fila.csv')}")
