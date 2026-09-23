#!/usr/bin/env python3
# Compara NBIS (bozorth3) e SourceAFIS sobre os mesmos pares.
# Rótulos: "genuino" = mesma base e mesmo id; "artefato" = bases diferentes e mesmo id
# (mesmo dedo em dois sensores / identidade sintética reaproveitada — ver REPORT.md);
# "impostor" = o resto. A pergunta: os dois algoritmos concordam sobre QUAIS pares de dedos
# diferentes são os mais parecidos? Se não, a cauda é propriedade do algoritmo, não da digital.
import os, csv, json
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out")

with open(os.path.join(OUT, "index.csv")) as f:
    rec = list(csv.DictReader(f))
n = len(rec)
base = np.array([r["base"] for r in rec])
dedo = np.array([r["id_dedo"] for r in rec])
nome = [f'{r["base"]}/{r["id_dedo"]}_{r["impressao"]}' for r in rec]

nb = np.fromfile(os.path.join(OUT, "nbis", "matrix.i32"), dtype="<i4").reshape(n, n).astype(float)
sa = np.fromfile(os.path.join(OUT, "sourceafis", "matrix.f32"), dtype="<f4").reshape(n, n).astype(float)
sa = np.maximum(sa, sa.T)  # SourceAFIS não é simétrico; vale o maior dos dois sentidos

iu, ju = np.triu_indices(n, 1)
mesma_base = base[iu] == base[ju]
mesmo_id = dedo[iu] == dedo[ju]
gen = mesma_base & mesmo_id
art = ~mesma_base & mesmo_id
imp = ~mesmo_id
s_nb, s_sa = nb[iu, ju], sa[iu, ju]

def fmr(scores_imp, x):
    """Fração de impostores com escore >= x (taxa empírica de falso positivo)."""
    return float((scores_imp >= x).mean())

def resumo(s, limiar):
    q = [50, 90, 99, 99.9, 99.99]
    return {
        "genuino": {**{f"p{p}": round(float(np.percentile(s[gen], p)), 1) for p in q}, "max": round(float(s[gen].max()), 1)},
        "impostor": {**{f"p{p}": round(float(np.percentile(s[imp], p)), 1) for p in q}, "max": round(float(s[imp].max()), 1)},
        "artefato_mediana": round(float(np.median(s[art])), 1),
        "limiar": limiar,
        "impostores_acima_limiar": int((s[imp] >= limiar).sum()),
        "genuinos_abaixo_limiar_frac": round(float((s[gen] < limiar).mean()), 4),
    }

res = {
    "pares": {"genuino": int(gen.sum()), "artefato": int(art.sum()), "impostor": int(imp.sum())},
    # 40 é o limiar operacional usual do bozorth3; no SourceAFIS 40 corresponde a FMR ~0,01% (documentação).
    "nbis": resumo(s_nb, 40),
    "sourceafis": resumo(s_sa, 40),
    "spearman_impostores": round(float(spearmanr(s_nb[imp], s_sa[imp]).statistic), 3),
    "spearman_genuinos": round(float(spearmanr(s_nb[gen], s_sa[gen]).statistic), 3),
}

# O artefato de registro duplicado aparece nos dois?
a2000 = art & (((base[iu] == "FVC2000_DB1_B") & (base[ju] == "FVC2000_DB2_B")) |
               ((base[iu] == "FVC2000_DB2_B") & (base[ju] == "FVC2000_DB1_B")))
res["artefato_fvc2000_db1xdb2"] = {
    "pares": int(a2000.sum()),
    "nbis_acima_40": int((s_nb[a2000] >= 40).sum()),
    "sourceafis_acima_40": int((s_sa[a2000] >= 40).sum()),
}

# Cauda: os top-K impostores de cada um, e onde caem no outro (como FMR empírica no outro).
imp_idx = np.flatnonzero(imp)
def topo(s, k):
    o = imp_idx[np.argsort(-s[imp_idx], kind="stable")[:k]]
    return o
K = 20
tabela = []
for quem, s_a, s_b in (("nbis", s_nb, s_sa), ("sourceafis", s_sa, s_nb)):
    for p in topo(s_a, K):
        tabela.append({
            "ordenado_por": quem,
            "par": [nome[iu[p]], nome[ju[p]]],
            "nbis": round(float(s_nb[p]), 1), "nbis_fmr": fmr(s_nb[imp], s_nb[p]),
            "sourceafis": round(float(s_sa[p]), 1), "sourceafis_fmr": fmr(s_sa[imp], s_sa[p]),
        })
res["topo"] = tabela
for k in (10, 100, 1000):
    a, b = set(topo(s_nb, k)), set(topo(s_sa, k))
    res[f"sobreposicao_top{k}"] = len(a & b)

# Algum par impostor fica no extremo (FMR <= 1e-4) nos DOIS ao mesmo tempo?
r_nb = np.argsort(np.argsort(-s_nb[imp_idx])) / len(imp_idx)
r_sa = np.argsort(np.argsort(-s_sa[imp_idx])) / len(imp_idx)
ambos = np.flatnonzero((r_nb < 1e-4) & (r_sa < 1e-4))
res["impostores_no_extremo_dos_dois_(top_0.01%)"] = [
    [nome[iu[imp_idx[x]]], nome[ju[imp_idx[x]]], float(s_nb[imp_idx[x]]), round(float(s_sa[imp_idx[x]]), 1)] for x in ambos]

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "comparacao_matchers.json"), "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "topo"}, indent=1, ensure_ascii=False))
for t in tabela:
    print(f'{t["ordenado_por"]:10s} {t["par"][0]:22s} {t["par"][1]:22s} nbis={t["nbis"]:6.1f} (fmr {t["nbis_fmr"]:.1e})  sa={t["sourceafis"]:6.1f} (fmr {t["sourceafis_fmr"]:.1e})')

# ---- Nível de DEDO: agrega as 64 comparações (8x8 impressões) de cada par de dedos. ----
# Distingue "registro duplicado escondido" (quase todas as 64 altas, como o artefato DB1xDB2)
# de "semelhança estrutural" (poucos picos). Rank por mediana de cada matcher.
dedo_key = np.array([f"{b}/{d}" for b, d in zip(base, dedo)])
chaves = sorted(set(dedo_key))
pos = {k: np.flatnonzero(dedo_key == k) for k in chaves}
linhas = []
for a in range(len(chaves)):
    for b in range(a + 1, len(chaves)):
        ka, kb = chaves[a], chaves[b]
        ia, ib = pos[ka], pos[kb]
        bl_nb = nb[np.ix_(ia, ib)].ravel(); bl_sa = sa[np.ix_(ia, ib)].ravel()
        tipo = "artefato" if ka.split("/")[1] == kb.split("/")[1] else "impostor"
        linhas.append((ka, kb, tipo, np.median(bl_nb), (bl_nb >= 40).mean(), np.median(bl_sa), (bl_sa >= 40).mean()))
L = [l for l in linhas if l[2] == "impostor"]
med_nb = np.array([l[3] for l in L]); med_sa = np.array([l[5] for l in L])
rk_nb = np.argsort(np.argsort(-med_nb)); rk_sa = np.argsort(np.argsort(-med_sa))
ordem = np.argsort(rk_nb + rk_sa)
art_l = [l for l in linhas if l[2] == "artefato" and l[3] >= 20]
dedos = {
    "pares_de_dedos_impostores": len(L),
    "spearman_mediana_por_dedo": round(float(spearmanr(med_nb, med_sa).statistic), 3),
    "topo_combinado": [
        {"dedos": [L[x][0], L[x][1]], "rank_nbis": int(rk_nb[x]) + 1, "rank_sa": int(rk_sa[x]) + 1,
         "nbis_mediana": float(L[x][3]), "nbis_frac40": round(float(L[x][4]), 3),
         "sa_mediana": round(float(L[x][5]), 1), "sa_frac40": round(float(L[x][6]), 3)} for x in ordem[:10]],
    "referencia_artefatos_DB1xDB2_mediana_nbis_min": float(min(l[3] for l in art_l)) if art_l else None,
    "referencia_genuino_mediana_nbis": float(np.median(s_nb[gen])),
}
res["nivel_dedo"] = dedos
with open(os.path.join(ROOT, "results", "comparacao_matchers.json"), "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print(json.dumps(dedos, indent=1, ensure_ascii=False))
