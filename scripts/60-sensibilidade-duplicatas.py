#!/usr/bin/env python3
# scripts/60-sensibilidade-duplicatas.py — análise de SENSIBILIDADE prevista no pré-registro do estudo 02:
# os 41 pares candidatos (out/socofing/completo/fila_ambos.csv) foram examinados por perito em 28/09/2026 —
# 41/41 "mesmo dedo" (out/comparar/vereditos.jsonl). Pessoas envolvidas têm o MESMO dedo sob rótulos diferentes
# (inclusive esquerda×direita), o que infla a simetria de homólogos. Aqui: refaz a simetria livre de sessão
# (método do 58) sem essas pessoas, e compara com a amostra completa.
import importlib.util, json, os, re, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
def carregar(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(AQUI, arq))
    m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m
m58 = carregar("m58", "58-exploratorio-maos.py")

vered = [json.loads(l) for l in open("out/comparar/vereditos.jsonl")]
assert len({(v["na"], v["nb"]) for v in vered}) == 41 and all(v["veredito"] == "MESMO DEDO" for v in vered)
excluir = {int(re.search(r"/(\d+)__", v[c]).group(1)) for v in vered for c in ("na", "nb")}

d = m58.an.carregar_dados()
manter = np.array([int(p) not in excluir for p in d["pessoas"]])
out = {"ATENCAO": "SENSIBILIDADE (prevista no pré-registro) — pós-desbloqueio",
       "excluidas": sorted(excluir), "N_completo": int(d["N"]), "N_sem_duplicatas": int(manter.sum())}
for k in ("O_entropia_ajustada", "O_entropia", "qualidade_media", "area_util"):
    out[k] = {"completo": m58.resumo(d[k], np.random.default_rng(60)),
              "sem_duplicatas": m58.resumo(d[k][manter], np.random.default_rng(60))}
os.makedirs("out/estudo02/exploratorio", exist_ok=True)
json.dump(out, open("out/estudo02/exploratorio/sensibilidade_duplicatas.json", "w"), indent=1, ensure_ascii=False)
print("excluidas:", out["excluidas"], "| N", out["N_completo"], "->", out["N_sem_duplicatas"])
for k in ("O_entropia_ajustada", "O_entropia", "qualidade_media", "area_util"):
    for q in ("completo", "sem_duplicatas"):
        r = out[k][q]
        print(f"{k:22s} {q:15s} simetria {r['simetria_livre_de_sessao']:.4f} (p {r['p_simetria_livre_de_sessao']:.4f})"
              f" | efeito mão {r['efeito_mao']:.4f}")
