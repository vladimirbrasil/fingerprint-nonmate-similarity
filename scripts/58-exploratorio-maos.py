#!/usr/bin/env python3
# scripts/58-exploratorio-maos.py — EXPLORATÓRIO (feito DEPOIS do desbloqueio de 28/09/2026).
# Pergunta: o excesso de informação entre dedos homólogos (H2) é simetria do corpo ou efeito de sessão?
# Se cada mão foi colhida à parte, a sessão aproxima dedos da MESMA mão. Comparando só pares de mãos
# OPOSTAS, homólogos (Esq. indicador × Dir. indicador) e não-homólogos (Esq. indicador × Dir. anelar)
# sofrem a mesma exposição à sessão: a diferença entre eles é a simetria livre da sessão.
# Nulo: permutar pessoas independentemente em cada dedo (igual ao pré-registro), 10.000 vezes.
import importlib.util, json, os, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
def carregar(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(AQUI, arq))
    m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m
an = carregar("analise52", "52-analise-cega.py")
est = carregar("est51", "51-estimadores.py")

def mi_de_R(R):  # MI gaussiana par a par a partir da correlação
    r2 = np.clip(R ** 2, 0, 0.999999)
    return -0.5 * np.log(1 - r2)

def classes(MI):
    esq, dir_ = range(5), range(5, 10)
    homol = [MI[i, i + 5] for i in esq]
    cruz_nao = [MI[i, j] for i in esq for j in dir_ if j != i + 5]
    mesma = [MI[i, j] for grupo in (esq, dir_) for i in grupo for j in grupo if i < j]
    return np.mean(homol), np.mean(cruz_nao), np.mean(mesma)

def resumo(mat, rng, n_nulo=10000):
    Z = est.copula_transform(mat)
    h, c, m = classes(mi_de_R(np.corrcoef(Z, rowvar=False)))
    obs = {"homologos": h, "cruzados_nao_homologos": c, "mesma_mao": m,
           "simetria_livre_de_sessao": h - c, "efeito_mao": m - c}
    nulo = {k: [] for k in ("simetria_livre_de_sessao", "efeito_mao")}
    Zp = np.empty_like(Z)
    for _ in range(n_nulo):
        for j in range(10):
            Zp[:, j] = Z[rng.permutation(len(Z)), j]
        hh, cc, mm = classes(mi_de_R(np.corrcoef(Zp, rowvar=False)))
        nulo["simetria_livre_de_sessao"].append(hh - cc); nulo["efeito_mao"].append(mm - cc)
    for k, v in nulo.items():
        v = np.array(v); obs[f"p_{k}"] = float((np.sum(v >= obs[k]) + 1) / (n_nulo + 1))
    return {k: float(v) for k, v in obs.items()}

if __name__ == "__main__":
    d = an.carregar_dados()
    rng = np.random.default_rng(58)
    out = {"ATENCAO": "EXPLORATÓRIO — pós-desbloqueio", "N": int(d["N"])}
    # primárias ajustadas + qualidade/área como diagnóstico de sessão (ângulo fica fora: não é gaussiano)
    for k in ("O_entropia_ajustada", "Q_densidade_ajustada", "qualidade_media", "area_util"):
        out[k] = resumo(d[k], rng)
    os.makedirs("out/estudo02/exploratorio", exist_ok=True)
    json.dump(out, open("out/estudo02/exploratorio/maos.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
