#!/usr/bin/env python3
# scripts/59-exploratorio-tamanho.py — EXPLORATÓRIO (feito DEPOIS do desbloqueio de 28/09/2026).
# Pergunta: o achado de simetria entre dedos homólogos no fluxo das cristas (H2: o_entropia)
# sobrevive ao controle do confundidor "tamanho do dedo" (medido por log(area_util))?
#
# Compara 4 versões:
# 1) o_entropia bruta (não ajustada);
# 2) Resíduo de o_entropia regredida em log(area_util) por posição de dedo;
# 3) Resíduo de o_entropia regredida em log(area_util) + qualidade_media + contraste por posição de dedo;
# 4) Perfil da própria log(area_util) (simetria de tamanho: homólogos × cruzados × mesma mão).
#
# Modelo: pares de mãos opostas (homólogos vs cruzados não-homólogos) e mesma mão.
# Nulo: permutar pessoas independentemente por dedo (10.000x).
import importlib.util, json, os, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
def carregar(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(AQUI, arq))
    m = importlib.util.module_from_spec(spec)
    sys.modules[nome] = m
    spec.loader.exec_module(m)
    return m

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
        nulo["simetria_livre_de_sessao"].append(hh - cc)
        nulo["efeito_mao"].append(mm - cc)
    for k, v in nulo.items():
        v = np.array(v)
        obs[f"p_{k}"] = float((np.sum(v >= obs[k]) + 1) / (n_nulo + 1))
    return {k: float(v) for k, v in obs.items()}

def regredir_residuos(Y_mat, X_cols_list):
    """
    Regride cada coluna j de Y_mat nas variáveis explicativas correspondentes da coluna j.
    X_cols_list: lista de matrizes (N, 10), cada uma fornecendo uma covariável para o dedo j.
    """
    N, K = Y_mat.shape
    Res = np.zeros((N, K), dtype=float)
    for j in range(K):
        covs = [cov[:, j] for cov in X_cols_list]
        X = np.column_stack([np.ones(N)] + covs)
        beta, _, _, _ = np.linalg.lstsq(X, Y_mat[:, j], rcond=None)
        Res[:, j] = Y_mat[:, j] - X @ beta
    return Res

def main():
    d = an.carregar_dados()
    N = int(d["N"])
    rng = np.random.default_rng(59)

    mat_o_entropia = d["O_entropia"]
    mat_log_area = np.log(d["area_util"])
    mat_qm = d["qualidade_media"]
    mat_ct = d["contraste"]

    # 1. Bruta
    mat_v1 = mat_o_entropia

    # 2. Resíduo em log(area_util)
    mat_v2 = regredir_residuos(mat_o_entropia, [mat_log_area])

    # 3. Resíduo em log(area_util) + qualidade_media + contraste
    mat_v3 = regredir_residuos(mat_o_entropia, [mat_log_area, mat_qm, mat_ct])

    # 4. Perfil de log(area_util)
    mat_v4 = mat_log_area

    versoes = {
        "1_o_entropia_bruta": {
            "descricao": "o_entropia bruta (sem ajuste)",
            "mat": mat_v1
        },
        "2_o_entropia_residuo_log_area": {
            "descricao": "Resíduo de o_entropia regredida em log(area_util) por dedo",
            "mat": mat_v2
        },
        "3_o_entropia_residuo_log_area_qm_ct": {
            "descricao": "Resíduo de o_entropia regredida em log(area_util)+qualidade_media+contraste por dedo",
            "mat": mat_v3
        },
        "4_log_area_util": {
            "descricao": "Perfil da própria log(area_util) (tamanho do dedo)",
            "mat": mat_v4
        }
    }

    out = {
        "ATENCAO": "EXPLORATÓRIO — pós-desbloqueio",
        "N": N,
        "n_nulo": 10000,
        "semente": 59,
        "resultados": {}
    }

    for k, info in versoes.items():
        res = resumo(info["mat"], rng)
        res["descricao"] = info["descricao"]
        out["resultados"][k] = res

    os.makedirs("out/estudo02/exploratorio", exist_ok=True)
    out_path = "out/estudo02/exploratorio/tamanho.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("=" * 115)
    print("ESTUDO 02 — EXPLORATÓRIO: CONFUNDIDOR TAMANHO DO DEDO (H2 / o_entropia)")
    print("=" * 115)
    print(f"N = {N} sujeitos | 10.000 permutações nulas por posição de dedo | Seed = 59")
    print("-" * 115)
    header = f"{'Versão':<42} | {'MI Homól':<9} | {'MI Cruz':<9} | {'MI Mesma':<9} | {'Simetria (H-C)':<15} | {'p (Simetria)':<12} | {'Efeito Mão':<11} | {'p (Mão)':<8}"
    print(header)
    print("-" * 115)

    for k, v in out["resultados"].items():
        linha = (
            f"{v['descricao']:<42} | "
            f"{v['homologos']:<9.4f} | "
            f"{v['cruzados_nao_homologos']:<9.4f} | "
            f"{v['mesma_mao']:<9.4f} | "
            f"{v['simetria_livre_de_sessao']:<15.4f} | "
            f"{v['p_simetria_livre_de_sessao']:<12.4e} | "
            f"{v['efeito_mao']:<11.4f} | "
            f"{v['p_efeito_mao']:<8.4e}"
        )
        print(linha)
    print("-" * 115)
    print(f"Arquivo salvo em: {out_path}")
    print("=" * 115)

if __name__ == "__main__":
    main()
