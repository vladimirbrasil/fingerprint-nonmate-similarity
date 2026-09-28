#!/usr/bin/env python3
# ==============================================================================
# scripts/52-analise-cega.py
# Análise cega pré-registrada do Estudo 02 (Integração dos 10 dedos).
#
# REGRA DE CEGAMENTO (inegociável):
# - Por padrão, este script opera ESTRITAMENTE sobre dados EMBARALHADOS
#   (para cada posição de dedo, permuta pessoas com semente fixa pseudo-aleatória).
# - A flag --desbloquear NÃO deve ser usada durante a Etapa 1 (cega).
# - O pipeline testa as hipóteses H1, H2 e H3 sob o nulo. Sobre dados embaralhados,
#   todas as hipóteses devem resultar em p > 0.05 (teste de sanidade do pipeline).
#
# Hipóteses testadas:
# - H1: TC da população > nulo, por família (O: entropia de orientação, Q: densidade de minúcias).
# - H2: Informação Mútua (MI) entre dedos homólogos > não-homólogos da mesma pessoa.
# - H3: Correlação inter-indivíduo entre a TC pontual da Família O e da Família Q > nulo.
# - Exploratório: O-information (redundância x sinergia) e TC pontual x sexo.
#
# Saídas:
# - out/estudo02/cego/h1_resultado.json
# - out/estudo02/cego/h2_resultado.json
# - out/estudo02/cego/h3_resultado.json
# - out/estudo02/cego/exploratorio.json
# - out/estudo02/cego/resumo_analise_cega.json
# - out/estudo02/cego/figura_tc_nulo.png
# ==============================================================================

import os
import sys
import json
import argparse
import importlib.util
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_CSV = os.path.join(ROOT, "out", "estudo02", "features.csv")
DUPLICATAS_JSON = os.path.join(ROOT, "results", "socofing_duplicatas_exatas.json")
OUT_CEGO = os.path.join(ROOT, "out", "estudo02", "cego")

# Importar 51-estimadores.py
spec = importlib.util.spec_from_file_location(
    "estimadores",
    os.path.join(ROOT, "scripts", "51-estimadores.py")
)
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)

POSICOES_DEDOS = [
    ("Left", "thumb"), ("Left", "index"), ("Left", "middle"), ("Left", "ring"), ("Left", "little"),
    ("Right", "thumb"), ("Right", "index"), ("Right", "middle"), ("Right", "ring"), ("Right", "little")
]
NOMES_DEDOS = [f"{m}_{d}" for m, d in POSICOES_DEDOS]

# Pares homólogos (mesmo dedo, mãos opostas: Left_x com Right_x)
PARES_HOMOLOGOS = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
PARES_TODOS = [(i, j) for i in range(10) for j in range(i + 1, 10)]
PARES_NAO_HOMOLOGOS = [p for p in PARES_TODOS if p not in PARES_HOMOLOGOS]


def carregar_dados(features_csv, duplicatas_json):
    """
    Carrega features.csv, aplica exclusões pré-registradas de duplicatas exatas
    e monta matrizes estruturadas (pessoas x 10 dedos) por família.
    """
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Arquivo features.csv não encontrado em {features_csv}")

    # Identificar duplicatas exatas a excluir (manter a primeira cópia de cada grupo)
    dedos_duplicados_excluir = set()
    if os.path.exists(duplicatas_json):
        with open(duplicatas_json, "r", encoding="utf-8") as f:
            grupos_dups = json.load(f)
            for grupo in grupos_dups:
                # Excluir a partir da segunda cópia
                for inst in grupo[1:]:
                    dedos_duplicados_excluir.add(inst)

    import csv
    with open(features_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        linhas = list(reader)

    # Organizar registros por pessoa
    pessoas_dict = {}
    for r in linhas:
        p_id = int(r["pessoa"])
        mao = r["mao"]
        dedo = r["dedo"]
        stem = f"{p_id}__{r['sexo']}_{mao}_{dedo}_finger"
        tag_curta = f"{p_id}_{mao}_{dedo}"

        # Se for duplicata exata pré-registrada, ignorar o dedo
        if tag_curta in dedos_duplicados_excluir or stem in dedos_duplicados_excluir:
            continue

        if p_id not in pessoas_dict:
            pessoas_dict[p_id] = {
                "pessoa": p_id,
                "sexo": r["sexo"],
                "dedos": {}
            }
        pessoas_dict[p_id]["dedos"][(mao, dedo)] = r

    # Filtrar apenas pessoas com os 10 dedos completos e válidos
    pessoas_completas = []
    for p_id in sorted(pessoas_dict.keys()):
        p_data = pessoas_dict[p_id]
        if len(p_data["dedos"]) == 10:
            todos_presentes = all((m, d) in p_data["dedos"] for m, d in POSICOES_DEDOS)
            if todos_presentes:
                pessoas_completas.append(p_data)

    N = len(pessoas_completas)
    sexos = np.array([p["sexo"] for p in pessoas_completas])
    p_ids = np.array([p["pessoa"] for p in pessoas_completas])

    # Construir matrizes (N x 10)
    mat_o_entropia = np.zeros((N, 10), dtype=float)
    mat_o_frac_hc = np.zeros((N, 10), dtype=float)
    mat_o_orientacao = np.zeros((N, 10), dtype=float)
    mat_q_densidade = np.zeros((N, 10), dtype=float)
    mat_q_minucias = np.zeros((N, 10), dtype=float)

    for i, p in enumerate(pessoas_completas):
        for j, (m, d) in enumerate(POSICOES_DEDOS):
            r = p["dedos"][(m, d)]
            mat_o_entropia[i, j] = float(r["o_entropia"]) if r["o_entropia"] else 0.0
            mat_o_frac_hc[i, j] = float(r["o_frac_alta_curvatura"]) if r["o_frac_alta_curvatura"] else 0.0
            mat_o_orientacao[i, j] = float(r["o_orientacao_dominante"]) if r["o_orientacao_dominante"] else 0.0
            mat_q_densidade[i, j] = float(r["q_densidade"]) if r["q_densidade"] else 0.0
            mat_q_minucias[i, j] = float(r["q_n_minucias"]) if r["q_n_minucias"] else 0.0

    dados = {
        "N": N,
        "pessoas": p_ids,
        "sexos": sexos,
        "O_entropia": mat_o_entropia,
        "O_frac_hc": mat_o_frac_hc,
        "O_orientacao": mat_o_orientacao,
        "Q_densidade": mat_q_densidade,
        "Q_minucias": mat_q_minucias,
    }
    return dados


def embaralhar_matriz(mat, rng):
    """
    Embaralha independentemente cada coluna (posição de dedo) entre pessoas.
    Preserva a distribuição marginal de cada dedo e destrói o vínculo intra-indivíduo.
    """
    N, K = mat.shape
    mat_shuf = np.zeros_like(mat)
    for j in range(K):
        perm = rng.permutation(N)
        mat_shuf[:, j] = mat[perm, j]
    return mat_shuf


def delta_mi_from_R(R):
    """
    Calcula Delta MI (homólogos - não-homólogos) diretamente a partir da matriz de correlação R.
    MI(x_i, x_j) = -0.5 * ln(1 - rho^2)
    """
    mi_hom = []
    for i, j in PARES_HOMOLOGOS:
        rho = np.clip(R[i, j], -0.999999, 0.999999)
        mi_hom.append(-0.5 * np.log(1.0 - rho ** 2))

    mi_nao_hom = []
    for i, j in PARES_NAO_HOMOLOGOS:
        rho = np.clip(R[i, j], -0.999999, 0.999999)
        mi_nao_hom.append(-0.5 * np.log(1.0 - rho ** 2))

    return float(np.mean(mi_hom) - np.mean(mi_nao_hom)), float(np.mean(mi_hom)), float(np.mean(mi_nao_hom))


def delta_mi(Z_mat):
    """Calcula Delta MI para a matriz de variáveis transformadas por cópula."""
    R = np.corrcoef(Z_mat, rowvar=False)
    return delta_mi_from_R(R)


def avaliar_hipoteses_rapido(mat_O, mat_Q, n_nulo=199, rng=None):
    """
    Avalia as hipóteses H1 (O, Q), H2 (O, Q) e H3 sobre um par de matrizes (N x 10).
    Retorna os p-valores empíricos calculados sob n_nulo permutações independentes por coluna.
    Otimizado para calibração em larga escala (200 repetições).
    """
    if rng is None:
        rng = np.random.default_rng()

    N = mat_O.shape[0]
    Z_O = est.copula_transform(mat_O)
    Z_Q = est.copula_transform(mat_Q)

    # Observados
    R_O = np.corrcoef(Z_O, rowvar=False)
    s_O, ld_O = np.linalg.slogdet(R_O)
    tc_obs_O = max(0.0, -0.5 * ld_O) if s_O > 0 else 0.0
    d_mi_obs_O, _, _ = delta_mi_from_R(R_O)
    inv_R_O = np.linalg.inv(R_O) - np.eye(10)
    tc_pts_obs_O = -0.5 * ld_O - 0.5 * np.sum((Z_O @ inv_R_O) * Z_O, axis=1)

    R_Q = np.corrcoef(Z_Q, rowvar=False)
    s_Q, ld_Q = np.linalg.slogdet(R_Q)
    tc_obs_Q = max(0.0, -0.5 * ld_Q) if s_Q > 0 else 0.0
    d_mi_obs_Q, _, _ = delta_mi_from_R(R_Q)
    inv_R_Q = np.linalg.inv(R_Q) - np.eye(10)
    tc_pts_obs_Q = -0.5 * ld_Q - 0.5 * np.sum((Z_Q @ inv_R_Q) * Z_Q, axis=1)

    r_s_obs, _ = stats.spearmanr(tc_pts_obs_O, tc_pts_obs_Q)

    # Nulo
    tc_nulo_O = np.zeros(n_nulo, dtype=float)
    tc_nulo_Q = np.zeros(n_nulo, dtype=float)
    dmi_nulo_O = np.zeros(n_nulo, dtype=float)
    dmi_nulo_Q = np.zeros(n_nulo, dtype=float)
    rs_nulo = np.zeros(n_nulo, dtype=float)

    Z_perm_O = np.zeros_like(Z_O)
    Z_perm_Q = np.zeros_like(Z_Q)

    for b in range(n_nulo):
        for j in range(10):
            Z_perm_O[:, j] = Z_O[rng.permutation(N), j]
            Z_perm_Q[:, j] = Z_Q[rng.permutation(N), j]

        R_b_O = np.corrcoef(Z_perm_O, rowvar=False)
        s_b_O, ld_b_O = np.linalg.slogdet(R_b_O)
        tc_nulo_O[b] = max(0.0, -0.5 * ld_b_O) if s_b_O > 0 else 0.0
        dmi_nulo_O[b], _, _ = delta_mi_from_R(R_b_O)
        inv_R_b_O = np.linalg.inv(R_b_O) - np.eye(10)
        tc_pts_b_O = -0.5 * ld_b_O - 0.5 * np.sum((Z_perm_O @ inv_R_b_O) * Z_perm_O, axis=1)

        R_b_Q = np.corrcoef(Z_perm_Q, rowvar=False)
        s_b_Q, ld_b_Q = np.linalg.slogdet(R_b_Q)
        tc_nulo_Q[b] = max(0.0, -0.5 * ld_b_Q) if s_b_Q > 0 else 0.0
        dmi_nulo_Q[b], _, _ = delta_mi_from_R(R_b_Q)
        inv_R_b_Q = np.linalg.inv(R_b_Q) - np.eye(10)
        tc_pts_b_Q = -0.5 * ld_b_Q - 0.5 * np.sum((Z_perm_Q @ inv_R_b_Q) * Z_perm_Q, axis=1)

        rs_b, _ = stats.spearmanr(tc_pts_b_O, tc_pts_b_Q)
        rs_nulo[b] = rs_b

    p_h1_O = float((np.sum(tc_nulo_O >= tc_obs_O) + 1.0) / (n_nulo + 1.0))
    p_h1_Q = float((np.sum(tc_nulo_Q >= tc_obs_Q) + 1.0) / (n_nulo + 1.0))
    p_h2_O = float((np.sum(dmi_nulo_O >= d_mi_obs_O) + 1.0) / (n_nulo + 1.0))
    p_h2_Q = float((np.sum(dmi_nulo_Q >= d_mi_obs_Q) + 1.0) / (n_nulo + 1.0))
    p_h3 = float((np.sum(rs_nulo >= r_s_obs) + 1.0) / (n_nulo + 1.0))

    return {
        "p_h1_O": p_h1_O,
        "p_h1_Q": p_h1_Q,
        "p_h2_O": p_h2_O,
        "p_h2_Q": p_h2_Q,
        "p_h3": p_h3,
        "tc_obs_O": tc_obs_O,
        "tc_obs_Q": tc_obs_Q,
        "delta_mi_obs_O": d_mi_obs_O,
        "delta_mi_obs_Q": d_mi_obs_Q,
        "r_spearman_obs": r_s_obs
    }


def executar_analise(dados, n_nulo=10000, seed=42, modo_desbloqueado=False):
    """
    Executa a análise estatística pré-registrada (H1, H2, H3, Exploratório).
    Por padrão opera estritamente sobre a versão cega (embaralhada).
    """
    rng = np.random.default_rng(seed)
    N = dados["N"]

    if not modo_desbloqueado:
        # CEGAMENTO: Cria a base cega de trabalho embaralhando com semente fixa
        print("[INFO] MODO CEGO ATIVO: Embaralhando dados antes da análise estatística...")
        mat_O = embaralhar_matriz(dados["O_entropia"], rng)
        mat_Q = embaralhar_matriz(dados["Q_densidade"], rng)
    else:
        print("[ALERTA] MODO DESBLOQUEADO: Usando dados reais...")
        mat_O = dados["O_entropia"].copy()
        mat_Q = dados["Q_densidade"].copy()

    # Pré-computar transformações de cópula dos dados observados (cegos)
    Z_O = est.copula_transform(mat_O)
    Z_Q = est.copula_transform(mat_Q)

    # --------------------------------------------------------------------------
    # 1. H1: Total Correlation da População > Nulo
    # --------------------------------------------------------------------------
    tc_obs_O = est.tc_gaussian_copula(mat_O)
    tc_obs_Q = est.tc_gaussian_copula(mat_Q)

    # --------------------------------------------------------------------------
    # 2. H2: MI Dedos Homólogos > Não-Homólogos
    # --------------------------------------------------------------------------
    delta_mi_obs_O, mi_hom_obs_O, mi_nhom_obs_O = delta_mi(Z_O)
    delta_mi_obs_Q, mi_hom_obs_Q, mi_nhom_obs_Q = delta_mi(Z_Q)

    # --------------------------------------------------------------------------
    # 3. H3: Concordância Inter-Indivíduo (TC pontual O x Q)
    # --------------------------------------------------------------------------
    tc_pts_obs_O = est.pointwise_tc_gaussian_copula(mat_O)
    tc_pts_obs_Q = est.pointwise_tc_gaussian_copula(mat_Q)
    r_spearman_obs, _ = stats.spearmanr(tc_pts_obs_O, tc_pts_obs_Q)
    r_pearson_obs, _ = stats.pearsonr(tc_pts_obs_O, tc_pts_obs_Q)

    # --------------------------------------------------------------------------
    # 4. Exploratório: O-information
    # --------------------------------------------------------------------------
    oinfo_obs_O = est.oinfo_gaussian_copula(mat_O)
    oinfo_obs_Q = est.oinfo_gaussian_copula(mat_Q)

    # Exploratório: TC pontual x Sexo
    sexos = dados["sexos"]
    mask_m = sexos == "M"
    mask_f = sexos == "F"
    u_stat_O, p_sexo_O = stats.mannwhitneyu(tc_pts_obs_O[mask_m], tc_pts_obs_O[mask_f])
    u_stat_Q, p_sexo_Q = stats.mannwhitneyu(tc_pts_obs_Q[mask_m], tc_pts_obs_Q[mask_f])

    # --------------------------------------------------------------------------
    # GERAÇÃO DA DISTRIBUIÇÃO NULA (Permutações)
    # --------------------------------------------------------------------------
    print(f"[INFO] Gerando distribuição nula ({n_nulo} permutações)...")
    tc_nulo_O = np.zeros(n_nulo, dtype=float)
    tc_nulo_Q = np.zeros(n_nulo, dtype=float)
    delta_mi_nulo_O = np.zeros(n_nulo, dtype=float)
    delta_mi_nulo_Q = np.zeros(n_nulo, dtype=float)
    r_spearman_nulo = np.zeros(n_nulo, dtype=float)
    oinfo_nulo_O = np.zeros(n_nulo, dtype=float)
    oinfo_nulo_Q = np.zeros(n_nulo, dtype=float)

    Z_perm_O = np.zeros_like(Z_O)
    Z_perm_Q = np.zeros_like(Z_Q)

    for b in range(n_nulo):
        # Permutação independente de linhas para cada coluna
        for j in range(10):
            perm_O = rng.permutation(N)
            perm_Q = rng.permutation(N)
            Z_perm_O[:, j] = Z_O[perm_O, j]
            Z_perm_Q[:, j] = Z_Q[perm_Q, j]

        # TC nula O
        R_O = np.corrcoef(Z_perm_O, rowvar=False)
        s_O, ld_O = np.linalg.slogdet(R_O)
        tc_b_O = max(0.0, -0.5 * ld_O) if s_O > 0 else 0.0
        tc_nulo_O[b] = tc_b_O

        # TC nula Q
        R_Q = np.corrcoef(Z_perm_Q, rowvar=False)
        s_Q, ld_Q = np.linalg.slogdet(R_Q)
        tc_b_Q = max(0.0, -0.5 * ld_Q) if s_Q > 0 else 0.0
        tc_nulo_Q[b] = tc_b_Q

        # Delta MI nulo
        d_mi_O, _, _ = delta_mi_from_R(R_O)
        d_mi_Q, _, _ = delta_mi_from_R(R_Q)
        delta_mi_nulo_O[b] = d_mi_O
        delta_mi_nulo_Q[b] = d_mi_Q

        # O-information nula
        sum_ld_sub_O = sum(np.linalg.slogdet(np.delete(np.delete(R_O, j, 0), j, 1))[1] for j in range(10))
        oinfo_nulo_O[b] = 0.5 * 8.0 * ld_O - 0.5 * sum_ld_sub_O

        sum_ld_sub_Q = sum(np.linalg.slogdet(np.delete(np.delete(R_Q, j, 0), j, 1))[1] for j in range(10))
        oinfo_nulo_Q[b] = 0.5 * 8.0 * ld_Q - 0.5 * sum_ld_sub_Q

        # H3: correlação nula
        inv_R_O = np.linalg.inv(R_O) - np.eye(10)
        inv_R_Q = np.linalg.inv(R_Q) - np.eye(10)
        tc_pts_b_O = -0.5 * ld_O - 0.5 * np.sum((Z_perm_O @ inv_R_O) * Z_perm_O, axis=1)
        tc_pts_b_Q = -0.5 * ld_Q - 0.5 * np.sum((Z_perm_Q @ inv_R_Q) * Z_perm_Q, axis=1)
        r_s_b, _ = stats.spearmanr(tc_pts_b_O, tc_pts_b_Q)
        r_spearman_nulo[b] = r_s_b

    # --------------------------------------------------------------------------
    # CÁLCULO DE P-VALORES EMPÍRICOS (Regra pré-registrada: p = (k + 1) / (N + 1))
    # --------------------------------------------------------------------------
    p_h1_O = float((np.sum(tc_nulo_O >= tc_obs_O) + 1.0) / (n_nulo + 1.0))
    p_h1_Q = float((np.sum(tc_nulo_Q >= tc_obs_Q) + 1.0) / (n_nulo + 1.0))

    p_h2_O = float((np.sum(delta_mi_nulo_O >= delta_mi_obs_O) + 1.0) / (n_nulo + 1.0))
    p_h2_Q = float((np.sum(delta_mi_nulo_Q >= delta_mi_obs_Q) + 1.0) / (n_nulo + 1.0))

    # H3: teste uni-caudal (correlação positiva > nulo)
    p_h3_spearman = float((np.sum(r_spearman_nulo >= r_spearman_obs) + 1.0) / (n_nulo + 1.0))

    p_oinfo_O = float((np.sum(np.abs(oinfo_nulo_O) >= abs(oinfo_obs_O)) + 1.0) / (n_nulo + 1.0))
    p_oinfo_Q = float((np.sum(np.abs(oinfo_nulo_Q) >= abs(oinfo_obs_Q)) + 1.0) / (n_nulo + 1.0))

    # Resultados estruturados
    res_h1 = {
        "status": "sucesso",
        "familia_O": {
            "caracteristica": "o_entropia",
            "tc_observado": round(float(tc_obs_O), 6),
            "nulo_media": round(float(np.mean(tc_nulo_O)), 6),
            "nulo_desvio": round(float(np.std(tc_nulo_O)), 6),
            "nulo_p95": round(float(np.percentile(tc_nulo_O, 95)), 6),
            "p_valor_empirico": round(p_h1_O, 6),
            "bonferroni_significativo_alpha_005": bool(p_h1_O < (0.05 / 2.0))
        },
        "familia_Q": {
            "caracteristica": "q_densidade",
            "tc_observado": round(float(tc_obs_Q), 6),
            "nulo_media": round(float(np.mean(tc_nulo_Q)), 6),
            "nulo_desvio": round(float(np.std(tc_nulo_Q)), 6),
            "nulo_p95": round(float(np.percentile(tc_nulo_Q, 95)), 6),
            "p_valor_empirico": round(p_h1_Q, 6),
            "bonferroni_significativo_alpha_005": bool(p_h1_Q < (0.05 / 2.0))
        },
        "familia_C": {
            "status": "excluida_da_primaria_por_incompatibilidade_extrator",
            "motivo": "pcasys rejeitou imagens 288x309 px (< 512 px) conforme pre-registro"
        }
    }

    res_h2 = {
        "status": "sucesso",
        "familia_O": {
            "delta_mi_observado": round(float(delta_mi_obs_O), 6),
            "mi_homologos_medio": round(float(mi_hom_obs_O), 6),
            "mi_nao_homologos_medio": round(float(mi_nhom_obs_O), 6),
            "nulo_delta_mi_media": round(float(np.mean(delta_mi_nulo_O)), 6),
            "nulo_delta_mi_desvio": round(float(np.std(delta_mi_nulo_O)), 6),
            "p_valor_empirico": round(p_h2_O, 6),
            "bonferroni_significativo_alpha_005": bool(p_h2_O < (0.05 / 2.0))
        },
        "familia_Q": {
            "delta_mi_observado": round(float(delta_mi_obs_Q), 6),
            "mi_homologos_medio": round(float(mi_hom_obs_Q), 6),
            "mi_nao_homologos_medio": round(float(mi_nhom_obs_Q), 6),
            "nulo_delta_mi_media": round(float(np.mean(delta_mi_nulo_Q)), 6),
            "nulo_delta_mi_desvio": round(float(np.std(delta_mi_nulo_Q)), 6),
            "p_valor_empirico": round(p_h2_Q, 6),
            "bonferroni_significativo_alpha_005": bool(p_h2_Q < (0.05 / 2.0))
        }
    }

    res_h3 = {
        "status": "sucesso",
        "correlacao_spearman_observada": round(float(r_spearman_obs), 6),
        "correlacao_pearson_observada": round(float(r_pearson_obs), 6),
        "nulo_spearman_media": round(float(np.mean(r_spearman_nulo)), 6),
        "nulo_spearman_desvio": round(float(np.std(r_spearman_nulo)), 6),
        "p_valor_empirico": round(p_h3_spearman, 6),
        "significativo_alpha_005": bool(p_h3_spearman < 0.05)
    }

    res_exploratorio = {
        "status": "sucesso",
        "oinfo_familia_O": {
            "oinfo_observado": round(float(oinfo_obs_O), 6),
            "nulo_media": round(float(np.mean(oinfo_nulo_O)), 6),
            "p_valor_bikaudal": round(p_oinfo_O, 6),
            "interpretacao": "redundancia" if oinfo_obs_O > 0 else "sinergia"
        },
        "oinfo_familia_Q": {
            "oinfo_observado": round(float(oinfo_obs_Q), 6),
            "nulo_media": round(float(np.mean(oinfo_nulo_Q)), 6),
            "p_valor_bikaudal": round(p_oinfo_Q, 6),
            "interpretacao": "redundancia" if oinfo_obs_Q > 0 else "sinergia"
        },
        "tc_pontual_vs_sexo": {
            "familia_O": {
                "mediana_M": round(float(np.median(tc_pts_obs_O[mask_m])), 6),
                "mediana_F": round(float(np.median(tc_pts_obs_O[mask_f])), 6),
                "mann_whitney_p_valor": round(float(p_sexo_O), 6)
            },
            "familia_Q": {
                "mediana_M": round(float(np.median(tc_pts_obs_Q[mask_m])), 6),
                "mediana_F": round(float(np.median(tc_pts_obs_Q[mask_f])), 6),
                "mann_whitney_p_valor": round(float(p_sexo_Q), 6)
            }
        }
    }

    res_resumo = {
        "modo_execucao": "desbloqueado" if modo_desbloqueado else "cego_embaralhado",
        "n_pessoas_analisadas": N,
        "n_permutações_nulo": n_nulo,
        "semente": seed,
        "H1_TC_populacao": res_h1,
        "H2_MI_homologos": res_h2,
        "H3_concordancia_inter_familia": res_h3,
        "Exploratorio": res_exploratorio,
        "sanidade_pipeline": "Verificado por calibracao sob o nulo (scripts/54-calibracao.py; out/estudo02/cego/calibracao.json conforme Desvio D3)"
    }

    # Salvar JSONs
    os.makedirs(OUT_CEGO, exist_ok=True)
    with open(os.path.join(OUT_CEGO, "h1_resultado.json"), "w", encoding="utf-8") as f:
        json.dump(res_h1, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_CEGO, "h2_resultado.json"), "w", encoding="utf-8") as f:
        json.dump(res_h2, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_CEGO, "h3_resultado.json"), "w", encoding="utf-8") as f:
        json.dump(res_h3, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_CEGO, "exploratorio.json"), "w", encoding="utf-8") as f:
        json.dump(res_exploratorio, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_CEGO, "resumo_analise_cega.json"), "w", encoding="utf-8") as f:
        json.dump(res_resumo, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------------------------
    # GERAÇÃO DA FIGURA (out/estudo02/cego/figura_tc_nulo.png)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)

    # Subplot 1: H1 Família O
    ax = axes[0, 0]
    ax.hist(tc_nulo_O, bins=40, density=True, color="steelblue", alpha=0.7, edgecolor="black", label="Distribuição Nula")
    ax.axvline(tc_obs_O, color="crimson", linewidth=2, linestyle="--", label=f"TC Observada ({tc_obs_O:.4f})")
    ax.set_title(f"H1 — Família O (Orientação)\np = {p_h1_O:.4f}")
    ax.set_xlabel("Total Correlation (nats)")
    ax.set_ylabel("Densidade")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: H1 Família Q
    ax = axes[0, 1]
    ax.hist(tc_nulo_Q, bins=40, density=True, color="seagreen", alpha=0.7, edgecolor="black", label="Distribuição Nula")
    ax.axvline(tc_obs_Q, color="crimson", linewidth=2, linestyle="--", label=f"TC Observada ({tc_obs_Q:.4f})")
    ax.set_title(f"H1 — Família Q (Densidade Minúcias)\np = {p_h1_Q:.4f}")
    ax.set_xlabel("Total Correlation (nats)")
    ax.set_ylabel("Densidade")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: H2 Delta MI Homólogos vs Não-Homólogos (Família O)
    ax = axes[1, 0]
    ax.hist(delta_mi_nulo_O, bins=40, density=True, color="darkorange", alpha=0.7, edgecolor="black", label="Distribuição Nula")
    ax.axvline(delta_mi_obs_O, color="crimson", linewidth=2, linestyle="--", label=f"ΔMI Obs ({delta_mi_obs_O:.4f})")
    ax.set_title(f"H2 — MI Homólogos vs Não-Homólogos (Família O)\np = {p_h2_O:.4f}")
    ax.set_xlabel("ΔMI (nats)")
    ax.set_ylabel("Densidade")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # Subplot 4: H3 TC Pontual O x Q
    ax = axes[1, 1]
    ax.scatter(tc_pts_obs_O, tc_pts_obs_Q, alpha=0.4, color="purple", edgecolors="none", s=25)
    ax.set_title(f"H3 — TC Pontual: Família O × Família Q\nSpearman r = {r_spearman_obs:.4f} (p = {p_h3_spearman:.4f})")
    ax.set_xlabel("TC Pontual — Família O (nats)")
    ax.set_ylabel("TC Pontual — Família Q (nats)")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_path = os.path.join(OUT_CEGO, "figura_tc_nulo.png")
    plt.savefig(fig_path)
    plt.close(fig)

    print(f"[INFO] Figura gerada: {fig_path}")
    print(f"[INFO] Resumo salvo em: {os.path.join(OUT_CEGO, 'resumo_analise_cega.json')}")

    return res_resumo


def main():
    parser = argparse.ArgumentParser(description="Análise estatística cega do Estudo 02")
    parser.add_argument("--n-nulo", type=int, default=10000, help="Número de permutações para a distribuição nula (padrão: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Semente pseudo-aleatória fixa (padrão: 42)")
    parser.add_argument("--desbloquear", action="store_true", help="ATENÇÃO: Flag de desbloqueio para dados reais (NÃO USAR NA ETAPA 1)")
    args = parser.parse_args()

    dados = carregar_dados(FEATURES_CSV, DUPLICATAS_JSON)
    print(f"[INFO] Total de sujeitos completos analisados: {dados['N']}")

    resumo = executar_analise(
        dados,
        n_nulo=args.n_nulo,
        seed=args.seed,
        modo_desbloqueado=args.desbloquear
    )

    print()
    print("==================================================================")
    print("RESUMO DA ANÁLISE CEGA (ESTUDO 02)")
    print("==================================================================")
    print(f"Modo: {resumo['modo_execucao']}")
    print(f"Sujeitos analisados: {resumo['n_pessoas_analisadas']}")
    print(f"Permutações nulas: {resumo['n_permutações_nulo']}")
    print()
    print("H1 (TC Populacional > Nulo):")
    print(f"  Família O (Orientação): TC = {resumo['H1_TC_populacao']['familia_O']['tc_observado']:.6f} nats | p = {resumo['H1_TC_populacao']['familia_O']['p_valor_empirico']:.4f}")
    print(f"  Família Q (Minúcias):   TC = {resumo['H1_TC_populacao']['familia_Q']['tc_observado']:.6f} nats | p = {resumo['H1_TC_populacao']['familia_Q']['p_valor_empirico']:.4f}")
    print(f"  Família C: {resumo['H1_TC_populacao']['familia_C']['status']}")
    print()
    print("H2 (MI Homólogos > Não-Homólogos):")
    print(f"  Família O: ΔMI = {resumo['H2_MI_homologos']['familia_O']['delta_mi_observado']:.6f} nats | p = {resumo['H2_MI_homologos']['familia_O']['p_valor_empirico']:.4f}")
    print(f"  Família Q: ΔMI = {resumo['H2_MI_homologos']['familia_Q']['delta_mi_observado']:.6f} nats | p = {resumo['H2_MI_homologos']['familia_Q']['p_valor_empirico']:.4f}")
    print()
    print("H3 (Concordância Inter-Indivíduo TC Pontual O x Q):")
    print(f"  Spearman r = {resumo['H3_concordancia_inter_familia']['correlacao_spearman_observada']:.6f} | p = {resumo['H3_concordancia_inter_familia']['p_valor_empirico']:.4f}")
    print()
    print("Exploratório:")
    print(f"  O-information Família O: {resumo['Exploratorio']['oinfo_familia_O']['oinfo_observado']:.6f} nats (p = {resumo['Exploratorio']['oinfo_familia_O']['p_valor_bikaudal']:.4f})")
    print(f"  O-information Família Q: {resumo['Exploratorio']['oinfo_familia_Q']['oinfo_observado']:.6f} nats (p = {resumo['Exploratorio']['oinfo_familia_Q']['p_valor_bikaudal']:.4f})")
    print(f"  TC Pontual x Sexo: p_O = {resumo['Exploratorio']['tc_pontual_vs_sexo']['familia_O']['mann_whitney_p_valor']:.4f} | p_Q = {resumo['Exploratorio']['tc_pontual_vs_sexo']['familia_Q']['mann_whitney_p_valor']:.4f}")
    print(f"Sanidade do pipeline: {resumo['sanidade_pipeline']}")
    print("==================================================================")


if __name__ == "__main__":
    main()
