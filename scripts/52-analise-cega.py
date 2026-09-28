#!/usr/bin/env python3
# ==============================================================================
# scripts/52-analise-cega.py
# Análise cega pré-registrada do Estudo 02 (Integração dos 10 dedos).
#
# REGRA DE CEGAMENTO (inegociável):
# - Por padrão, este script opera ESTRITAMENTE sobre dados EMBARALHADOS
#   (para cada posição de dedo, permuta pessoas com semente fixa pseudo-aleatória).
# - A flag --desbloquear NÃO deve ser usada durante a Etapa 1 (cega).
# - O pipeline testa as hipóteses H1, H2 e H3 sob o nulo.
#
# ESTRUTURA DOS TESTES (Desvios D4b, D5, D7):
# 1) Análise PRIMÁRIA (Ajustada):
#    - Para cada posição de dedo j (0..9), calcula o resíduo após regressão linear
#      em [qualidade_media, area_util, contraste, sexo].
#    - H1 (O, Q), H2 (O, Q) e H3 são testados sobre esses resíduos ajustados.
#    - Bonferroni com alpha = 0.01 (0.05 / 5 testes primários).
# 2) Análise SECUNDÁRIA (Bruta):
#    - Executa os mesmos testes sobre as variáveis brutas (o_entropia e q_densidade).
# 3) Teste D4a (Efeito de Sessão):
#    - Total Correlation da própria qualidade_media nos 10 dedos x nulo (sob cegamento).
#
# Saídas:
# - out/estudo02/cego/h1_resultado.json
# - out/estudo02/cego/h2_resultado.json
# - out/estudo02/cego/h3_resultado.json
# - out/estudo02/cego/d4a_qualidade_resultado.json
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
FEATURES_QUALIDADE_CSV = os.path.join(ROOT, "out", "estudo02", "features_qualidade.csv")
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


def ajustar_residuos_por_dedo(Y_mat, QM_mat, AU_mat, CT_mat, sexos_bin):
    """
    Ajusta para cada posição de dedo j in 0..9 a regressão linear:
    Y_j = beta_0 + beta_1*QM_j + beta_2*AU_j + beta_3*CT_j + beta_4*sexo + erro_j
    Retorna a matriz de resíduos (N x 10) e os coeficientes ajustados.

    Ordem e Justificativa (Desvios D4b + D5):
    O ajuste deve ser feito ANTES do embaralhamento no pipeline cego e dentro de
    cada iteração do nulo de forma coerente: o que se testa sob H0 é a independência
    inter-digital dos resíduos ajustados (características intrínsecas livres de artefatos
    de sessão/qualidade e sexo). O embaralhamento permuta os resíduos de cada dedo
    independentemente entre indivíduos, preservando a distribuição marginal empírica
    exata de cada resíduo e destruindo o vínculo intra-corpo.
    """
    N, K = Y_mat.shape
    Res = np.zeros_like(Y_mat, dtype=float)
    betas = []
    for j in range(K):
        X = np.column_stack([np.ones(N), QM_mat[:, j], AU_mat[:, j], CT_mat[:, j], sexos_bin])
        beta, _, _, _ = np.linalg.lstsq(X, Y_mat[:, j], rcond=None)
        y_pred = X @ beta
        Res[:, j] = Y_mat[:, j] - y_pred
        betas.append(beta.tolist())
    return Res, betas


def carregar_dados(features_csv=None, duplicatas_json=None):
    """
    Carrega features_qualidade.csv (ou features.csv), aplica exclusões pré-registradas
    de duplicatas exatas e monta matrizes estruturadas (pessoas x 10 dedos).
    Calcula os resíduos ajustados (Desvios D4b e D5).
    """
    if features_csv is None:
        if os.path.exists(FEATURES_QUALIDADE_CSV):
            features_csv = FEATURES_QUALIDADE_CSV
        else:
            features_csv = FEATURES_CSV

    if duplicatas_json is None:
        duplicatas_json = DUPLICATAS_JSON

    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Arquivo de características não encontrado em {features_csv}")

    # Identificar duplicatas exatas a excluir (manter a primeira cópia de cada grupo)
    dedos_duplicados_excluir = set()
    if os.path.exists(duplicatas_json):
        with open(duplicatas_json, "r", encoding="utf-8") as f:
            grupos_dups = json.load(f)
            for grupo in grupos_dups:
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
    sexos_bin = np.array([1.0 if s == "M" else 0.0 for s in sexos])
    p_ids = np.array([p["pessoa"] for p in pessoas_completas])

    # Construir matrizes (N x 10)
    mat_o_entropia = np.zeros((N, 10), dtype=float)
    mat_o_frac_hc = np.zeros((N, 10), dtype=float)
    mat_o_orientacao = np.zeros((N, 10), dtype=float)
    mat_q_densidade = np.zeros((N, 10), dtype=float)
    mat_q_minucias = np.zeros((N, 10), dtype=float)

    mat_qualidade_media = np.zeros((N, 10), dtype=float)
    mat_area_util = np.zeros((N, 10), dtype=float)
    mat_contraste = np.zeros((N, 10), dtype=float)

    for i, p in enumerate(pessoas_completas):
        for j, (m, d) in enumerate(POSICOES_DEDOS):
            r = p["dedos"][(m, d)]
            mat_o_entropia[i, j] = float(r["o_entropia"]) if r.get("o_entropia") else 0.0
            mat_o_frac_hc[i, j] = float(r["o_frac_alta_curvatura"]) if r.get("o_frac_alta_curvatura") else 0.0
            mat_o_orientacao[i, j] = float(r["o_orientacao_dominante"]) if r.get("o_orientacao_dominante") else 0.0
            mat_q_densidade[i, j] = float(r["q_densidade"]) if r.get("q_densidade") else 0.0
            mat_q_minucias[i, j] = float(r["q_n_minucias"]) if r.get("q_n_minucias") else 0.0

            mat_qualidade_media[i, j] = float(r["qualidade_media"]) if r.get("qualidade_media") else 0.0
            mat_area_util[i, j] = float(r["area_util"]) if r.get("area_util") else float(r.get("q_area_util", 0))
            mat_contraste[i, j] = float(r["contraste"]) if r.get("contraste") else 0.0

    # Ajuste dos resíduos por regressão linear (D4b + D5)
    res_O, betas_O = ajustar_residuos_por_dedo(mat_o_entropia, mat_qualidade_media, mat_area_util, mat_contraste, sexos_bin)
    res_Q, betas_Q = ajustar_residuos_por_dedo(mat_q_densidade, mat_qualidade_media, mat_area_util, mat_contraste, sexos_bin)

    dados = {
        "N": N,
        "pessoas": p_ids,
        "sexos": sexos,
        "sexos_bin": sexos_bin,
        "O_entropia": mat_o_entropia,
        "O_frac_hc": mat_o_frac_hc,
        "O_orientacao": mat_o_orientacao,
        "Q_densidade": mat_q_densidade,
        "Q_minucias": mat_q_minucias,
        "qualidade_media": mat_qualidade_media,
        "area_util": mat_area_util,
        "contraste": mat_contraste,
        "O_entropia_ajustada": res_O,
        "Q_densidade_ajustada": res_Q,
        "betas_ajuste_O": betas_O,
        "betas_ajuste_Q": betas_Q
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
    Otimizado para calibração e controle positivo.
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


def _rodar_bloco_analise(mat_O, mat_Q, n_nulo, rng, prefixo="primaria_ajustada"):
    """
    Roda os testes H1, H2, H3 e Exploratório para um par de matrizes O e Q.
    """
    N = mat_O.shape[0]
    Z_O = est.copula_transform(mat_O)
    Z_Q = est.copula_transform(mat_Q)

    # 1. H1: Total Correlation
    tc_obs_O = est.tc_gaussian_copula(mat_O)
    tc_obs_Q = est.tc_gaussian_copula(mat_Q)

    # 2. H2: MI Homólogos > Não-Homólogos
    delta_mi_obs_O, mi_hom_obs_O, mi_nhom_obs_O = delta_mi(Z_O)
    delta_mi_obs_Q, mi_hom_obs_Q, mi_nhom_obs_Q = delta_mi(Z_Q)

    # 3. H3: Concordância Inter-Indivíduo (TC pontual O x Q)
    tc_pts_obs_O = est.pointwise_tc_gaussian_copula(mat_O)
    tc_pts_obs_Q = est.pointwise_tc_gaussian_copula(mat_Q)
    r_spearman_obs, _ = stats.spearmanr(tc_pts_obs_O, tc_pts_obs_Q)
    r_pearson_obs, _ = stats.pearsonr(tc_pts_obs_O, tc_pts_obs_Q)

    # 4. Exploratório: O-information
    oinfo_obs_O = est.oinfo_gaussian_copula(mat_O)
    oinfo_obs_Q = est.oinfo_gaussian_copula(mat_Q)

    # Nulo
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

    p_h1_O = float((np.sum(tc_nulo_O >= tc_obs_O) + 1.0) / (n_nulo + 1.0))
    p_h1_Q = float((np.sum(tc_nulo_Q >= tc_obs_Q) + 1.0) / (n_nulo + 1.0))
    p_h2_O = float((np.sum(delta_mi_nulo_O >= delta_mi_obs_O) + 1.0) / (n_nulo + 1.0))
    p_h2_Q = float((np.sum(delta_mi_nulo_Q >= delta_mi_obs_Q) + 1.0) / (n_nulo + 1.0))
    p_h3_spearman = float((np.sum(r_spearman_nulo >= r_spearman_obs) + 1.0) / (n_nulo + 1.0))
    p_oinfo_O = float((np.sum(np.abs(oinfo_nulo_O) >= abs(oinfo_obs_O)) + 1.0) / (n_nulo + 1.0))
    p_oinfo_Q = float((np.sum(np.abs(oinfo_nulo_Q) >= abs(oinfo_obs_Q)) + 1.0) / (n_nulo + 1.0))

    return {
        "h1": {
            "familia_O": {
                "caracteristica": "o_entropia_ajustada" if "ajustada" in prefixo else "o_entropia_bruta",
                "tc_observado": round(float(tc_obs_O), 6),
                "nulo_media": round(float(np.mean(tc_nulo_O)), 6),
                "nulo_desvio": round(float(np.std(tc_nulo_O)), 6),
                "nulo_p95": round(float(np.percentile(tc_nulo_O, 95)), 6),
                "p_valor_empirico": round(p_h1_O, 6),
                "significativo_bonferroni": bool(p_h1_O < 0.01)
            },
            "familia_Q": {
                "caracteristica": "q_densidade_ajustada" if "ajustada" in prefixo else "q_densidade_bruta",
                "tc_observado": round(float(tc_obs_Q), 6),
                "nulo_media": round(float(np.mean(tc_nulo_Q)), 6),
                "nulo_desvio": round(float(np.std(tc_nulo_Q)), 6),
                "nulo_p95": round(float(np.percentile(tc_nulo_Q, 95)), 6),
                "p_valor_empirico": round(p_h1_Q, 6),
                "significativo_bonferroni": bool(p_h1_Q < 0.01)
            },
            "familia_C": {
                "status": "excluida_da_primaria_por_incompatibilidade_extrator",
                "motivo": "pcasys rejeitou imagens 288x309 px (< 512 px) e C' (Poincaré) obteve acurácia de 31% (< 80%)"
            }
        },
        "h2": {
            "familia_O": {
                "delta_mi_observado": round(float(delta_mi_obs_O), 6),
                "mi_homologos_medio": round(float(mi_hom_obs_O), 6),
                "mi_nao_homologos_medio": round(float(mi_nhom_obs_O), 6),
                "nulo_delta_mi_media": round(float(np.mean(delta_mi_nulo_O)), 6),
                "nulo_delta_mi_desvio": round(float(np.std(delta_mi_nulo_O)), 6),
                "p_valor_empirico": round(p_h2_O, 6),
                "significativo_bonferroni": bool(p_h2_O < 0.01)
            },
            "familia_Q": {
                "delta_mi_observado": round(float(delta_mi_obs_Q), 6),
                "mi_homologos_medio": round(float(mi_hom_obs_Q), 6),
                "mi_nao_homologos_medio": round(float(mi_nhom_obs_Q), 6),
                "nulo_delta_mi_media": round(float(np.mean(delta_mi_nulo_Q)), 6),
                "nulo_delta_mi_desvio": round(float(np.std(delta_mi_nulo_Q)), 6),
                "p_valor_empirico": round(p_h2_Q, 6),
                "significativo_bonferroni": bool(p_h2_Q < 0.01)
            }
        },
        "h3": {
            "correlacao_spearman_observada": round(float(r_spearman_obs), 6),
            "correlacao_pearson_observada": round(float(r_pearson_obs), 6),
            "nulo_spearman_media": round(float(np.mean(r_spearman_nulo)), 6),
            "nulo_spearman_desvio": round(float(np.std(r_spearman_nulo)), 6),
            "p_valor_empirico": round(p_h3_spearman, 6),
            "significativo_bonferroni": bool(p_h3_spearman < 0.01)
        },
        "exploratorio": {
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
            }
        },
        "internals": {
            "tc_obs_O": tc_obs_O, "tc_nulo_O": tc_nulo_O,
            "tc_obs_Q": tc_obs_Q, "tc_nulo_Q": tc_nulo_Q,
            "delta_mi_obs_O": delta_mi_obs_O, "delta_mi_nulo_O": delta_mi_nulo_O,
            "delta_mi_obs_Q": delta_mi_obs_Q, "delta_mi_nulo_Q": delta_mi_nulo_Q,
            "r_spearman_obs": r_spearman_obs, "r_spearman_nulo": r_spearman_nulo,
            "tc_pts_obs_O": tc_pts_obs_O, "tc_pts_obs_Q": tc_pts_obs_Q,
            "p_h1_O": p_h1_O, "p_h1_Q": p_h1_Q, "p_h2_O": p_h2_O, "p_h2_Q": p_h2_Q, "p_h3_spearman": p_h3_spearman
        }
    }


def executar_analise(dados, n_nulo=10000, seed=42, modo_desbloqueado=False):
    """
    Executa a análise estatística pré-registrada:
    1) Primária: Resíduos ajustados por regressão linear em qualidade e sexo (D4b, D5, D7).
    2) Secundária: Variáveis brutas.
    3) D4a: TC da qualidade_media x nulo (sob cegamento).
    """
    rng = np.random.default_rng(seed)
    N = dados["N"]

    if not modo_desbloqueado:
        print("[INFO] MODO CEGO ATIVO: Embaralhando resíduos e variáveis antes da análise...")
        mat_O_prim = embaralhar_matriz(dados["O_entropia_ajustada"], rng)
        mat_Q_prim = embaralhar_matriz(dados["Q_densidade_ajustada"], rng)
        mat_O_sec = embaralhar_matriz(dados["O_entropia"], rng)
        mat_Q_sec = embaralhar_matriz(dados["Q_densidade"], rng)
        mat_qualidade_cega = embaralhar_matriz(dados["qualidade_media"], rng)
    else:
        print("[ALERTA] MODO DESBLOQUEADO: Usando dados reais...")
        mat_O_prim = dados["O_entropia_ajustada"].copy()
        mat_Q_prim = dados["Q_densidade_ajustada"].copy()
        mat_O_sec = dados["O_entropia"].copy()
        mat_Q_sec = dados["Q_densidade"].copy()
        mat_qualidade_cega = dados["qualidade_media"].copy()

    # --------------------------------------------------------------------------
    # 1. ANÁLISE PRIMÁRIA (Ajustada por qualidade e sexo)
    # --------------------------------------------------------------------------
    print(f"[INFO] Executando Análise Primária Ajustada ({n_nulo} permutações)...")
    res_prim = _rodar_bloco_analise(mat_O_prim, mat_Q_prim, n_nulo, rng, prefixo="primaria_ajustada")

    # --------------------------------------------------------------------------
    # 2. ANÁLISE SECUNDÁRIA (Bruta / Não-ajustada)
    # --------------------------------------------------------------------------
    print(f"[INFO] Executando Análise Secundária Bruta ({n_nulo} permutações)...")
    res_sec = _rodar_bloco_analise(mat_O_sec, mat_Q_sec, n_nulo, rng, prefixo="secundaria_bruta")

    # --------------------------------------------------------------------------
    # 3. TESTE D4a: Total Correlation da própria qualidade_media x Nulo
    # --------------------------------------------------------------------------
    print(f"[INFO] Executando Teste D4a (TC da Qualidade Média x Nulo)...")
    tc_qual_obs = est.tc_gaussian_copula(mat_qualidade_cega)
    Z_qual = est.copula_transform(mat_qualidade_cega)
    tc_qual_nulo = np.zeros(n_nulo, dtype=float)
    Z_perm_qual = np.zeros_like(Z_qual)
    for b in range(n_nulo):
        for j in range(10):
            Z_perm_qual[:, j] = Z_qual[rng.permutation(N), j]
        R_qual = np.corrcoef(Z_perm_qual, rowvar=False)
        s_qual, ld_qual = np.linalg.slogdet(R_qual)
        tc_qual_nulo[b] = max(0.0, -0.5 * ld_qual) if s_qual > 0 else 0.0

    p_qual = float((np.sum(tc_qual_nulo >= tc_qual_obs) + 1.0) / (n_nulo + 1.0))

    res_d4a = {
        "status": "sucesso",
        "teste_d4a_efeito_sessao": {
            "descricao": "Total Correlation de qualidade_media nos 10 dedos x distribuicao nula",
            "modo_execucao": "desbloqueado" if modo_desbloqueado else "cego_embaralhado",
            "tc_qualidade_observado": round(float(tc_qual_obs), 6),
            "nulo_media": round(float(np.mean(tc_qual_nulo)), 6),
            "nulo_desvio": round(float(np.std(tc_qual_nulo)), 6),
            "nulo_p95": round(float(np.percentile(tc_qual_nulo, 95)), 6),
            "p_valor_empirico": round(p_qual, 6),
            "confirmacao_efeito_sessao_alpha_005": bool(p_qual < 0.05),
            "nota_metodologica": "No modo cego opera sobre dados embaralhados; no desbloqueio, p < 0.05 confirma efeito de sessão."
        }
    }

    # TC pontual x Sexo para análise exploratória
    sexos = dados["sexos"]
    mask_m = sexos == "M"
    mask_f = sexos == "F"
    tc_pts_prim_O = res_prim["internals"]["tc_pts_obs_O"]
    tc_pts_prim_Q = res_prim["internals"]["tc_pts_obs_Q"]
    u_O, p_sexo_O = stats.mannwhitneyu(tc_pts_prim_O[mask_m], tc_pts_prim_O[mask_f])
    u_Q, p_sexo_Q = stats.mannwhitneyu(tc_pts_prim_Q[mask_m], tc_pts_prim_Q[mask_f])

    res_exploratorio = {
        "status": "sucesso",
        "primaria_ajustada": res_prim["exploratorio"],
        "secundaria_bruta": res_sec["exploratorio"],
        "tc_pontual_vs_sexo": {
            "familia_O_ajustada": {
                "mediana_M": round(float(np.median(tc_pts_prim_O[mask_m])), 6),
                "mediana_F": round(float(np.median(tc_pts_prim_O[mask_f])), 6),
                "mann_whitney_p_valor": round(float(p_sexo_O), 6)
            },
            "familia_Q_ajustada": {
                "mediana_M": round(float(np.median(tc_pts_prim_Q[mask_m])), 6),
                "mediana_F": round(float(np.median(tc_pts_prim_Q[mask_f])), 6),
                "mann_whitney_p_valor": round(float(p_sexo_Q), 6)
            }
        }
    }

    # Montar JSONs consolidados
    res_h1 = {
        "status": "sucesso",
        "analise_primaria_ajustada": res_prim["h1"],
        "analise_secundaria_bruta": res_sec["h1"]
    }

    res_h2 = {
        "status": "sucesso",
        "analise_primaria_ajustada": res_prim["h2"],
        "analise_secundaria_bruta": res_sec["h2"]
    }

    res_h3 = {
        "status": "sucesso",
        "analise_primaria_ajustada": res_prim["h3"],
        "analise_secundaria_bruta": res_sec["h3"]
    }

    res_resumo = {
        "modo_execucao": "desbloqueado" if modo_desbloqueado else "cego_embaralhado",
        "n_pessoas_analisadas": N,
        "n_permutações_nulo": n_nulo,
        "semente": seed,
        "correcao_multiplas_comparacoes": {
            "metodo": "Bonferroni",
            "alpha_global": 0.05,
            "n_testes_primarios": 5,
            "alpha_por_teste": 0.01,
            "testes_primarios": [
                "H1_familia_O (Entropia Orientação ajustada)",
                "H1_familia_Q (Densidade Minúcias ajustada)",
                "H2_familia_O (MI Homólogos > Não-Homólogos ajustada)",
                "H2_familia_Q (MI Homólogos > Não-Homólogos ajustada)",
                "H3_concordancia (Spearman TC pontual O x Q ajustada)"
            ]
        },
        "H1_TC_populacao": res_h1,
        "H2_MI_homologos": res_h2,
        "H3_concordancia_inter_familia": res_h3,
        "D4a_TC_qualidade": res_d4a["teste_d4a_efeito_sessao"],
        "Exploratorio": res_exploratorio,
        "sanidade_pipeline": "Verificado por calibração sob o nulo (scripts/54-calibracao.py; out/estudo02/cego/calibracao.json conforme Desvio D3)"
    }

    # Salvar JSONs
    os.makedirs(OUT_CEGO, exist_ok=True)
    with open(os.path.join(OUT_CEGO, "h1_resultado.json"), "w", encoding="utf-8") as f:
        json.dump(res_h1, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_CEGO, "h2_resultado.json"), "w", encoding="utf-8") as f:
        json.dump(res_h2, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_CEGO, "h3_resultado.json"), "w", encoding="utf-8") as f:
        json.dump(res_h3, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_CEGO, "d4a_qualidade_resultado.json"), "w", encoding="utf-8") as f:
        json.dump(res_d4a, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_CEGO, "exploratorio.json"), "w", encoding="utf-8") as f:
        json.dump(res_exploratorio, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_CEGO, "resumo_analise_cega.json"), "w", encoding="utf-8") as f:
        json.dump(res_resumo, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------------------------
    # FIGURA RESUMO (out/estudo02/cego/figura_tc_nulo.png)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(13, 15), dpi=150)
    int_p = res_prim["internals"]

    # 1. H1 Família O Primária
    ax = axes[0, 0]
    ax.hist(int_p["tc_nulo_O"], bins=40, density=True, color="steelblue", alpha=0.7, edgecolor="black", label="Distribuição Nula")
    ax.axvline(int_p["tc_obs_O"], color="crimson", linewidth=2, linestyle="--", label=f"TC Obs ({int_p['tc_obs_O']:.4f})")
    ax.set_title(f"H1 Primária — Família O (Orientação Ajustada)\np = {int_p['p_h1_O']:.4f} (Bonferroni: {'SIM' if int_p['p_h1_O'] < 0.01 else 'NÃO'})")
    ax.set_xlabel("Total Correlation (nats)")
    ax.set_ylabel("Densidade")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # 2. H1 Família Q Primária
    ax = axes[0, 1]
    ax.hist(int_p["tc_nulo_Q"], bins=40, density=True, color="seagreen", alpha=0.7, edgecolor="black", label="Distribuição Nula")
    ax.axvline(int_p["tc_obs_Q"], color="crimson", linewidth=2, linestyle="--", label=f"TC Obs ({int_p['tc_obs_Q']:.4f})")
    ax.set_title(f"H1 Primária — Família Q (Densidade Ajustada)\np = {int_p['p_h1_Q']:.4f} (Bonferroni: {'SIM' if int_p['p_h1_Q'] < 0.01 else 'NÃO'})")
    ax.set_xlabel("Total Correlation (nats)")
    ax.set_ylabel("Densidade")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # 3. H2 Família O Primária
    ax = axes[1, 0]
    ax.hist(int_p["delta_mi_nulo_O"], bins=40, density=True, color="darkorange", alpha=0.7, edgecolor="black", label="Distribuição Nula")
    ax.axvline(int_p["delta_mi_obs_O"], color="crimson", linewidth=2, linestyle="--", label=f"ΔMI Obs ({int_p['delta_mi_obs_O']:.4f})")
    ax.set_title(f"H2 Primária — ΔMI Homólogos (Família O Ajustada)\np = {int_p['p_h2_O']:.4f} (Bonferroni: {'SIM' if int_p['p_h2_O'] < 0.01 else 'NÃO'})")
    ax.set_xlabel("ΔMI (nats)")
    ax.set_ylabel("Densidade")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # 4. H2 Família Q Primária
    ax = axes[1, 1]
    ax.hist(int_p["delta_mi_nulo_Q"], bins=40, density=True, color="coral", alpha=0.7, edgecolor="black", label="Distribuição Nula")
    ax.axvline(int_p["delta_mi_obs_Q"], color="crimson", linewidth=2, linestyle="--", label=f"ΔMI Obs ({int_p['delta_mi_obs_Q']:.4f})")
    ax.set_title(f"H2 Primária — ΔMI Homólogos (Família Q Ajustada)\np = {int_p['p_h2_Q']:.4f} (Bonferroni: {'SIM' if int_p['p_h2_Q'] < 0.01 else 'NÃO'})")
    ax.set_xlabel("ΔMI (nats)")
    ax.set_ylabel("Densidade")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # 5. H3 Primária (TC Pontual O x Q)
    ax = axes[2, 0]
    ax.scatter(tc_pts_prim_O, tc_pts_prim_Q, alpha=0.4, color="purple", edgecolors="none", s=25)
    ax.set_title(f"H3 Primária — Concordância Inter-Família (O × Q Ajustadas)\nSpearman r = {int_p['r_spearman_obs']:.4f} (p = {int_p['p_h3_spearman']:.4f})")
    ax.set_xlabel("TC Pontual — O Ajustada (nats)")
    ax.set_ylabel("TC Pontual — Q Ajustada (nats)")
    ax.grid(True, linestyle=":", alpha=0.6)

    # 6. D4a Qualidade Média TC x Nulo
    ax = axes[2, 1]
    ax.hist(tc_qual_nulo, bins=40, density=True, color="teal", alpha=0.7, edgecolor="black", label="Distribuição Nula")
    ax.axvline(tc_qual_obs, color="crimson", linewidth=2, linestyle="--", label=f"TC Obs ({tc_qual_obs:.4f})")
    ax.set_title(f"D4a — TC da Qualidade Média dos 10 Dedos\np = {p_qual:.4f} (Efeito de Sessão)")
    ax.set_xlabel("Total Correlation (nats)")
    ax.set_ylabel("Densidade")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_path = os.path.join(OUT_CEGO, "figura_tc_nulo.png")
    plt.savefig(fig_path)
    plt.close(fig)

    print(f"[INFO] Figura gerada: {fig_path}")
    print(f"[INFO] Resumo salvo em: {os.path.join(OUT_CEGO, 'resumo_analise_cega.json')}")

    return res_resumo


def main():
    parser = argparse.ArgumentParser(description="Análise estatística cega do Estudo 02 (Desvios D4-D7)")
    parser.add_argument("--n-nulo", type=int, default=10000, help="Número de permutações para a distribuição nula (padrão: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Semente pseudo-aleatória fixa (padrão: 42)")
    parser.add_argument("--desbloquear", action="store_true", help="ATENÇÃO: Flag de desbloqueio para dados reais (NÃO USAR NA ETAPA 1)")
    args = parser.parse_args()

    dados = carregar_dados()
    print(f"[INFO] Total de sujeitos completos analisados: {dados['N']}")

    resumo = executar_analise(
        dados,
        n_nulo=args.n_nulo,
        seed=args.seed,
        modo_desbloqueado=args.desbloquear
    )

    h1_p = resumo["H1_TC_populacao"]["analise_primaria_ajustada"]
    h2_p = resumo["H2_MI_homologos"]["analise_primaria_ajustada"]
    h3_p = resumo["H3_concordancia_inter_familia"]["analise_primaria_ajustada"]
    d4a = resumo["D4a_TC_qualidade"]

    print()
    print("==================================================================")
    print("RESUMO DA ANÁLISE CEGA — ESTUDO 02 (Desvios D4b, D5, D7)")
    print("==================================================================")
    print(f"Modo: {resumo['modo_execucao']}")
    print(f"Sujeitos analisados: {resumo['n_pessoas_analisadas']}")
    print(f"Permutações nulas: {resumo['n_permutações_nulo']}")
    print(f"Critério Bonferroni: alpha = {resumo['correcao_multiplas_comparacoes']['alpha_por_teste']} ({resumo['correcao_multiplas_comparacoes']['n_testes_primarios']} testes)")
    print()
    print("ANÁLISE PRIMÁRIA AJUSTADA (Qualidade + Sexo):")
    print(f"  H1 Família O: TC = {h1_p['familia_O']['tc_observado']:.6f} | p = {h1_p['familia_O']['p_valor_empirico']:.4f} | Bonf (p<0.01): {h1_p['familia_O']['significativo_bonferroni']}")
    print(f"  H1 Família Q: TC = {h1_p['familia_Q']['tc_observado']:.6f} | p = {h1_p['familia_Q']['p_valor_empirico']:.4f} | Bonf (p<0.01): {h1_p['familia_Q']['significativo_bonferroni']}")
    print(f"  H2 Família O: ΔMI = {h2_p['familia_O']['delta_mi_observado']:.6f} | p = {h2_p['familia_O']['p_valor_empirico']:.4f} | Bonf (p<0.01): {h2_p['familia_O']['significativo_bonferroni']}")
    print(f"  H2 Família Q: ΔMI = {h2_p['familia_Q']['delta_mi_observado']:.6f} | p = {h2_p['familia_Q']['p_valor_empirico']:.4f} | Bonf (p<0.01): {h2_p['familia_Q']['significativo_bonferroni']}")
    print(f"  H3 Concordância: Spearman r = {h3_p['correlacao_spearman_observada']:.6f} | p = {h3_p['p_valor_empirico']:.4f} | Bonf (p<0.01): {h3_p['significativo_bonferroni']}")
    print()
    print("TESTE D4a (TC da Qualidade nos 10 Dedos):")
    print(f"  TC Qualidade: {d4a['tc_qualidade_observado']:.6f} | p = {d4a['p_valor_empirico']:.4f}")
    print()
    print(f"Sanidade do pipeline: {resumo['sanidade_pipeline']}")
    print("==================================================================")


if __name__ == "__main__":
    main()
