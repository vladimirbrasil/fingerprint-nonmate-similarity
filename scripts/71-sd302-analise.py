#!/usr/bin/env python3
# ==============================================================================
# scripts/71-sd302-analise.py
# Análise pré-registrada do Estudo 03 (NIST SD302 — Aparelho e Operador).
#
# Pré-registro: estudo 03, datado por commit antes do desbloqueio (hipóteses e regra de leitura abaixo).
#
# Hipóteses primárias (Bonferroni alpha = 0.05 / 3 ~= 0.0167):
# - H1 (réplica): S(U, U) > nulo.
# - H2 (troca de operador): S_cruz = media(S(U, V), S(V, U)) > nulo.
#   Razao r2 = S_cruz / media(S(U, U), S(V, V)) com IC bootstrap 95%.
#   Regra: limite inferior > 0.5 -> "corpo"; limite superior < 0.5 -> "captura";
#   contendo 0.5 -> "inconclusivo".
# - H3 (troca de aparelho): mediana de S(c1, c2) sobre todos os pares ordenados
#   c1 != c2 de capturas roladas (U, V, A..H) com >= 150 pessoas validas > nulo.
#   Razao r3 = mediana S(c1!=c2) / mediana S(c, c) com IC bootstrap 95%.
#
# Confiabilidade:
# - R(c1, c2) = media de MI do mesmo dedo (10 dedos) entre c1 e c2.
#
# REGRA DE CEGAMENTO:
# - Por padrão, este script opera ESTRITAMENTE em modo CEGO:
#   embaralha as pessoas independentemente por (captura, dedo) antes de qualquer conta
#   e grava em out/estudo03/cego/.
# - Somente com a flag explícita --desbloquear usa os dados reais (grava em out/estudo03/real/).
#
# Reaproveita estimadores e funções de:
# - scripts/51-estimadores.py (copula_transform)
# - scripts/58-exploratorio-maos.py (classes, mi_de_R)
# ==============================================================================

import argparse
import csv
import json
import importlib.util
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_CSV = os.path.join(ROOT, "out", "estudo03", "features.csv")
ERRATA_PATH = os.path.join(ROOT, "data", "nist-sd302", "ERRATA_SD302.txt")

# Carregar módulos 51 e 58 dinamicamente
def carregar_modulo(nome, rel_path):
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(nome, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod

est51 = carregar_modulo("est51", os.path.join("scripts", "51-estimadores.py"))
maos58 = carregar_modulo("maos58", os.path.join("scripts", "58-exploratorio-maos.py"))
feat70 = carregar_modulo("feat70", os.path.join("scripts", "70-sd302-features.py"))

# FRGPs na ordem obrigatória: L1..L5, R1..R5 -> [6, 7, 8, 9, 10, 1, 2, 3, 4, 5]
ORDEM_FRGP = [6, 7, 8, 9, 10, 1, 2, 3, 4, 5]


def carregar_matrizes_por_captura(features_csv=FEATURES_CSV, errata_path=ERRATA_PATH, cegar=True, seed_cego=42):
    """
    Lê features.csv e organiza matrizes pessoas x 10 dedos por captura.
    Aplica filtros pré-registrados:
    - Exclui pessoas com excluido_errata == 1 (ou citadas na errata)
    - Exclui pessoas sem os 10 dedos válidos na captura
    - Ordena colunas rigorosamente em: [L1..L5, R1..R5] (FRGP [6..10, 1..5])

    Se cegar=True, embaralha as pessoas independentemente por (captura, dedo).
    """
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Arquivo features não encontrado: {features_csv}")

    errata = feat70.carregar_errata(errata_path)

    dados_por_cap = {}
    with open(features_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cap = row["captura"]
            pessoa = row["pessoa"]
            tipo = row["tipo"]
            excluido_err = int(row.get("excluido_errata", 0))

            if excluido_err == 1 or (pessoa, cap.upper()) in errata:
                continue

            frgp = int(row["frgp"])
            if frgp not in ORDEM_FRGP:
                continue

            o_ent = row.get("o_entropia", "")
            if not o_ent:
                continue

            dados_por_cap.setdefault(cap, {}).setdefault(pessoa, {
                "tipo": tipo,
                "dedos": {}
            })
            dados_por_cap[cap][pessoa]["dedos"][frgp] = row

    capturas = {}
    for cap, pessoas_dict in dados_por_cap.items():
        pessoas_validas = []
        for p, pdata in pessoas_dict.items():
            if len(pdata["dedos"]) == 10 and all(f in pdata["dedos"] for f in ORDEM_FRGP):
                pessoas_validas.append(p)
        pessoas_validas.sort()

        if not pessoas_validas:
            continue

        N = len(pessoas_validas)
        tipo_cap = pessoas_dict[pessoas_validas[0]]["tipo"]

        mat_o = np.zeros((N, 10), dtype=float)
        mat_q = np.zeros((N, 10), dtype=float)
        mat_area = np.zeros((N, 10), dtype=float)
        mat_qm = np.zeros((N, 10), dtype=float)
        mat_ct = np.zeros((N, 10), dtype=float)

        for i, p in enumerate(pessoas_validas):
            for j, f in enumerate(ORDEM_FRGP):
                r = pessoas_dict[p]["dedos"][f]
                mat_o[i, j] = float(r["o_entropia"])
                mat_q[i, j] = float(r["q_densidade"]) if r.get("q_densidade") else 0.0
                mat_area[i, j] = float(r["q_area_util"]) if r.get("q_area_util") else 1.0
                mat_qm[i, j] = float(r["qualidade_media"]) if r.get("qualidade_media") else 0.0
                mat_ct[i, j] = float(r["contraste"]) if r.get("contraste") else 0.0

        capturas[cap] = {
            "captura": cap,
            "tipo": tipo_cap,
            "pessoas": np.array(pessoas_validas),
            "N": N,
            "o_entropia": mat_o,
            "q_densidade": mat_q,
            "area_util": mat_area,
            "qualidade_media": mat_qm,
            "contraste": mat_ct
        }

    if cegar:
        rng = np.random.default_rng(seed_cego)
        for cap, cdata in capturas.items():
            N = cdata["N"]
            for col in range(10):
                perm = rng.permutation(N)
                cdata["o_entropia"][:, col] = cdata["o_entropia"][perm, col]
                cdata["q_densidade"][:, col] = cdata["q_densidade"][perm, col]
                cdata["area_util"][:, col] = cdata["area_util"][perm, col]
                cdata["qualidade_media"][:, col] = cdata["qualidade_media"][perm, col]
                cdata["contraste"][:, col] = cdata["contraste"][perm, col]

    return capturas


def calcular_S_e_classes(mat_10):
    """
    mat_10: (N, 10) onde colunas 0..4 são mão esquerda e 5..9 são mão direita.
    Aplica copula_transform -> correlação -> mi_de_R -> classes.
    Retorna S = homologos - cruzados_nao_homologos.
    """
    Z = est51.copula_transform(mat_10)
    R = np.corrcoef(Z, rowvar=False)
    MI = maos58.mi_de_R(R)
    homol, cruz_nao, mesma = maos58.classes(MI)
    S = float(homol - cruz_nao)
    return S, float(homol), float(cruz_nao), float(mesma), Z


def calcular_S_par(cdata1, cdata2, var="o_entropia"):
    """
    Calcula S(c1, c2) usando mão esquerda de c1 e mão direita de c2 para pessoas comuns.
    Retorna (S, homol, cruz_nao, N_intersecao, Z_comb).
    """
    p1 = cdata1["pessoas"]
    p2 = cdata2["pessoas"]

    comuns = np.intersect1d(p1, p2)
    N_com = len(comuns)
    if N_com < 5:
        return np.nan, np.nan, np.nan, N_com, None

    idx1 = np.searchsorted(p1, comuns)
    idx2 = np.searchsorted(p2, comuns)

    mat_L = cdata1[var][idx1, :5]  # colunas 0..4 = L1..L5
    mat_R = cdata2[var][idx2, 5:]  # colunas 5..9 = R1..R5

    mat_comb = np.column_stack([mat_L, mat_R])
    S, homol, cruz_nao, _, Z = calcular_S_e_classes(mat_comb)
    return S, homol, cruz_nao, N_com, Z


def calcular_R_par(cdata1, cdata2, var="o_entropia"):
    """
    R(c1, c2): média da MI entre o mesmo dedo (todas as 10 posições) em c1 e c2.
    """
    p1 = cdata1["pessoas"]
    p2 = cdata2["pessoas"]
    comuns = np.intersect1d(p1, p2)
    N_com = len(comuns)
    if N_com < 5:
        return np.nan, N_com

    idx1 = np.searchsorted(p1, comuns)
    idx2 = np.searchsorted(p2, comuns)

    m1 = cdata1[var][idx1, :]
    m2 = cdata2[var][idx2, :]

    mis = []
    for j in range(10):
        pair = np.column_stack([m1[:, j], m2[:, j]])
        Z = est51.copula_transform(pair)
        r = np.corrcoef(Z, rowvar=False)[0, 1]
        r2 = np.clip(r ** 2, 0, 0.999999)
        mi = -0.5 * np.log(1 - r2)
        mis.append(mi)

    return float(np.mean(mis)), N_com


def calcular_nulo_S(Z, n_nulo=10000, seed=303):
    """
    Nulo: permutar pessoas independentemente por coluna (seed 303).
    p = (k + 1) / (N + 1) para teste unilateral à direita (S >= S_obs).
    """
    rng = np.random.default_rng(seed)
    N, K = Z.shape
    Zp = np.empty_like(Z)
    s_nulo = np.empty(n_nulo, dtype=float)

    for i in range(n_nulo):
        for j in range(K):
            Zp[:, j] = Z[rng.permutation(N), j]
        R = np.corrcoef(Zp, rowvar=False)
        MI = maos58.mi_de_R(R)
        homol, cruz_nao, _ = maos58.classes(MI)
        s_nulo[i] = homol - cruz_nao

    return s_nulo


def bootstrap_razao_H2(cdata_U, cdata_V, n_boot=2000, seed=304, var="o_entropia"):
    """
    Bootstrap de pessoas (seed 304) para IC 95% da razão r2:
    r2 = S_cruz / media(S(U,U), S(V,V))
    onde S_cruz = media(S(U,V), S(V,U)).
    """
    pU = cdata_U["pessoas"]
    pV = cdata_V["pessoas"]
    comuns = np.intersect1d(pU, pV)
    N_com = len(comuns)
    if N_com < 10:
        return np.nan, (np.nan, np.nan)

    idxU = np.searchsorted(pU, comuns)
    idxV = np.searchsorted(pV, comuns)

    matU = cdata_U[var][idxU, :]
    matV = cdata_V[var][idxV, :]

    rng = np.random.default_rng(seed)
    r2_dist = []

    for _ in range(n_boot):
        boot_idx = rng.choice(N_com, size=N_com, replace=True)
        bU = matU[boot_idx, :]
        bV = matV[boot_idx, :]

        # S(U, U)
        s_uu, _, _, _, _ = calcular_S_e_classes(bU)
        # S(V, V)
        s_vv, _, _, _, _ = calcular_S_e_classes(bV)
        # S(U, V)
        s_uv, _, _, _, _ = calcular_S_e_classes(np.column_stack([bU[:, :5], bV[:, 5:]]))
        # S(V, U)
        s_vu, _, _, _, _ = calcular_S_e_classes(np.column_stack([bV[:, :5], bU[:, 5:]]))

        s_cruz = 0.5 * (s_uv + s_vu)
        s_auto = 0.5 * (s_uu + s_vv)
        if abs(s_auto) > 1e-9:
            r2_dist.append(s_cruz / s_auto)

    if not r2_dist:
        return np.nan, (np.nan, np.nan)

    ic_baixo = float(np.percentile(r2_dist, 2.5))
    ic_alto = float(np.percentile(r2_dist, 97.5))
    return float(np.median(r2_dist)), (ic_baixo, ic_alto)


def regredir_residuos(Y_mat, X_cols_list):
    """Regride linearmente coluna a coluna de Y_mat nas covariáveis correspondentes."""
    N, K = Y_mat.shape
    Res = np.zeros((N, K), dtype=float)
    for j in range(K):
        covs = [cov[:, j] for cov in X_cols_list]
        X = np.column_stack([np.ones(N)] + covs)
        beta, _, _, _ = np.linalg.lstsq(X, Y_mat[:, j], rcond=None)
        Res[:, j] = Y_mat[:, j] - X @ beta
    return Res


def executar_pipeline(capturas, out_dir):
    """Executa todos os testes pré-registrados (H1, H2, H3, Confiabilidade, Exploratório)."""
    os.makedirs(out_dir, exist_ok=True)
    resumo = {}

    # Adicionar versões ajustadas a cada captura
    for cap, cdata in capturas.items():
        log_area = np.log(np.maximum(cdata["area_util"], 1.0))
        qm = cdata["qualidade_media"]
        ct = cdata["contraste"]
        cdata["o_entropia_ajustada"] = regredir_residuos(cdata["o_entropia"], [log_area, qm, ct])
        cdata["q_densidade_ajustada"] = regredir_residuos(cdata["q_densidade"], [log_area, qm, ct])

    # ----------------------------------------------------
    # H1 (réplica): S(U, U)
    # ----------------------------------------------------
    print("[INFO] Executando H1: S(U, U)...")
    if "U" in capturas:
        S_uu, hom_uu, cruz_uu, _, Z_uu = calcular_S_e_classes(capturas["U"]["o_entropia"])
        N_u = int(capturas["U"]["N"])
        if Z_uu is not None and N_u >= 10:
            nulo_h1 = calcular_nulo_S(Z_uu, n_nulo=10000, seed=303)
            p_h1 = float((np.sum(nulo_h1 >= S_uu) + 1) / (len(nulo_h1) + 1))
        else:
            p_h1 = np.nan
        resumo["H1"] = {
            "S_UU": S_uu,
            "homologos": hom_uu,
            "cruzados_nao_homologos": cruz_uu,
            "N": N_u,
            "p_valor": p_h1,
            "passou_bonferroni": bool(p_h1 < 0.05 / 3) if not np.isnan(p_h1) else False
        }
    else:
        resumo["H1"] = {"status": "captura U nao encontrada"}

    # ----------------------------------------------------
    # H2 (troca de operador): S_cruz = média(S(U, V), S(V, U))
    # ----------------------------------------------------
    print("[INFO] Executando H2: S(U, V) e S(V, U)...")
    if "U" in capturas and "V" in capturas:
        S_uv, hom_uv, cruz_uv, N_uv, Z_uv = calcular_S_par(capturas["U"], capturas["V"], "o_entropia")
        S_vu, hom_vu, cruz_vu, N_vu, Z_vu = calcular_S_par(capturas["V"], capturas["U"], "o_entropia")
        S_uu, _, _, N_uu, _ = calcular_S_par(capturas["U"], capturas["U"], "o_entropia")
        S_vv, _, _, N_vv, _ = calcular_S_par(capturas["V"], capturas["V"], "o_entropia")

        S_cruz = 0.5 * (S_uv + S_vu)
        S_auto = 0.5 * (S_uu + S_vv)
        r2 = float(S_cruz / S_auto) if abs(S_auto) > 1e-9 else np.nan

        # Nulo para H2 (permutar pessoas independentemente por dedo em U e V e recalcular S_cruz)
        pU = capturas["U"]["pessoas"]
        pV = capturas["V"]["pessoas"]
        comuns = np.intersect1d(pU, pV)
        idxU = np.searchsorted(pU, comuns)
        idxV = np.searchsorted(pV, comuns)
        matU = capturas["U"]["o_entropia"][idxU, :]
        matV = capturas["V"]["o_entropia"][idxV, :]

        rng303 = np.random.default_rng(303)
        n_nulo_h2 = 10000
        nulo_h2 = np.empty(n_nulo_h2, dtype=float)
        bU = np.empty_like(matU)
        bV = np.empty_like(matV)

        for i in range(n_nulo_h2):
            for col in range(10):
                bU[:, col] = matU[rng303.permutation(len(comuns)), col]
                bV[:, col] = matV[rng303.permutation(len(comuns)), col]
            s_uv_p, _, _, _, _ = calcular_S_e_classes(np.column_stack([bU[:, :5], bV[:, 5:]]))
            s_vu_p, _, _, _, _ = calcular_S_e_classes(np.column_stack([bV[:, :5], bU[:, 5:]]))
            nulo_h2[i] = 0.5 * (s_uv_p + s_vu_p)

        p_h2 = float((np.sum(nulo_h2 >= S_cruz) + 1) / (n_nulo_h2 + 1)) if len(comuns) >= 5 else np.nan

        # Bootstrap de pessoas para r2
        r2_med, (ic_inf_r2, ic_sup_r2) = bootstrap_razao_H2(capturas["U"], capturas["V"], n_boot=2000, seed=304)

        leitura_h2 = "inconclusivo"
        if ic_inf_r2 > 0.5:
            leitura_h2 = "a maior parte da simetria sobrevive a troca de operador (corpo)"
        elif ic_sup_r2 < 0.5:
            leitura_h2 = "a simetria e principalmente da captura"

        # Confiabilidade R(U, V)
        R_uv, _ = calcular_R_par(capturas["U"], capturas["V"], "o_entropia")

        resumo["H2"] = {
            "S_cruz": S_cruz,
            "S_UV": S_uv,
            "S_VU": S_vu,
            "S_UU": S_uu,
            "S_VV": S_vv,
            "N_intersecao": len(comuns),
            "p_valor": p_h2,
            "r2_observado": r2,
            "r2_bootstrap_mediana": r2_med,
            "r2_ic95": [ic_inf_r2, ic_sup_r2],
            "leitura": leitura_h2,
            "R_UV_confiabilidade": R_uv
        }
    else:
        resumo["H2"] = {"status": "U ou V nao encontradas"}

    # ----------------------------------------------------
    # H3 (troca de aparelho): pares roll (U, V, A..H)
    # ----------------------------------------------------
    print("[INFO] Executando H3: aparelhos roll (U, V, A..H)...")
    caps_roll = [c for c in ["U", "V", "A", "B", "C", "D", "E", "F", "G", "H"] if c in capturas]
    pares_distintos = []
    s_auto_list = []

    # Identificar pares ordenados com >= 150 pessoas (ou >= 5 em teste com limite)
    # O pré-registro define >= 150 pessoas
    limite_pessoas = 150  # fixo no pré-registro; com --limite de teste o H3 fica vazio, e está certo

    caps_auto = []
    for c in caps_roll:
        s_cc, _, _, n_cc, _ = calcular_S_par(capturas[c], capturas[c], "o_entropia")
        if n_cc >= limite_pessoas:
            s_auto_list.append(s_cc)
            caps_auto.append(c)

    mat_pares = {}
    for c1 in caps_roll:
        for c2 in caps_roll:
            if c1 != c2:
                s_val, _, _, n_com, _ = calcular_S_par(capturas[c1], capturas[c2], "o_entropia")
                if n_com >= limite_pessoas:
                    pares_distintos.append((c1, c2, s_val, n_com))
                    mat_pares[(c1, c2)] = s_val

    if pares_distintos:
        s_cruzados_vals = [p[2] for p in pares_distintos]
        mediana_s_cruz = float(np.median(s_cruzados_vals))
        mediana_s_auto = float(np.median(s_auto_list)) if s_auto_list else np.nan
        r3 = float(mediana_s_cruz / mediana_s_auto) if abs(mediana_s_auto) > 1e-9 else np.nan

        # Nulo para H3: permutar pessoas independentemente por dedo em cada captura
        # e recalcular a mediana dos pares válidos
        rng303_h3 = np.random.default_rng(303)
        n_nulo_h3 = 10000
        nulo_h3 = np.empty(n_nulo_h3, dtype=float)

        # Preparar cópias para permutação
        caps_mat_o = {c: np.array(capturas[c]["o_entropia"]) for c in caps_roll}

        for i in range(n_nulo_h3):
            # Permutar colunas de cada captura
            shuf_caps = {}
            for c in caps_roll:
                m_orig = caps_mat_o[c]
                shuf_m = np.empty_like(m_orig)
                for col in range(10):
                    shuf_m[:, col] = m_orig[rng303_h3.permutation(len(m_orig)), col]
                shuf_caps[c] = shuf_m

            s_perm_vals = []
            for c1, c2, _, _ in pares_distintos:
                p1 = capturas[c1]["pessoas"]
                p2 = capturas[c2]["pessoas"]
                com = np.intersect1d(p1, p2)
                idx1 = np.searchsorted(p1, com)
                idx2 = np.searchsorted(p2, com)
                m_comb = np.column_stack([shuf_caps[c1][idx1, :5], shuf_caps[c2][idx2, 5:]])
                s_p, _, _, _, _ = calcular_S_e_classes(m_comb)
                s_perm_vals.append(s_p)

            nulo_h3[i] = np.median(s_perm_vals)

        p_h3 = float((np.sum(nulo_h3 >= mediana_s_cruz) + 1) / (n_nulo_h3 + 1))

        # Bootstrap para IC 95% de r3
        rng304_h3 = np.random.default_rng(304)
        n_boot_h3 = 2000
        boot_r3 = []
        # Bootstrap CONJUNTO: um sorteio de pessoas por iteração, usado em todas as capturas e pares
        # (os S de pares diferentes compartilham pessoas; sortear separado subestimaria a variância).
        universo = np.unique(np.concatenate([capturas[c]["pessoas"] for c in caps_roll]))
        linhas = {c: {p: i for i, p in enumerate(capturas[c]["pessoas"])} for c in caps_roll}
        for _ in range(n_boot_h3):
            sorteio = universo[rng304_h3.choice(len(universo), size=len(universo), replace=True)]
            boot_s_cruz = []
            boot_s_auto = []
            for c in caps_auto:
                b_idx = [linhas[c][p] for p in sorteio if p in linhas[c]]
                s_b, _, _, _, _ = calcular_S_e_classes(capturas[c]["o_entropia"][b_idx, :])
                boot_s_auto.append(s_b)
            for c1, c2, _, _ in pares_distintos:
                ps = [p for p in sorteio if p in linhas[c1] and p in linhas[c2]]
                idx1 = [linhas[c1][p] for p in ps]
                idx2 = [linhas[c2][p] for p in ps]
                m_comb = np.column_stack([capturas[c1]["o_entropia"][idx1, :5], capturas[c2]["o_entropia"][idx2, 5:]])
                s_b_pair, _, _, _, _ = calcular_S_e_classes(m_comb)
                boot_s_cruz.append(s_b_pair)

            med_cruz = np.median(boot_s_cruz)
            med_auto = np.median(boot_s_auto)
            if abs(med_auto) > 1e-9:
                boot_r3.append(med_cruz / med_auto)

        if boot_r3:
            ic_inf_r3 = float(np.percentile(boot_r3, 2.5))
            ic_sup_r3 = float(np.percentile(boot_r3, 97.5))
            r3_med = float(np.median(boot_r3))
        else:
            ic_inf_r3, ic_sup_r3, r3_med = np.nan, np.nan, np.nan

        leitura_h3 = "inconclusivo"
        if ic_inf_r3 > 0.5:
            leitura_h3 = "a maior parte da simetria sobrevive a troca de aparelho (corpo)"
        elif ic_sup_r3 < 0.5:
            leitura_h3 = "a simetria e principalmente da captura"

        R_pares = [calcular_R_par(capturas[c1], capturas[c2], "o_entropia")[0] for c1, c2, _, _ in pares_distintos]
        resumo["H3"] = {
            "n_pares_validos": len(pares_distintos),
            "S_por_par": {f"{c1}x{c2}": float(sv) for c1, c2, sv, _ in pares_distintos},
            "mediana_R_confiabilidade": float(np.median(R_pares)),
            "mediana_S_cruz": mediana_s_cruz,
            "mediana_S_auto": mediana_s_auto,
            "r3_observado": r3,
            "r3_bootstrap_mediana": r3_med,
            "r3_ic95": [ic_inf_r3, ic_sup_r3],
            "p_valor": p_h3,
            "leitura": leitura_h3
        }
    else:
        resumo["H3"] = {"status": f"nenhum par atingiu N >= {limite_pessoas}"}

    # ----------------------------------------------------
    # Análise Exploratória
    # ----------------------------------------------------
    print("[INFO] Executando análises exploratórias (ajustada, q_densidade, slap/plain)...")
    exploratorio = {}

    # 1. H1 e H2 com o_entropia ajustada
    if "U" in capturas and "V" in capturas:
        s_uu_adj, _, _, _, _ = calcular_S_par(capturas["U"], capturas["U"], "o_entropia_ajustada")
        s_vv_adj, _, _, _, _ = calcular_S_par(capturas["V"], capturas["V"], "o_entropia_ajustada")
        s_uv_adj, _, _, _, _ = calcular_S_par(capturas["U"], capturas["V"], "o_entropia_ajustada")
        s_vu_adj, _, _, _, _ = calcular_S_par(capturas["V"], capturas["U"], "o_entropia_ajustada")
        exploratorio["o_entropia_ajustada_H2"] = {
            "S_UU": s_uu_adj,
            "S_VV": s_vv_adj,
            "S_UV": s_uv_adj,
            "S_VU": s_vu_adj,
            "S_cruz": 0.5 * (s_uv_adj + s_vu_adj)
        }

    # 2. H1 e H2 com q_densidade
    if "U" in capturas and "V" in capturas:
        s_uu_q, _, _, _, _ = calcular_S_par(capturas["U"], capturas["U"], "q_densidade")
        s_vv_q, _, _, _, _ = calcular_S_par(capturas["V"], capturas["V"], "q_densidade")
        s_uv_q, _, _, _, _ = calcular_S_par(capturas["U"], capturas["V"], "q_densidade")
        s_vu_q, _, _, _, _ = calcular_S_par(capturas["V"], capturas["U"], "q_densidade")
        exploratorio["q_densidade_H2"] = {
            "S_UU": s_uu_q,
            "S_VV": s_vv_q,
            "S_UV": s_uv_q,
            "S_VU": s_vu_q,
            "S_cruz": 0.5 * (s_uv_q + s_vu_q)
        }

    # 3. Cruzamento roll x plain / slap
    slap_caps = [c for c in ["R", "S"] if c in capturas]
    plain_caps = [c for c in ["K", "L", "M", "P"] if c in capturas]
    cruz_tipos = {}
    if "U" in capturas:
        for c_other in slap_caps + plain_caps:
            s_val, _, _, n_com, _ = calcular_S_par(capturas["U"], capturas[c_other], "o_entropia")
            cruz_tipos[f"U_x_{c_other}"] = {"S": s_val, "N": n_com, "tipo2": capturas[c_other]["tipo"]}
    exploratorio["cruzamento_tipos"] = cruz_tipos

    resumo["exploratorio"] = exploratorio

    out_file = os.path.join(out_dir, "resultado.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Análise concluída. Resultados salvos em: {out_file}")
    return resumo


def main():
    parser = argparse.ArgumentParser(description="Análise pré-registrada Estudo 03 (NIST SD302)")
    parser.add_argument("--desbloquear", action="store_true", help="Desbloqueia os dados reais (NÃO USE no cegamento)")
    parser.add_argument("--features", default=FEATURES_CSV, help="Caminho para features.csv")
    parser.add_argument("--errata", default=ERRATA_PATH, help="Caminho para ERRATA_SD302.txt")
    parser.add_argument("--excluir", default="", help="Capturas a remover, ex.: D,H (EXPLORATÓRIO: D e H são binárias)")
    args = parser.parse_args()

    cegar = not args.desbloquear
    modo = "cego" if cegar else "real"
    excluir = [c for c in args.excluir.split(",") if c]
    if excluir:
        modo += "_sem_" + "".join(excluir)
    out_dir = os.path.join(ROOT, "out", "estudo03", modo)

    print(f"==================================================")
    print(f" Executando Estudo 03 em MODO: {modo.upper()}")
    print(f"==================================================")

    capturas = carregar_matrizes_por_captura(
        features_csv=args.features,
        errata_path=args.errata,
        cegar=cegar,
        seed_cego=42
    )
    for c in excluir:
        capturas.pop(c, None)
    print(f"[INFO] Capturas carregadas: {list(capturas.keys())}")
    for cap, cdata in capturas.items():
        print(f"  - {cap} ({cdata['tipo']}): {cdata['N']} pessoas válidas com 10 dedos")

    resumo = executar_pipeline(capturas, out_dir)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
