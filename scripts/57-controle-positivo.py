#!/usr/bin/env python3
# ==============================================================================
# scripts/57-controle-positivo.py
# Controle Positivo e Curva de Poder Estatístico (Desvio D6).
#
# REGRA DE CEGAMENTO E CONTROLE POSITIVO:
# - Opera sobre dados embaralhados sob a primária ajustada por qualidade e sexo.
# - Injeta um fator comum z = sqrt(rho)*f + sqrt(1 - rho)*z_original (na escala
#   da cópula gaussiana) para rho in {0, 0.01, 0.02, 0.05, 0.1, 0.2}.
# - 100 repetições independentes por valor de rho, com --n-nulo 199 permutações.
# - Para H1 (O, Q) e H3: fator comum compartilhado entre os 10 dedos.
# - Para H2 (O, Q): fator compartilhado estritamente entre pares homólogos.
# - Critério de detecção: p < 0.01 (Bonferroni para 5 testes primários, Desvio D7).
# - Paralelização com 3 núcleos via joblib com nice.
#
# Saídas:
# - out/estudo02/cego/poder.json
# - out/estudo02/cego/figura_curva_poder.png
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
from joblib import Parallel, delayed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_CEGO = os.path.join(ROOT, "out", "estudo02", "cego")

# Importar 51-estimadores.py e 52-analise-cega.py
spec51 = importlib.util.spec_from_file_location(
    "estimadores",
    os.path.join(ROOT, "scripts", "51-estimadores.py")
)
est = importlib.util.module_from_spec(spec51)
spec51.loader.exec_module(est)

spec52 = importlib.util.spec_from_file_location(
    "analise_cega",
    os.path.join(ROOT, "scripts", "52-analise-cega.py")
)
cega = importlib.util.module_from_spec(spec52)
spec52.loader.exec_module(cega)

PARES_HOMOLOGOS = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
PARES_TODOS = [(i, j) for i in range(10) for j in range(i + 1, 10)]
PARES_NAO_HOMOLOGOS = [p for p in PARES_TODOS if p not in PARES_HOMOLOGOS]


def delta_mi_from_R(R):
    mi_hom = []
    for i, j in PARES_HOMOLOGOS:
        rho = np.clip(R[i, j], -0.999999, 0.999999)
        mi_hom.append(-0.5 * np.log(1.0 - rho ** 2))

    mi_nao_hom = []
    for i, j in PARES_NAO_HOMOLOGOS:
        rho = np.clip(R[i, j], -0.999999, 0.999999)
        mi_nao_hom.append(-0.5 * np.log(1.0 - rho ** 2))

    return float(np.mean(mi_hom) - np.mean(mi_nao_hom))


def avaliar_uma_repeticao(rep_idx, rho, mat_O_base, mat_Q_base, n_nulo, seed):
    """
    Executa 1 repetição de injeção de sinal para um valor de rho dado.
    """
    rng = np.random.default_rng(seed)
    N = mat_O_base.shape[0]

    # 1. Embaralhar sob o nulo primário ajustado
    mat_O_shuf = cega.embaralhar_matriz(mat_O_base, rng)
    mat_Q_shuf = cega.embaralhar_matriz(mat_Q_base, rng)

    Z_O_orig = est.copula_transform(mat_O_shuf)
    Z_Q_orig = est.copula_transform(mat_Q_shuf)

    # 2. Injeção de Fator Comum Global para H1 e H3
    # z = sqrt(rho)*f + sqrt(1 - rho)*z_original
    sqrt_rho = np.sqrt(rho)
    sqrt_1_rho = np.sqrt(max(0.0, 1.0 - rho))

    f_global = rng.standard_normal(N)
    Z_O_global = np.zeros_like(Z_O_orig)
    Z_Q_global = np.zeros_like(Z_Q_orig)
    for j in range(10):
        Z_O_global[:, j] = sqrt_rho * f_global + sqrt_1_rho * Z_O_orig[:, j]
        Z_Q_global[:, j] = sqrt_rho * f_global + sqrt_1_rho * Z_Q_orig[:, j]

    # Re-padronizar cópula pós-injeção
    Z_O_global = est.copula_transform(Z_O_global)
    Z_Q_global = est.copula_transform(Z_Q_global)

    # Observados H1 e H3
    R_O_g = np.corrcoef(Z_O_global, rowvar=False)
    s_O_g, ld_O_g = np.linalg.slogdet(R_O_g)
    tc_obs_O = max(0.0, -0.5 * ld_O_g) if s_O_g > 0 else 0.0
    inv_R_O_g = np.linalg.inv(R_O_g) - np.eye(10)
    tc_pts_O = -0.5 * ld_O_g - 0.5 * np.sum((Z_O_global @ inv_R_O_g) * Z_O_global, axis=1)

    R_Q_g = np.corrcoef(Z_Q_global, rowvar=False)
    s_Q_g, ld_Q_g = np.linalg.slogdet(R_Q_g)
    tc_obs_Q = max(0.0, -0.5 * ld_Q_g) if s_Q_g > 0 else 0.0
    inv_R_Q_g = np.linalg.inv(R_Q_g) - np.eye(10)
    tc_pts_Q = -0.5 * ld_Q_g - 0.5 * np.sum((Z_Q_global @ inv_R_Q_g) * Z_Q_global, axis=1)

    r_s_obs, _ = stats.spearmanr(tc_pts_O, tc_pts_Q)

    # 3. Injeção de Fator Homólogo para H2
    # Para cada par homólogo (k, k+5), injeta fator g_k independente
    Z_O_hom = Z_O_orig.copy()
    Z_Q_hom = Z_Q_orig.copy()
    for k in range(5):
        g_O = rng.standard_normal(N)
        g_Q = rng.standard_normal(N)
        Z_O_hom[:, k] = sqrt_rho * g_O + sqrt_1_rho * Z_O_orig[:, k]
        Z_O_hom[:, k + 5] = sqrt_rho * g_O + sqrt_1_rho * Z_O_orig[:, k + 5]
        Z_Q_hom[:, k] = sqrt_rho * g_Q + sqrt_1_rho * Z_Q_orig[:, k]
        Z_Q_hom[:, k + 5] = sqrt_rho * g_Q + sqrt_1_rho * Z_Q_orig[:, k + 5]

    Z_O_hom = est.copula_transform(Z_O_hom)
    Z_Q_hom = est.copula_transform(Z_Q_hom)

    R_O_h = np.corrcoef(Z_O_hom, rowvar=False)
    dmi_obs_O = delta_mi_from_R(R_O_h)

    R_Q_h = np.corrcoef(Z_Q_hom, rowvar=False)
    dmi_obs_Q = delta_mi_from_R(R_Q_h)

    # 4. Avaliação Nula (Permutações)
    tc_nulo_O = np.zeros(n_nulo, dtype=float)
    tc_nulo_Q = np.zeros(n_nulo, dtype=float)
    rs_nulo = np.zeros(n_nulo, dtype=float)
    dmi_nulo_O = np.zeros(n_nulo, dtype=float)
    dmi_nulo_Q = np.zeros(n_nulo, dtype=float)

    Z_perm_O_g = np.zeros_like(Z_O_global)
    Z_perm_Q_g = np.zeros_like(Z_Q_global)
    Z_perm_O_h = np.zeros_like(Z_O_hom)
    Z_perm_Q_h = np.zeros_like(Z_Q_hom)

    for b in range(n_nulo):
        for j in range(10):
            p_Og = rng.permutation(N)
            p_Qg = rng.permutation(N)
            p_Oh = rng.permutation(N)
            p_Qh = rng.permutation(N)
            Z_perm_O_g[:, j] = Z_O_global[p_Og, j]
            Z_perm_Q_g[:, j] = Z_Q_global[p_Qg, j]
            Z_perm_O_h[:, j] = Z_O_hom[p_Oh, j]
            Z_perm_Q_h[:, j] = Z_Q_hom[p_Qh, j]

        # Nulo H1 O, H1 Q, H3
        R_b_Og = np.corrcoef(Z_perm_O_g, rowvar=False)
        s_b_Og, ld_b_Og = np.linalg.slogdet(R_b_Og)
        tc_nulo_O[b] = max(0.0, -0.5 * ld_b_Og) if s_b_Og > 0 else 0.0
        inv_R_b_Og = np.linalg.inv(R_b_Og) - np.eye(10)
        tc_pts_b_O = -0.5 * ld_b_Og - 0.5 * np.sum((Z_perm_O_g @ inv_R_b_Og) * Z_perm_O_g, axis=1)

        R_b_Qg = np.corrcoef(Z_perm_Q_g, rowvar=False)
        s_b_Qg, ld_b_Qg = np.linalg.slogdet(R_b_Qg)
        tc_nulo_Q[b] = max(0.0, -0.5 * ld_b_Qg) if s_b_Qg > 0 else 0.0
        inv_R_b_Qg = np.linalg.inv(R_b_Qg) - np.eye(10)
        tc_pts_b_Q = -0.5 * ld_b_Qg - 0.5 * np.sum((Z_perm_Q_g @ inv_R_b_Qg) * Z_perm_Q_g, axis=1)

        rs_b, _ = stats.spearmanr(tc_pts_b_O, tc_pts_b_Q)
        rs_nulo[b] = rs_b

        # Nulo H2 O, H2 Q
        R_b_Oh = np.corrcoef(Z_perm_O_h, rowvar=False)
        dmi_nulo_O[b] = delta_mi_from_R(R_b_Oh)

        R_b_Qh = np.corrcoef(Z_perm_Q_h, rowvar=False)
        dmi_nulo_Q[b] = delta_mi_from_R(R_b_Qh)

    p_h1_O = float((np.sum(tc_nulo_O >= tc_obs_O) + 1.0) / (n_nulo + 1.0))
    p_h1_Q = float((np.sum(tc_nulo_Q >= tc_obs_Q) + 1.0) / (n_nulo + 1.0))
    p_h2_O = float((np.sum(dmi_nulo_O >= dmi_obs_O) + 1.0) / (n_nulo + 1.0))
    p_h2_Q = float((np.sum(dmi_nulo_Q >= dmi_obs_Q) + 1.0) / (n_nulo + 1.0))
    p_h3 = float((np.sum(rs_nulo >= r_s_obs) + 1.0) / (n_nulo + 1.0))

    return {
        "rep_idx": rep_idx,
        "rho": rho,
        "p_h1_O": p_h1_O,
        "p_h1_Q": p_h1_Q,
        "p_h2_O": p_h2_O,
        "p_h2_Q": p_h2_Q,
        "p_h3": p_h3,
        "det_h1_O": bool(p_h1_O < 0.01),
        "det_h1_Q": bool(p_h1_Q < 0.01),
        "det_h2_O": bool(p_h2_O < 0.01),
        "det_h2_Q": bool(p_h2_Q < 0.01),
        "det_h3": bool(p_h3 < 0.01)
    }


def main():
    parser = argparse.ArgumentParser(description="Controle positivo e curva de poder estatístico (Desvio D6)")
    parser.add_argument("--n-rep", type=int, default=100, help="Repetições por valor de rho (padrão: 100)")
    parser.add_argument("--n-nulo", type=int, default=199, help="Permutações nulas por teste (padrão: 199)")
    parser.add_argument("--seed-base", type=int, default=42, help="Semente base pseudo-aleatória (padrão: 42)")
    parser.add_argument("--n-jobs", type=int, default=3, help="Processos paralelos (padrão: 3)")
    args = parser.parse_args()

    dados = cega.carregar_dados()
    mat_O_ajustada = dados["O_entropia_ajustada"]
    mat_Q_ajustada = dados["Q_densidade_ajustada"]
    N_pessoas = dados["N"]

    rhos = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]

    print("==================================================================")
    print("CONTROLE POSITIVO E CURVA DE PODER — ESTUDO 02 (Desvio D6)")
    print("==================================================================")
    print(f"Sujeitos completos analisados: {N_pessoas}")
    print(f"Valores de rho avaliados: {rhos}")
    print(f"Repetições por rho: {args.n_rep} (total: {len(rhos) * args.n_rep} simulações)")
    print(f"Permutações nulas por simulação: {args.n_nulo}")
    print(f"Critério de detecção: p < 0.01 (Bonferroni alpha = 0.01 para 5 testes primários)")
    print(f"Paralelização: {args.n_jobs} núcleos (joblib)")
    print("==================================================================")

    # Preparar tarefas
    tarefas = []
    task_idx = 0
    for r_idx, rho in enumerate(rhos):
        for rep in range(args.n_rep):
            seed = args.seed_base + r_idx * 10000 + rep * 7 + 1
            tarefas.append((task_idx, rep, rho, seed))
            task_idx += 1

    print(f"[INFO] Executando {len(tarefas)} simulações em paralelo ({args.n_jobs} núcleos)...")
    res_list = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(avaliar_uma_repeticao)(
            rep, rho, mat_O_ajustada, mat_Q_ajustada, args.n_nulo, seed
        ) for (t_idx, rep, rho, seed) in tarefas
    )

    # Agregar resultados por rho
    resultados_por_rho = {}
    poder_curvas = {
        "H1_familia_O": [],
        "H1_familia_Q": [],
        "H2_familia_O": [],
        "H2_familia_Q": [],
        "H3_concordancia": []
    }

    for rho in rhos:
        sub = [r for r in res_list if abs(r["rho"] - rho) < 1e-6]
        n_sub = len(sub)
        pod_h1_O = float(np.mean([r["det_h1_O"] for r in sub]))
        pod_h1_Q = float(np.mean([r["det_h1_Q"] for r in sub]))
        pod_h2_O = float(np.mean([r["det_h2_O"] for r in sub]))
        pod_h2_Q = float(np.mean([r["det_h2_Q"] for r in sub]))
        pod_h3 = float(np.mean([r["det_h3"] for r in sub]))

        poder_curvas["H1_familia_O"].append(pod_h1_O)
        poder_curvas["H1_familia_Q"].append(pod_h1_Q)
        poder_curvas["H2_familia_O"].append(pod_h2_O)
        poder_curvas["H2_familia_Q"].append(pod_h2_Q)
        poder_curvas["H3_concordancia"].append(pod_h3)

        resultados_por_rho[str(rho)] = {
            "rho": rho,
            "n_repeticoes": n_sub,
            "fracao_deteccao_p_menor_001": {
                "H1_familia_O": round(pod_h1_O, 4),
                "H1_familia_Q": round(pod_h1_Q, 4),
                "H2_familia_O": round(pod_h2_O, 4),
                "H2_familia_Q": round(pod_h2_Q, 4),
                "H3_concordancia": round(pod_h3, 4)
            },
            "p_valores_medios": {
                "H1_familia_O": round(float(np.mean([r["p_h1_O"] for r in sub])), 4),
                "H1_familia_Q": round(float(np.mean([r["p_h1_Q"] for r in sub])), 4),
                "H2_familia_O": round(float(np.mean([r["p_h2_O"] for r in sub])), 4),
                "H2_familia_Q": round(float(np.mean([r["p_h2_Q"] for r in sub])), 4),
                "H3_concordancia": round(float(np.mean([r["p_h3"] for r in sub])), 4)
            }
        }

    res_final = {
        "status": "sucesso",
        "descricao": "Curva de poder estatístico por injeção de sinal sintético em dados embaralhados (Desvio D6)",
        "configuracao": {
            "n_pessoas": N_pessoas,
            "valores_rho": rhos,
            "repeticoes_por_rho": args.n_rep,
            "n_permutações_nulo": args.n_nulo,
            "alpha_deteccao_bonferroni": 0.01,
            "semente_base": args.seed_base
        },
        "resultados_por_rho": resultados_por_rho,
        "resumo_poder": {
            "rho_minimo_poder_80_porcento": {
                "H1_familia_O": next((r for r in rhos if resultados_por_rho[str(r)]["fracao_deteccao_p_menor_001"]["H1_familia_O"] >= 0.8), "> 0.20"),
                "H1_familia_Q": next((r for r in rhos if resultados_por_rho[str(r)]["fracao_deteccao_p_menor_001"]["H1_familia_Q"] >= 0.8), "> 0.20"),
                "H2_familia_O": next((r for r in rhos if resultados_por_rho[str(r)]["fracao_deteccao_p_menor_001"]["H2_familia_O"] >= 0.8), "> 0.20"),
                "H2_familia_Q": next((r for r in rhos if resultados_por_rho[str(r)]["fracao_deteccao_p_menor_001"]["H2_familia_Q"] >= 0.8), "> 0.20"),
                "H3_concordancia": next((r for r in rhos if resultados_por_rho[str(r)]["fracao_deteccao_p_menor_001"]["H3_concordancia"] >= 0.8), "> 0.20")
            }
        }
    }

    # Salvar JSON
    os.makedirs(OUT_CEGO, exist_ok=True)
    json_path = os.path.join(OUT_CEGO, "poder.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res_final, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Resultados salvos em: {json_path}")

    # Gerar Figura da Curva de Poder
    plt.figure(figsize=(10, 6), dpi=150)
    cores = {
        "H1_familia_O": ("#1f77b4", "o-", "H1: TC Família O (Orientação)"),
        "H1_familia_Q": ("#2ca02c", "s-", "H1: TC Família Q (Densidade)"),
        "H2_familia_O": ("#ff7f0e", "^--", "H2: ΔMI Homólogos (Família O)"),
        "H2_familia_Q": ("#d62728", "v--", "H2: ΔMI Homólogos (Família Q)"),
        "H3_concordancia": ("#9467bd", "D-.", "H3: Concordância Inter-Família (O x Q)")
    }

    for chave, (cor, fmt, rotulo) in cores.items():
        plt.plot(rhos, poder_curvas[chave], fmt, color=cor, linewidth=2, markersize=7, label=rotulo)

    plt.axhline(0.01, color="gray", linestyle=":", label=r"Taxa de falso positivo ($\alpha = 0.01$)")
    plt.axhline(0.80, color="black", linestyle="--", alpha=0.7, label="Poder 80%")

    plt.title("Curva de Poder Estatístico sob Injeção de Sinal (Desvio D6)", fontsize=13, fontweight="bold")
    plt.xlabel(r"Força do Fator Comum Injetado ($\rho$)", fontsize=11)
    plt.ylabel(r"Fração de Detecção / Poder ($p < 0.01$)", fontsize=11)
    plt.xticks(rhos, [str(r) for r in rhos])
    plt.ylim(-0.02, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()

    fig_path = os.path.join(OUT_CEGO, "figura_curva_poder.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"[INFO] Figura salva em: {fig_path}")

    # Exibir tabela resumo
    print()
    print("==================================================================")
    print("RESUMO DA CURVA DE PODER ESTATÍSTICO (Desvio D6)")
    print("==================================================================")
    print(f"{'rho':<8} | {'H1 (O)':<10} | {'H1 (Q)':<10} | {'H2 (O)':<10} | {'H2 (Q)':<10} | {'H3 (O x Q)':<12}")
    print("-" * 70)
    for rho in rhos:
        d = resultados_por_rho[str(rho)]["fracao_deteccao_p_menor_001"]
        print(f"{rho:<8.2f} | {d['H1_familia_O']:<10.2f} | {d['H1_familia_Q']:<10.2f} | {d['H2_familia_O']:<10.2f} | {d['H2_familia_Q']:<10.2f} | {d['H3_concordancia']:<12.2f}")
    print("==================================================================")
    print(f"rho mínimo para poder >= 80%:")
    for k, v in res_final["resumo_poder"]["rho_minimo_poder_80_porcento"].items():
        print(f"  - {k}: rho = {v}")
    print("==================================================================")


if __name__ == "__main__":
    main()
