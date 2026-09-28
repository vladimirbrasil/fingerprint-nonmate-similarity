#!/usr/bin/env python3
# ==============================================================================
# scripts/54-calibracao.py
# Calibração do pipeline de testes estatísticos sob a hipótese nula (Desvio D3).
#
# REGRA DE CEGAMENTO E CALIBRAÇÃO:
# - Avalia 200 conjuntos de dados embaralhados independentes (cada um gerado por
#   permutação independente por posição de dedo, destruindo o vínculo do mesmo corpo).
# - Para cada conjunto, roda o pipeline de testes nulos com --n-nulo 199 permutações
#   para as hipóteses H1 (O, Q), H2 (O, Q) e H3.
# - Sob o nulo verdadeiro, os p-valores devem seguir distribuição ~Uniforme(0, 1).
# - Para alpha = 0.05 e N = 200 conjuntos, a contagem esperada de p < 0.05 é 10 (5%).
# - O critério de aprovação (PASSA) exige que a fração observada de p < 0.05 caia
#   dentro do intervalo binomial exato de 95% em torno de 0.05 (4 a 16 rejeições em 200).
#
# Saídas:
# - out/estudo02/cego/calibracao.json
# - out/estudo02/cego/calibracao_pvalores.png
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
FEATURES_CSV = os.path.join(ROOT, "out", "estudo02", "features.csv")
DUPLICATAS_JSON = os.path.join(ROOT, "results", "socofing_duplicatas_exatas.json")
OUT_CEGO = os.path.join(ROOT, "out", "estudo02", "cego")

# Importar 52-analise-cega.py
spec52 = importlib.util.spec_from_file_location(
    "analise_cega",
    os.path.join(ROOT, "scripts", "52-analise-cega.py")
)
cega = importlib.util.module_from_spec(spec52)
spec52.loader.exec_module(cega)


def avaliar_um_conjunto(seed_idx, seed_val, mat_O_raw, mat_Q_raw, n_nulo):
    """
    Gera um conjunto embaralhado independente com a semente documentada
    e executa o pipeline de testes sob o nulo.
    """
    rng = np.random.default_rng(seed_val)
    mat_O_shuf = cega.embaralhar_matriz(mat_O_raw, rng)
    mat_Q_shuf = cega.embaralhar_matriz(mat_Q_raw, rng)

    res = cega.avaliar_hipoteses_rapido(mat_O_shuf, mat_Q_shuf, n_nulo=n_nulo, rng=rng)
    res["seed"] = seed_val
    res["idx"] = seed_idx
    return res


def main():
    parser = argparse.ArgumentParser(description="Calibração do pipeline sob o nulo (Desvio D3)")
    parser.add_argument("--n-conjuntos", type=int, default=200, help="Número de conjuntos embaralhados independentes (padrão: 200)")
    parser.add_argument("--n-nulo", type=int, default=199, help="Número de permutações nulas por conjunto (padrão: 199)")
    parser.add_argument("--seed-base", type=int, default=1000, help="Semente base documentada (padrão: 1000)")
    parser.add_argument("--n-jobs", type=int, default=3, help="Número de processos paralelos (padrão: 3)")
    args = parser.parse_args()

    dados = cega.carregar_dados(FEATURES_CSV, DUPLICATAS_JSON)
    mat_O_raw = dados["O_entropia"]
    mat_Q_raw = dados["Q_densidade"]
    N_pessoas = dados["N"]

    print("==================================================================")
    print("CALIBRAÇÃO ESTATÍSTICA SOB O NULO — ESTUDO 02 (Desvio D3)")
    print("==================================================================")
    print(f"Sujeitos completos analisados: {N_pessoas}")
    print(f"Conjuntos independentes embaralhados: {args.n_conjuntos}")
    print(f"Permutações nulas por conjunto: {args.n_nulo}")
    print(f"Sementes documentadas: {args.seed_base} a {args.seed_base + args.n_conjuntos - 1}")
    print(f"Paralelização: {args.n_jobs} núcleos (joblib)")
    print("==================================================================")

    # Sementes distintas documentadas
    sementes = [args.seed_base + i for i in range(args.n_conjuntos)]

    # Execução paralela com nice
    print("[INFO] Executando calibração em paralelo...")
    resultados = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(avaliar_um_conjunto)(
            i, sementes[i], mat_O_raw, mat_Q_raw, args.n_nulo
        ) for i in range(args.n_conjuntos)
    )

    # Extrair arrays de p-valores por hipótese
    p_h1_O = np.array([r["p_h1_O"] for r in resultados])
    p_h1_Q = np.array([r["p_h1_Q"] for r in resultados])
    p_h2_O = np.array([r["p_h2_O"] for r in resultados])
    p_h2_Q = np.array([r["p_h2_Q"] for r in resultados])
    p_h3 = np.array([r["p_h3"] for r in resultados])

    # Intervalo binomial exato de 95% para n=200 sob H0 (p=0.05)
    # scipy.stats.binom.interval(0.95, n, p0)
    low_k, high_k = stats.binom.interval(0.95, args.n_conjuntos, 0.05)
    frac_min = float(low_k / args.n_conjuntos)
    frac_max = float(high_k / args.n_conjuntos)

    hipoteses_dict = {
        "H1_familia_O": ("H1: Família O (Entropia Orientação)", p_h1_O),
        "H1_familia_Q": ("H1: Família Q (Densidade Minúcias)", p_h1_Q),
        "H2_familia_O": ("H2: Família O (MI Homólogos > Não-Homólogos)", p_h2_O),
        "H2_familia_Q": ("H2: Família Q (MI Homólogos > Não-Homólogos)", p_h2_Q),
        "H3_concordancia": ("H3: Concordância Inter-Família (O x Q)", p_h3),
    }

    analise_hipoteses = {}
    todas_passaram = True

    for chave, (rotulo, p_arr) in hipoteses_dict.items():
        k_rejeicoes = int(np.sum(p_arr < 0.05))
        frac_obs = float(k_rejeicoes / args.n_conjuntos)
        passou = bool(frac_min <= frac_obs <= frac_max)
        if not passou:
            todas_passaram = False

        # Intervalo de confiança exato de Clopper-Pearson para a proporção observada
        btest = stats.binomtest(k_rejeicoes, args.n_conjuntos, 0.05)
        prop_ci = btest.proportion_ci(confidence_level=0.95)

        analise_hipoteses[chave] = {
            "rotulo": rotulo,
            "k_rejeicoes_alpha_005": k_rejeicoes,
            "fracao_rejeicoes_observada": round(frac_obs, 4),
            "intervalo_binomial_95_esperado": {
                "k_min": int(low_k),
                "k_max": int(high_k),
                "frac_min": round(frac_min, 4),
                "frac_max": round(frac_max, 4)
            },
            "ic_observado_95_clopper_pearson": [round(float(prop_ci.low), 4), round(float(prop_ci.high), 4)],
            "status": "PASSA" if passou else "FALHA",
            "p_valores": [round(float(x), 4) for x in p_arr]
        }

    res_calibracao = {
        "status_geral": "PASSA" if todas_passaram else "FALHA",
        "n_conjuntos_embaralhados": args.n_conjuntos,
        "n_permutações_nulo_por_conjunto": args.n_nulo,
        "sementes_documentadas": {
            "semente_inicial": args.seed_base,
            "semente_final": args.seed_base + args.n_conjuntos - 1
        },
        "criterio_calibracao": {
            "alpha_nominal": 0.05,
            "k_esperado": int(round(args.n_conjuntos * 0.05)),
            "k_min_aceitavel": int(low_k),
            "k_max_aceitavel": int(high_k),
            "frac_min_aceitavel": round(frac_min, 4),
            "frac_max_aceitavel": round(frac_max, 4),
            "descricao": "Fração de p < 0.05 deve cair dentro do intervalo binomial exato de 95% [0.02, 0.08]"
        },
        "resultados_por_hipotese": analise_hipoteses
    }

    # Salvar JSON de calibração
    os.makedirs(OUT_CEGO, exist_ok=True)
    json_path = os.path.join(OUT_CEGO, "calibracao.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res_calibracao, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Calibração salva em: {json_path}")

    # Gerar histograma dos p-valores por hipótese
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), dpi=150)
    axes_flat = axes.flatten()

    cores = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    for idx, (chave, (rotulo, p_arr)) in enumerate(hipoteses_dict.items()):
        ax = axes_flat[idx]
        dados_h = analise_hipoteses[chave]
        k_rej = dados_h["k_rejeicoes_alpha_005"]
        f_obs = dados_h["fracao_rejeicoes_observada"]
        st = dados_h["status"]

        ax.hist(p_arr, bins=10, range=(0.0, 1.0), density=True, color=cores[idx], alpha=0.7, edgecolor="black")
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Uniforme Esperada (densidade = 1.0)")
        ax.axvline(0.05, color="black", linestyle=":", linewidth=1.2, label=r"$\alpha = 0.05$")

        ax.set_title(f"{rotulo}\np < 0.05: {k_rej}/{args.n_conjuntos} ({f_obs*100:.1f}%) — [{st}]", fontsize=10, fontweight="bold")
        ax.set_xlabel("p-valor empírico")
        ax.set_ylabel("Densidade")
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")

    # O sexto subplot exibe o resumo textual
    ax_last = axes_flat[5]
    ax_last.axis("off")
    texto_resumo = (
        f"CALIBRAÇÃO ESTATÍSTICA (Desvio D3)\n\n"
        f"• Conjuntos nulos independentes: {args.n_conjuntos}\n"
        f"• Permutações por conjunto: {args.n_nulo}\n"
        f"• Intervalo 95% binomial exato: [{frac_min:.2f}, {frac_max:.2f}] (k: {int(low_k)}..{int(high_k)})\n\n"
        f"Resultados:\n"
    )
    for chave, (rotulo, _) in hipoteses_dict.items():
        dh = analise_hipoteses[chave]
        texto_resumo += f"  - {chave}: {dh['k_rejeicoes_alpha_005']}/{args.n_conjuntos} ({dh['fracao_rejeicoes_observada']*100:.1f}%) → {dh['status']}\n"

    texto_resumo += f"\nStatus Geral: {res_calibracao['status_geral']}"
    ax_last.text(0.1, 0.5, texto_resumo, fontsize=11, family="monospace", va="center")

    plt.tight_layout()
    fig_path = os.path.join(OUT_CEGO, "calibracao_pvalores.png")
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"[INFO] Figura gerada: {fig_path}")

    # Exibir resumo no console
    print()
    print("==================================================================")
    print("RESULTADO DA CALIBRAÇÃO")
    print("==================================================================")
    print(f"Status Geral: {res_calibracao['status_geral']}")
    print(f"Critério 95% binomial: k em [{int(low_k)}, {int(high_k)}] em {args.n_conjuntos} (fração em [{frac_min:.3f}, {frac_max:.3f}])")
    print()
    for chave, dh in analise_hipoteses.items():
        print(f"  {dh['rotulo']:50s}: k = {dh['k_rejeicoes_alpha_005']:2d}/200 ({dh['fracao_rejeicoes_observada']*100:4.1f}%) | IC95% = {dh['ic_observado_95_clopper_pearson']} | [{dh['status']}]")
    print("==================================================================")


if __name__ == "__main__":
    main()
