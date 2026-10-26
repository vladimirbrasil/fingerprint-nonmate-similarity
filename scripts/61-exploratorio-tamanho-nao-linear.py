#!/usr/bin/env python3
# scripts/61-exploratorio-tamanho-nao-linear.py — EXPLORATÓRIO (feito DEPOIS do desbloqueio de 28/09/2026).
# Pergunta: o achado de simetria entre dedos homólogos no fluxo das cristas (H2: o_entropia)
# sobrevive a controles NÃO LINEARES do tamanho do dedo (medido por log(area_util))?
#
# Versões reportadas:
# 1) o_entropia bruta (não ajustada);
# 2) Resíduo linear de o_entropia regredida em log(area_util) por dedo;
# 3) Resíduo em spline cúbica natural de log(area_util) com 4 graus de liberdade por dedo
#    (restricted cubic spline com nós nos quantis 5%, 35%, 65%, 95%);
# 4) Resíduo por estratos (10 decis de log(area_util) por dedo, subtraindo a média do decil);
# 5) Resíduo em spline cúbica natural + qualidade_media + contraste lineares por dedo.
#
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

def rcs_basis(x, knots):
    """
    Base de spline cúbica natural / restrita (Harrell / Stone-Koo).
    Com K nós, gera x e K-2 termos não lineares (total de K-1 colunas; com intercepto = K graus de liberdade).
    Para nós nos percentis [5, 35, 65, 95] (K=4), resulta em 3 colunas + 1 intercepto = 4 graus de liberdade.
    """
    knots = np.asarray(knots, dtype=float)
    K = len(knots)
    t = knots
    h = (t[-1] - t[0]) ** 2
    cols = [x]
    def pos_cube(u):
        return np.maximum(u, 0.0) ** 3
    for j in range(K - 2):
        term1 = pos_cube(x - t[j])
        term2 = pos_cube(x - t[K - 2]) * (t[K - 1] - t[j]) / (t[K - 1] - t[K - 2])
        term3 = pos_cube(x - t[K - 1]) * (t[K - 2] - t[j]) / (t[K - 1] - t[K - 2])
        col = (term1 - term2 + term3) / h
        cols.append(col)
    return np.column_stack(cols)

def main():
    d = an.carregar_dados()
    N = int(d["N"])
    rng = np.random.default_rng(59)

    mat_o_entropia = d["O_entropia"]
    mat_log_area = np.log(d["area_util"])
    mat_qm = d["qualidade_media"]
    mat_ct = d["contraste"]

    # 1. o_entropia bruta (não ajustada)
    mat_v1 = mat_o_entropia

    # 2. Resíduo linear em log(area_util)
    mat_v2 = np.zeros_like(mat_o_entropia)
    r2_linear = {}
    for j in range(10):
        y = mat_o_entropia[:, j]
        x = mat_log_area[:, j]
        X = np.column_stack([np.ones(N), x])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        mat_v2[:, j] = y - pred
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_linear[j] = float(1.0 - np.sum((y - pred) ** 2) / ss_tot)

    # 3. Resíduo em spline cúbica natural de log(area_util) (4 graus de liberdade)
    mat_v3 = np.zeros_like(mat_o_entropia)
    r2_spline = {}
    for j in range(10):
        y = mat_o_entropia[:, j]
        x = mat_log_area[:, j]
        knots = np.percentile(x, [5, 35, 65, 95])
        B = rcs_basis(x, knots)
        X = np.column_stack([np.ones(N), B])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        mat_v3[:, j] = y - pred
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_spline[j] = float(1.0 - np.sum((y - pred) ** 2) / ss_tot)

    # 4. Resíduo por estratos (10 decis de log(area_util))
    mat_v4 = np.zeros_like(mat_o_entropia)
    for j in range(10):
        y = mat_o_entropia[:, j]
        x = mat_log_area[:, j]
        qs = np.percentile(x, np.linspace(0, 100, 11))
        qs[0] -= 1e-8
        qs[-1] += 1e-8
        bins = np.digitize(x, qs) - 1
        for b in range(10):
            mask = (bins == b)
            if np.any(mask):
                mat_v4[mask, j] = y[mask] - np.mean(y[mask])

    # 5. Spline cúbica natural + qualidade_media + contraste lineares
    mat_v5 = np.zeros_like(mat_o_entropia)
    for j in range(10):
        y = mat_o_entropia[:, j]
        x = mat_log_area[:, j]
        knots = np.percentile(x, [5, 35, 65, 95])
        B = rcs_basis(x, knots)
        X = np.column_stack([np.ones(N), B, mat_qm[:, j], mat_ct[:, j]])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        mat_v5[:, j] = y - pred

    versoes = {
        "1_o_entropia_bruta": {
            "descricao": "1. o_entropia bruta (sem ajuste)",
            "mat": mat_v1
        },
        "2_residuo_linear_log_area": {
            "descricao": "2. Resíduo linear em log(area_util)",
            "mat": mat_v2
        },
        "3_residuo_spline_log_area": {
            "descricao": "3. Resíduo spline cúbica natural log(area_util) (4 gl)",
            "mat": mat_v3
        },
        "4_residuo_decis_log_area": {
            "descricao": "4. Resíduo estratificado em 10 decis de log(area_util)",
            "mat": mat_v4
        },
        "5_residuo_spline_log_area_qm_ct": {
            "descricao": "5. Resíduo spline cúbica + qualidade + contraste",
            "mat": mat_v5
        }
    }

    nomes_dedos = getattr(an, "NOMES_DEDOS", [f"Dedo_{i}" for i in range(10)])

    out = {
        "ATENCAO": "EXPLORATÓRIO — pós-desbloqueio",
        "N": N,
        "n_nulo": 10000,
        "semente": 59,
        "r2_explicado_por_dedo": {
            nomes_dedos[j]: {
                "r2_linear": r2_linear[j],
                "r2_spline": r2_spline[j]
            } for j in range(10)
        },
        "resultados": {}
    }

    for k, info in versoes.items():
        res = resumo(info["mat"], rng)
        res["descricao"] = info["descricao"]
        out["resultados"][k] = res

    os.makedirs("out/estudo02", exist_ok=True)
    out_path = "out/estudo02/61-tamanho-nao-linear.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("=" * 115)
    print("ESTUDO 02 — EXPLORATÓRIO: CONTROLE NÃO LINEAR DO TAMANHO DO DEDO (H2 / o_entropia)")
    print("=" * 115)
    print(f"N = {N} sujeitos | 10.000 permutações nulas por posição de dedo | Seed = 59")
    print("-" * 115)
    header = f"{'Versão':<52} | {'Simetria':<10} | {'p (Simetria)':<12} | {'Efeito Mão':<11} | {'p (Mão)':<8}"
    print(header)
    print("-" * 115)

    for k, v in out["resultados"].items():
        linha = (
            f"{v['descricao']:<52} | "
            f"{v['simetria_livre_de_sessao']:<10.4f} | "
            f"{v['p_simetria_livre_de_sessao']:<12.4e} | "
            f"{v['efeito_mao']:<11.4f} | "
            f"{v['p_efeito_mao']:<8.4e}"
        )
        print(linha)
    print("-" * 115)
    print()
    print("R² da regressão de o_entropia em log(area_util) por dedo:")
    print("-" * 55)
    print(f"{'Dedo':<18} | {'R² Linear':<14} | {'R² Spline (4 gl)':<16}")
    print("-" * 55)
    for j in range(10):
        d_nome = nomes_dedos[j]
        print(f"{d_nome:<18} | {r2_linear[j]:<14.4f} | {r2_spline[j]:<16.4f}")
    print("-" * 55)
    print()
    print(f"Arquivo salvo em: {out_path}")
    print("=" * 115)

if __name__ == "__main__":
    main()
