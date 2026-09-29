#!/usr/bin/env python3
"""
scripts/61-mesmo-corpo-sem-duplicatas.py

Reexecuta as análises do Estudo 01 na SOCOFing após exclusão rigorosa de duplicatas:
1) Grupos de imagens byte-idênticas (results/socofing_duplicatas_exatas.json).
2) Os 14 indivíduos confirmados como duplicatas / dedos trocados pelo exame pericial
   (out/estudo02/exploratorio/sensibilidade_duplicatas.json: 32, 71, 113, 167, 191,
    230, 301, 378, 481, 484, 528, 585, 596, 598).

Produz:
- Distribuição (mediana, p95, p99, máx) de escores 'mesmo corpo' vs 'corpos diferentes'
  (Antes e Depois).
- Teste de hipótese: teste com nulo por permutação de rótulos de pessoa e Mann-Whitney U,
  com múltiplos estimadores de tamanho de efeito (Cohen's d, Cliff's delta, AUC).
- Curva de crescimento do escore máximo com o tamanho da população (subamostras de sujeitos).
- Contagem e caracterização de pares com escore >= 40 antes e depois da exclusão.
- Salva: out/socofing/completo/sem_duplicatas.json
"""

import os
import csv
import json
import time
import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(ROOT, "out", "socofing", "completo")
INDEX_PATH = os.path.join(ROOT, "out", "socofing", "index.csv")
DUPS_EXATAS_PATH = os.path.join(ROOT, "results", "socofing_duplicatas_exatas.json")
SENSIBILIDADE_PATH = os.path.join(ROOT, "out", "estudo02", "exploratorio", "sensibilidade_duplicatas.json")
OUT_JSON = os.path.join(D, "sem_duplicatas.json")


def carregar_dados():
    rec = list(csv.DictReader(open(INDEX_PATH, encoding="utf-8")))
    n = len(rec)
    suj = np.array([int(r["sujeito"]) for r in rec])
    mao = np.array([r["mao"] for r in rec])
    dedo = np.array([r["dedo"] for r in rec])
    tags = np.array([f"{r['sujeito']}_{r['mao']}_{r['dedo']}" for r in rec])

    # NBIS: matriz salva no triângulo superior com -1 na parte inferior; simetrizar
    nb_raw = np.fromfile(os.path.join(D, "nb_real.i32"), "<i4").reshape(n, n).astype(np.float32)
    nb_full = np.maximum(nb_raw, nb_raw.T)
    np.fill_diagonal(nb_full, 0.0)
    del nb_raw

    # SourceAFIS: não-simétrico; vale o maior escore entre as duas direções
    sa_raw = np.fromfile(os.path.join(D, "sa_real.f32"), "<f4").reshape(n, n)
    sa_full = np.maximum(sa_raw, sa_raw.T)
    np.fill_diagonal(sa_full, 0.0)
    del sa_raw

    # Listas de exclusão
    dups_exact = json.load(open(DUPS_EXATAS_PATH, encoding="utf-8")) if os.path.exists(DUPS_EXATAS_PATH) else []
    dups_tail = {f for g in dups_exact for f in g[1:]}

    if os.path.exists(SENSIBILIDADE_PATH):
        sens = json.load(open(SENSIBILIDADE_PATH, encoding="utf-8"))
        excl_14 = set(sens["excluidas"])
    else:
        excl_14 = {32, 71, 113, 167, 191, 230, 301, 378, 481, 484, 528, 585, 596, 598}

    return {
        "rec": rec,
        "n": n,
        "suj": suj,
        "mao": mao,
        "dedo": dedo,
        "tags": tags,
        "nb_full": nb_full,
        "sa_full": sa_full,
        "dups_exact": dups_exact,
        "dups_tail": dups_tail,
        "excl_14": excl_14
    }


def estatisticas_distribuicao(s):
    if len(s) == 0:
        return {"n": 0}
    res = {
        "n": int(s.size),
        "mean": round(float(np.mean(s)), 3),
        "std": round(float(np.std(s)), 3),
        "mediana": round(float(np.median(s)), 2),
        "p50": round(float(np.percentile(s, 50)), 2),
        "p90": round(float(np.percentile(s, 90)), 2),
        "p95": round(float(np.percentile(s, 95)), 2),
        "p99": round(float(np.percentile(s, 99)), 2),
        "p99.9": round(float(np.percentile(s, 99.9)), 2),
        "p99.99": round(float(np.percentile(s, 99.99)), 2),
        "max": round(float(np.max(s)), 2),
    }
    for t in (30, 40, 60):
        k = int((s >= t).sum())
        res[f"n_ge_{t}"] = k
        res[f"frac_ge_{t}"] = round(k / s.size, 8)
    return res


def testar_diferenca(s_same, s_diff, score_mat, valid_indices, same_i, same_j, n_perm=10000, seed=42):
    """
    Executa:
    1) Mann-Whitney U test (unilateral: mesmo_corpo > corpos_diferentes).
    2) Tamanhos de efeito (Cohen's d, AUC / Probabilidade de Superioridade, Cliff's delta).
    3) Teste de hipótese nula por permutação de rótulos de pessoa (re-embaralhamento de dedos).
    """
    n_same = len(s_same)
    n_diff = len(s_diff)
    n_tot = n_same + n_diff

    # Mann-Whitney U
    mw_res = stats.mannwhitneyu(s_same, s_diff, alternative="greater")
    auc = float(mw_res.statistic / (n_same * n_diff))
    cliffs_delta = 2.0 * auc - 1.0

    # Cohen's d
    var_pooled = ((n_same - 1) * np.var(s_same, ddof=1) + (n_diff - 1) * np.var(s_diff, ddof=1)) / (n_tot - 2)
    sd_pooled = np.sqrt(var_pooled) if var_pooled > 0 else 1.0
    cohens_d = float((np.mean(s_same) - np.mean(s_diff)) / sd_pooled)

    diff_medians = float(np.median(s_same) - np.median(s_diff))
    diff_means = float(np.mean(s_same) - np.mean(s_diff))

    # Permutação de rótulos de pessoa (embaralhando dedos no conjunto de sujeitos)
    rng = np.random.default_rng(seed)
    n_val = len(valid_indices)
    all_sum = float(np.sum(s_same) + np.sum(s_diff))

    null_diffs = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        perm = rng.permutation(n_val)
        pi = perm[same_i]
        pj = perm[same_j]
        perm_vals = score_mat[pi, pj]
        sum_s = float(np.sum(perm_vals))
        m_s = sum_s / n_same
        m_d = (all_sum - sum_s) / n_diff
        null_diffs[k] = m_s - m_d

    # P-valor empírico
    p_perm = float((np.sum(null_diffs >= diff_means) + 1) / (n_perm + 1))
    p_perm_two_tailed = float((np.sum(np.abs(null_diffs) >= np.abs(diff_means)) + 1) / (n_perm + 1))

    return {
        "mann_whitney_u": float(mw_res.statistic),
        "mann_whitney_p": float(mw_res.pvalue),
        "auc": round(auc, 6),
        "cliffs_delta": round(cliffs_delta, 6),
        "cohens_d": round(cohens_d, 6),
        "diff_means": round(diff_means, 4),
        "diff_medians": round(diff_medians, 4),
        "permutacao": {
            "n_permutacoes": n_perm,
            "seed": seed,
            "obs_diff_means": round(diff_means, 4),
            "null_mean": round(float(np.mean(null_diffs)), 6),
            "null_std": round(float(np.std(null_diffs)), 6),
            "p_valor_unilateral": p_perm,
            "p_valor_bilateral": p_perm_two_tailed
        }
    }


def calcular_curva_maximo(suj_arr, nb_mat, sa_mat, k_list, reps=20, seed=0):
    rng = np.random.default_rng(seed)
    sujeitos = np.unique(suj_arr)
    n_suj = len(sujeitos)
    suj_to_indices = {s: np.where(suj_arr == s)[0] for s in sujeitos}
    res = []

    for k in k_list:
        k_eff = min(k, n_suj)
        mx_nb = []
        mx_sa = []
        n_reps = 1 if k_eff == n_suj else reps

        for _ in range(n_reps):
            sel_suj = rng.choice(sujeitos, k_eff, replace=False)
            idx_list = [suj_to_indices[s] for s in sel_suj]
            idx_all = np.concatenate(idx_list)

            # Submatrizes
            M_nb = nb_mat[np.ix_(idx_all, idx_all)].copy()
            M_sa = sa_mat[np.ix_(idx_all, idx_all)].copy()

            # Mascarar blocos intra-sujeito (mesmo corpo) com -1
            offset = 0
            for idx_s in idx_list:
                sz = len(idx_s)
                M_nb[offset:offset + sz, offset:offset + sz] = -1.0
                M_sa[offset:offset + sz, offset:offset + sz] = -1.0
                offset += sz

            mx_nb.append(float(M_nb.max()))
            mx_sa.append(float(M_sa.max()))

        n_pares_aprox = int(k_eff * 10 * (k_eff - 1) * 10 / 2) if k_eff > 1 else 0
        res.append({
            "sujeitos": k_eff,
            "pares_aprox": n_pares_aprox,
            "max_nbis": round(float(np.mean(mx_nb)), 1),
            "std_nbis": round(float(np.std(mx_nb)), 2),
            "max_sa": round(float(np.mean(mx_sa)), 1),
            "std_sa": round(float(np.std(mx_sa)), 2)
        })
        if k_eff == n_suj:
            break
    return res


def main():
    t_inicio = time.time()
    dados = carregar_dados()
    rec = dados["rec"]
    n = dados["n"]
    suj = dados["suj"]
    mao = dados["mao"]
    dedo = dados["dedo"]
    tags = dados["tags"]
    nb_full = dados["nb_full"]
    sa_full = dados["sa_full"]
    dups_tail = dados["dups_tail"]
    excl_14 = dados["excl_14"]

    iu_full, ju_full = np.triu_indices(n, 1)
    nb_pairs_raw = nb_full[iu_full, ju_full]
    sa_pairs_raw = sa_full[iu_full, ju_full]

    mesmo_raw = suj[iu_full] == suj[ju_full]
    homol_raw = mesmo_raw & (dedo[iu_full] == dedo[ju_full]) & (mao[iu_full] != mao[ju_full])

    # === ANTES DA EXCLUSÃO ===
    grupos_antes = {
        "corpos_diferentes": ~mesmo_raw,
        "mesmo_corpo": mesmo_raw,
        "mesmo_corpo_homologo": homol_raw,
        "mesmo_corpo_nao_homologo": mesmo_raw & ~homol_raw
    }

    dist_antes = {
        "nbis": {g: estatisticas_distribuicao(nb_pairs_raw[m]) for g, m in grupos_antes.items()},
        "sourceafis": {g: estatisticas_distribuicao(sa_pairs_raw[m]) for g, m in grupos_antes.items()}
    }

    # === DEPOIS DA EXCLUSÃO ===
    # Dedos válidos: não pertencem às 14 pessoas e não são cópias adicionais de imagens byte-idênticas
    valid_finger_mask = np.array([(s not in excl_14) and (t not in dups_tail) for s, t in zip(suj, tags)])
    valid_idx = np.where(valid_finger_mask)[0]
    n_valid = len(valid_idx)

    suj_val = suj[valid_idx]
    mao_val = mao[valid_idx]
    dedo_val = dedo[valid_idx]
    tags_val = tags[valid_idx]

    nb_mat_val = nb_full[np.ix_(valid_idx, valid_idx)]
    sa_mat_val = sa_full[np.ix_(valid_idx, valid_idx)]

    iu_val, ju_val = np.triu_indices(n_valid, 1)
    nb_pairs_val = nb_mat_val[iu_val, ju_val]
    sa_pairs_val = sa_mat_val[iu_val, ju_val]

    mesmo_val = suj_val[iu_val] == suj_val[ju_val]
    homol_val = mesmo_val & (dedo_val[iu_val] == dedo_val[ju_val]) & (mao_val[iu_val] != mao_val[ju_val])

    grupos_depois = {
        "corpos_diferentes": ~mesmo_val,
        "mesmo_corpo": mesmo_val,
        "mesmo_corpo_homologo": homol_val,
        "mesmo_corpo_nao_homologo": mesmo_val & ~homol_val
    }

    dist_depois = {
        "nbis": {g: estatisticas_distribuicao(nb_pairs_val[m]) for g, m in grupos_depois.items()},
        "sourceafis": {g: estatisticas_distribuicao(sa_pairs_val[m]) for g, m in grupos_depois.items()}
    }

    # Testes de diferença (mesmo corpo vs corpos diferentes) DEPOIS da exclusão
    same_pairs_idx = np.where(mesmo_val)[0]
    same_i = iu_val[same_pairs_idx]
    same_j = ju_val[same_pairs_idx]

    testes_depois = {
        "nbis": testar_diferenca(
            nb_pairs_val[mesmo_val], nb_pairs_val[~mesmo_val],
            nb_mat_val, valid_idx, same_i, same_j, n_perm=10000, seed=42
        ),
        "sourceafis": testar_diferenca(
            sa_pairs_val[mesmo_val], sa_pairs_val[~mesmo_val],
            sa_mat_val, valid_idx, same_i, same_j, n_perm=10000, seed=42
        )
    }

    # Curvas de crescimento do máximo
    k_targets = [30, 60, 120, 240, 480, 600]
    curva_antes = calcular_curva_maximo(suj, nb_full, sa_full, k_targets, reps=20, seed=0)
    curva_depois = calcular_curva_maximo(suj_val, nb_mat_val, sa_mat_val, [30, 60, 120, 240, 480, len(np.unique(suj_val))], reps=20, seed=0)

    # Contagem de pares >= 40
    pares_ge_40 = {
        "antes": {
            "nbis": {
                "corpos_diferentes": int((nb_pairs_raw[~mesmo_raw] >= 40).sum()),
                "mesmo_corpo": int((nb_pairs_raw[mesmo_raw] >= 40).sum()),
                "total": int((nb_pairs_raw >= 40).sum())
            },
            "sourceafis": {
                "corpos_diferentes": int((sa_pairs_raw[~mesmo_raw] >= 40).sum()),
                "mesmo_corpo": int((sa_pairs_raw[mesmo_raw] >= 40).sum()),
                "total": int((sa_pairs_raw >= 40).sum())
            },
            "ambos": {
                "corpos_diferentes": int(((sa_pairs_raw >= 40) & (nb_pairs_raw >= 40) & (~mesmo_raw)).sum()),
                "mesmo_corpo": int(((sa_pairs_raw >= 40) & (nb_pairs_raw >= 40) & (mesmo_raw)).sum()),
                "total": int(((sa_pairs_raw >= 40) & (nb_pairs_raw >= 40)).sum())
            }
        },
        "depois": {
            "nbis": {
                "corpos_diferentes": int((nb_pairs_val[~mesmo_val] >= 40).sum()),
                "mesmo_corpo": int((nb_pairs_val[mesmo_val] >= 40).sum()),
                "total": int((nb_pairs_val >= 40).sum())
            },
            "sourceafis": {
                "corpos_diferentes": int((sa_pairs_val[~mesmo_val] >= 40).sum()),
                "mesmo_corpo": int((sa_pairs_val[mesmo_val] >= 40).sum()),
                "total": int((sa_pairs_val >= 40).sum())
            },
            "ambos": {
                "corpos_diferentes": int(((sa_pairs_val >= 40) & (nb_pairs_val >= 40) & (~mesmo_val)).sum()),
                "mesmo_corpo": int(((sa_pairs_val >= 40) & (nb_pairs_val >= 40) & (mesmo_val)).sum()),
                "total": int(((sa_pairs_val >= 40) & (nb_pairs_val >= 40)).sum())
            }
        }
    }

    # Top pares remanescentes em SourceAFIS >= 40 após exclusão
    idx_sa_ge_40_depois = np.where(sa_pairs_val >= 40)[0]
    top_sa_depois = []
    for idx_p in idx_sa_ge_40_depois[np.argsort(-sa_pairs_val[idx_sa_ge_40_depois])]:
        i, j = iu_val[idx_p], ju_val[idx_p]
        top_sa_depois.append({
            "dedo_1": tags_val[i],
            "dedo_2": tags_val[j],
            "sa": round(float(sa_pairs_val[idx_p]), 2),
            "nbis": round(float(nb_pairs_val[idx_p]), 1),
            "mesmo_corpo": bool(mesmo_val[idx_p]),
            "homologo": bool(homol_val[idx_p])
        })

    resumo_completo = {
        "metadata": {
            "descricao": "Reanálise do Estudo 01 na SOCOFing sem duplicatas",
            "data_execucao": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tempo_execucao_s": round(time.time() - t_inicio, 2),
            "n_dedos_original": n,
            "n_sujeitos_original": int(len(set(suj))),
            "n_pares_original": int(len(iu_full)),
            "n_dedos_validos": n_valid,
            "n_sujeitos_validos": int(len(set(suj_val))),
            "n_pares_validos": int(len(iu_val)),
            "pessoas_excluidas_14": sorted(list(excl_14)),
            "imagens_byte_identicas_excluidas": sorted(list(dups_tail))
        },
        "distribuicoes": {
            "antes": dist_antes,
            "depois": dist_depois
        },
        "testes_diferenca": testes_depois,
        "curva_crescimento_maximo": {
            "antes": curva_antes,
            "depois": curva_depois
        },
        "pares_ge_40": pares_ge_40,
        "top_sa_ge_40_remanescentes": top_sa_depois
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resumo_completo, f, indent=2, ensure_ascii=False)

    # Imprimir relatório estruturado no terminal
    print("=" * 78)
    print("ESTUDO 01 — REANÁLISE SOCOFING SEM DUPLICATAS (NBIS e SourceAFIS)")
    print("=" * 78)
    print(f"Sujeitos: {len(set(suj))} -> {len(set(suj_val))} | Dedos: {n} -> {n_valid} | Pares: {len(iu_full):,} -> {len(iu_val):,}")
    print(f"Exclusões: 14 pessoas do exame pericial ({len(excl_14)} pessoas) + {len(dups_tail)} cópias byte-idênticas.")
    print()

    print("--- 1. DISTRIBUIÇÕES: 'MESMO CORPO' vs 'CORPOS DIFERENTES' ---")
    for matcher in ("sourceafis", "nbis"):
        print(f"\n[{matcher.upper()}]")
        print(f"{'Grupo':<28s} | {'Mediana':<8s} | {'p95':<8s} | {'p99':<8s} | {'Máximo':<8s} | {'N >= 40':<8s}")
        print("-" * 78)
        for estado, d_est in (("ANTES", dist_antes[matcher]), ("DEPOIS", dist_depois[matcher])):
            print(f"  -- {estado} --")
            for g_nome, g_label in [
                ("corpos_diferentes", "Corpos Diferentes"),
                ("mesmo_corpo", "Mesmo Corpo (Todos)"),
                ("mesmo_corpo_homologo", "Mesmo Corpo (Homólogos)"),
                ("mesmo_corpo_nao_homologo", "Mesmo Corpo (Não-Homólogos)")
            ]:
                st = d_est[g_nome]
                print(f"  {g_label:<26s} | {st['mediana']:<8.2f} | {st['p95']:<8.2f} | {st['p99']:<8.2f} | {st['max']:<8.2f} | {st['n_ge_40']:<8d}")

    print()
    print("--- 2. TESTES DE HIPÓTESE E TAMANHO DE EFEITO (DEPOIS DA EXCLUSÃO) ---")
    for matcher in ("sourceafis", "nbis"):
        tst = testes_depois[matcher]
        perm = tst["permutacao"]
        print(f"\n[{matcher.upper()}] Mesmo Corpo vs Corpos Diferentes:")
        print(f"  Diferença de Médias (Obs): {tst['diff_means']:+.4f} (Mediana: {tst['diff_medians']:+.4f})")
        print(f"  Cohen's d:                {tst['cohens_d']:+.4f}")
        print(f"  AUC (Superioridade):      {tst['auc']:.4f} | Cliff's delta: {tst['cliffs_delta']:+.4f}")
        print(f"  Mann-Whitney U:           U = {tst['mann_whitney_u']:.1f}, p = {tst['mann_whitney_p']:.3e}")
        print(f"  Permutação de Pessoas:    p (unilateral) = {perm['p_valor_unilateral']:.4f} "
              f"| Nulo médio = {perm['null_mean']:+.5f} ± {perm['null_std']:.5f} (10.000 perm)")

    print()
    print("--- 3. CURVA DO MÁXIMO (MÁXIMO IMPOSTOR vs Nº DE SUJEITOS) ---")
    print(f"{'Sujeitos':<10s} | {'SA Max (Antes)':<15s} | {'SA Max (Depois)':<15s} | {'NB Max (Antes)':<15s} | {'NB Max (Depois)':<15s}")
    print("-" * 78)
    for ca, cd in zip(curva_antes, curva_depois):
        print(f"{cd['sujeitos']:<10d} | {ca['max_sa']:<15.1f} | {cd['max_sa']:<15.1f} | {ca['max_nbis']:<15.1f} | {cd['max_nbis']:<15.1f}")

    print()
    print("--- 4. PARES COM ESCORE >= 40 ---")
    print(f"SourceAFIS >= 40: Antes = {pares_ge_40['antes']['sourceafis']['total']} ({pares_ge_40['antes']['sourceafis']['mesmo_corpo']} mesmo corpo, {pares_ge_40['antes']['sourceafis']['corpos_diferentes']} dif) "
          f"-> Depois = {pares_ge_40['depois']['sourceafis']['total']} ({pares_ge_40['depois']['sourceafis']['mesmo_corpo']} mesmo corpo, {pares_ge_40['depois']['sourceafis']['corpos_diferentes']} dif)")
    print(f"NBIS >= 40:       Antes = {pares_ge_40['antes']['nbis']['total']} -> Depois = {pares_ge_40['depois']['nbis']['total']}")
    print(f"AMBOS >= 40:      Antes = {pares_ge_40['antes']['ambos']['total']} (41 confirmados duplicatas) -> Depois = {pares_ge_40['depois']['ambos']['total']}")
    if len(top_sa_depois) > 0:
        print(f"\nTop 5 pares remanescentes em SourceAFIS >= 40 (Depois):")
        for p in top_sa_depois[:5]:
            tipo = "MESMO CORPO" if p["mesmo_corpo"] else "corpos diferentes"
            print(f"  {p['dedo_1']:<20s} x {p['dedo_2']:<20s} | SA = {p['sa']:<5.1f} | NBIS = {p['nbis']:<4.0f} | {tipo}")

    print()
    print(f"Arquivo salvo com sucesso em: {OUT_JSON}")
    print(f"Tempo total: {time.time() - t_inicio:.2f} s")


if __name__ == "__main__":
    main()
