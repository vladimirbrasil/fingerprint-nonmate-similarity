#!/usr/bin/env python3
# ==============================================================================
# scripts/51-estimadores.py
# Módulo de estimadores de Teoria da Informação Multivariada (Estudo 02).
#
# Métodos implementados:
# 1) Variáveis Contínuas (Gaussian Copula / Ince et al. 2017):
#    - copula_transform: Transformação de postos empíricos para normais padrão.
#    - tc_gaussian_copula: Correlação Total (TC / multi-informação) via -0.5 * ln det(R).
#    - oinfo_gaussian_copula: O-information (sinergia vs redundância) via Rosas et al. (2019).
#    - mi_pairwise_gaussian_copula: Informação mútua bivariada contínua.
#    - mi_matrix_gaussian_copula: Matriz de MI par a par para todas as variáveis.
#    - pointwise_tc_gaussian_copula: TC pontual (local) por amostra (log p(x) - sum log p(xi)).
#
# 2) Variáveis Discretas (Plug-in com correção Miller-Madow 1955):
#    - entropy_miller_madow: Entropia unidimensional com correção (m-1)/(2N).
#    - joint_entropy_miller_madow: Entropia conjunta com correção de Miller-Madow.
#    - tc_discrete_miller_madow: Correlação Total discreta (soma das marginais - conjunta).
#    - oinfo_discrete_miller_madow: O-information discreta.
#    - mi_pairwise_discrete_miller_madow: MI bivariada discreta.
#    - pointwise_tc_discrete: TC pontual discreta por observação.
# ==============================================================================

import numpy as np
from scipy import stats
from collections import Counter


def copula_transform(X):
    """
    Aplica a transformação de cópula gaussiana (Ince et al. 2017).
    Converte cada coluna de X (N x K) para uma distribuição normal padrão N(0, 1)
    preservando estritamente a estrutura de dependência não linear e invariância a
    transformações monotônicas.

    Parâmetros:
        X: array-like de formato (N, K) ou (N,)
    Retorna:
        Z: np.ndarray de formato (N, K) com marginais normais padrão
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, np.newaxis]

    N, K = X_arr.shape
    if N < 2:
        raise ValueError("Número de amostras N deve ser >= 2.")

    Z = np.zeros((N, K), dtype=float)
    for j in range(K):
        col = X_arr[:, j]
        # Postos empíricos com desempate pela média
        ranks = stats.rankdata(col)
        # Marginais uniformes estritas no intervalo aberto (0, 1)
        u = ranks / (N + 1.0)
        # Quantil da normal padrão
        Z[:, j] = stats.norm.ppf(u)

    return Z


def _calcular_logdet_seguro(R, regularize=1e-8):
    """
    Calcula log det(R) de forma numericamente estável para matriz de correlação.
    Garante que autovalores degenerados ou colineares sejam limitados por regularize.
    """
    K = R.shape[0]
    # Forçar simetria estrita e diagonal unitária
    R_sym = 0.5 * (R + R.T)
    np.fill_diagonal(R_sym, 1.0)

    # Decomposição em autovalores para tratamento de quase-singularidades
    eigvals = np.linalg.eigvalsh(R_sym)
    eigvals = np.maximum(eigvals, regularize)
    logdet = float(np.sum(np.log(eigvals)))
    return logdet, R_sym


def tc_gaussian_copula(X, regularize=1e-8):
    """
    Calcula a Correlação Total (TC / Multi-informação) via Cópula Gaussiana (Ince et al. 2017).
    TC(X_1, ..., X_K) = sum_{j=1}^K H(X_j) - H(X_1, ..., X_K) = -0.5 * ln det(R)
    onde R é a matriz de correlação das variáveis transformadas pela cópula.

    Parâmetros:
        X: array-like (N, K) contendo K variáveis contínuas
        regularize: piso para autovalores para evitar instabilidade numérica
    Retorna:
        tc: float (TC em nats >= 0)
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        return 0.0

    N, K = X_arr.shape
    if K <= 1:
        return 0.0

    Z = copula_transform(X_arr)
    R = np.corrcoef(Z, rowvar=False)
    logdet, _ = _calcular_logdet_seguro(R, regularize=regularize)

    tc = -0.5 * logdet
    return max(0.0, float(tc))


def oinfo_gaussian_copula(X, regularize=1e-8):
    """
    Calcula a O-information de Rosas et al. (2019) via Cópula Gaussiana.
    Omega(X) = TC(X) - DTC(X) = ((K - 2)/2) * ln det(R) - 0.5 * sum_{j=1}^K ln det(R_{-j})
    onde R_{-j} é a matriz de correlação excluindo a j-ésima variável.

    Interpretação:
        Omega > 0: Sistema dominado por redundância.
        Omega < 0: Sistema dominado por sinergia.
        Omega = 0: Independência ou equilíbrio sinergia/redundância (identicamente 0 para K=2).

    Parâmetros:
        X: array-like (N, K)
    Retorna:
        oinfo: float (O-information em nats)
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        return 0.0

    N, K = X_arr.shape
    if K < 2:
        return 0.0
    if K == 2:
        return 0.0  # O-information para 2 variáveis é identicamente zero

    Z = copula_transform(X_arr)
    R = np.corrcoef(Z, rowvar=False)
    logdet_full, R_sym = _calcular_logdet_seguro(R, regularize=regularize)

    soma_logdet_leave_one_out = 0.0
    for j in range(K):
        # Excluir j-ésima linha e coluna
        idx = [i for i in range(K) if i != j]
        R_sub = R_sym[np.ix_(idx, idx)]
        logdet_sub, _ = _calcular_logdet_seguro(R_sub, regularize=regularize)
        soma_logdet_leave_one_out += logdet_sub

    oinfo = 0.5 * (K - 2.0) * logdet_full - 0.5 * soma_logdet_leave_one_out
    return float(oinfo)


def mi_pairwise_gaussian_copula(x, y, regularize=1e-8):
    """
    Informação mútua bivariada entre x e y via Cópula Gaussiana:
    MI(x, y) = -0.5 * ln(1 - rho^2)
    onde rho é a correlação linear entre as cópulas normais padrão de x e y.
    """
    X_pair = np.column_stack([x, y])
    Z = copula_transform(X_pair)
    rho = float(np.corrcoef(Z[:, 0], Z[:, 1])[0, 1])
    # Limitar rho em (-1, 1)
    rho = max(-1.0 + regularize, min(1.0 - regularize, rho))
    mi = -0.5 * np.log(1.0 - rho ** 2)
    return max(0.0, float(mi))


def mi_matrix_gaussian_copula(X, regularize=1e-8):
    """
    Gera a matriz K x K de informações mútuas par a par via Cópula Gaussiana.
    """
    X_arr = np.asarray(X, dtype=float)
    N, K = X_arr.shape
    Z = copula_transform(X_arr)
    R = np.corrcoef(Z, rowvar=False)

    MI_mat = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(i + 1, K):
            rho = float(R[i, j])
            rho = max(-1.0 + regularize, min(1.0 - regularize, rho))
            mi = -0.5 * np.log(1.0 - rho ** 2)
            val = max(0.0, float(mi))
            MI_mat[i, j] = val
            MI_mat[j, i] = val

    return MI_mat


def pointwise_tc_gaussian_copula(X, regularize=1e-8):
    """
    Calcula a Correlação Total Pontual (local) de cada observação i:
    tc_i = ln p(z_i) - sum_{j=1}^K ln p(z_{i,j})
         = -0.5 * ln det(R) - 0.5 * z_i^T (R^{-1} - I) z_i

    Propriedade exata: E[tc_i] == TC(X).

    Parâmetros:
        X: array-like (N, K)
    Retorna:
        tc_local: np.ndarray (N,) com o valor de TC pontual de cada indivíduo
    """
    X_arr = np.asarray(X, dtype=float)
    N, K = X_arr.shape
    if K <= 1:
        return np.zeros(N, dtype=float)

    Z = copula_transform(X_arr)
    R = np.corrcoef(Z, rowvar=False)
    logdet, R_sym = _calcular_logdet_seguro(R, regularize=regularize)

    # Inversa segura de R via autovalores regularizados
    eigvals, eigvecs = np.linalg.eigh(R_sym)
    eigvals_reg = np.maximum(eigvals, regularize)
    R_inv = eigvecs @ np.diag(1.0 / eigvals_reg) @ eigvecs.T

    M = R_inv - np.eye(K)
    # Forma quadrática z_i^T M z_i para cada linha i
    quad = np.sum((Z @ M) * Z, axis=1)

    tc_local = -0.5 * logdet - 0.5 * quad
    return tc_local


# ==============================================================================
# ESTIMADORES DISCRETOS COM CORREÇÃO MILLER-MADOW (1955)
# ==============================================================================

def entropy_miller_madow(x):
    """
    Entropia de Shannon (em nats) para variável discreta com correção de Miller-Madow:
    H_MM = - sum (n_s/N) ln (n_s/N) + (m - 1) / (2N)
    onde m é o número de classes observadas com n_s > 0.
    """
    x_list = list(x)
    N = len(x_list)
    if N <= 1:
        return 0.0

    counts = np.array(list(Counter(x_list).values()), dtype=float)
    p = counts / float(N)
    p_pos = p[p > 0]
    h_plug = -float(np.sum(p_pos * np.log(p_pos)))
    m = len(counts)
    h_mm = h_plug + (m - 1.0) / (2.0 * N)
    return max(0.0, float(h_mm))


def joint_entropy_miller_madow(X):
    """
    Entropia conjunta (em nats) de variáveis discretas com correção de Miller-Madow.
    X: array 2D de formato (N, K) contendo tuplas/valores discretos.
    """
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        return entropy_miller_madow(X_arr)

    N, K = X_arr.shape
    if N <= 1:
        return 0.0

    tuples = [tuple(row) for row in X_arr]
    return entropy_miller_madow(tuples)


def tc_discrete_miller_madow(X):
    """
    Correlação Total (TC) para variáveis discretas com Miller-Madow:
    TC(X_1, ..., X_K) = sum_{j=1}^K H_MM(X_j) - H_MM(X_1, ..., X_K)
    """
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        return 0.0

    N, K = X_arr.shape
    if K <= 1:
        return 0.0

    sum_marg = sum(entropy_miller_madow(X_arr[:, j]) for j in range(K))
    h_joint = joint_entropy_miller_madow(X_arr)
    tc = sum_marg - h_joint
    return max(0.0, float(tc))


def oinfo_discrete_miller_madow(X):
    """
    O-information para variáveis discretas via Miller-Madow:
    Omega(X) = (K - 2) * H_MM(X) + sum_{j=1}^K [ H_MM(X_j) - H_MM(X_{-j}) ]
    """
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        return 0.0

    N, K = X_arr.shape
    if K < 2:
        return 0.0
    if K == 2:
        return 0.0

    h_full = joint_entropy_miller_madow(X_arr)
    h_margs = [entropy_miller_madow(X_arr[:, j]) for j in range(K)]

    soma_leave_one_out = 0.0
    for j in range(K):
        idx = [i for i in range(K) if i != j]
        h_sub = joint_entropy_miller_madow(X_arr[:, idx])
        soma_leave_one_out += (h_margs[j] - h_sub)

    oinfo = (K - 2.0) * h_full + soma_leave_one_out
    return float(oinfo)


def mi_pairwise_discrete_miller_madow(x, y):
    """
    Informação mútua bivariada para variáveis discretas:
    MI(x, y) = H_MM(x) + H_MM(y) - H_MM(x, y)
    """
    h_x = entropy_miller_madow(x)
    h_y = entropy_miller_madow(y)
    h_xy = joint_entropy_miller_madow(np.column_stack([x, y]))
    mi = h_x + h_y - h_xy
    return max(0.0, float(mi))


def pointwise_tc_discrete(X, eps=1e-12):
    """
    TC pontual (local) de cada observação para variáveis discretas:
    tc_i = ln p(x_{i,1}, ..., x_{i,K}) - sum_{j=1}^K ln p(x_{i,j})
    calculada com base nas probabilidades empíricas da amostra.
    """
    X_arr = np.asarray(X)
    N, K = X_arr.shape
    if K <= 1:
        return np.zeros(N, dtype=float)

    tuples = [tuple(row) for row in X_arr]
    counts_joint = Counter(tuples)
    counts_marg = [Counter(X_arr[:, j]) for j in range(K)]

    tc_local = np.zeros(N, dtype=float)
    for i in range(N):
        row_tuple = tuples[i]
        p_joint = counts_joint[row_tuple] / float(N)
        sum_log_marg = 0.0
        for j in range(K):
            val = X_arr[i, j]
            p_marg = counts_marg[j][val] / float(N)
            sum_log_marg += np.log(max(eps, p_marg))

        log_joint = np.log(max(eps, p_joint))
        tc_local[i] = log_joint - sum_log_marg

    return tc_local
