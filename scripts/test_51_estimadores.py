#!/usr/bin/env python3
# ==============================================================================
# scripts/test_51_estimadores.py
# Testes unitários para o módulo 51-estimadores.py contra soluções analíticas.
# ==============================================================================

import unittest
import importlib.util
import os
import numpy as np

# Carregar 51-estimadores.py dinamicamente (devido ao número no nome)
spec = importlib.util.spec_from_file_location(
    "estimadores",
    os.path.join(os.path.dirname(__file__), "51-estimadores.py")
)
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)


class TestEstimadoresContinuos(unittest.TestCase):
    """Testes dos estimadores contínuos via Cópula Gaussiana (Ince et al. 2017)."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_gaussian_copula_known_covariance(self):
        """
        Gaussiana multivariada com matriz de correlação equicorrelacionada R.
        Solução analítica: det(R) = (1 - rho)^(K-1) * (1 + (K-1)*rho)
        TC_teorico = -0.5 * ln det(R)
        Tolerância declarada: erro < 0.03 para N grande (100.000).
        """
        K = 5
        rho = 0.5
        N = 100000

        # Construir matriz de correlação teórica R
        R_true = np.full((K, K), rho) + (1.0 - rho) * np.eye(K)
        det_true = ((1.0 - rho) ** (K - 1)) * (1.0 + (K - 1) * rho)
        tc_teorico = -0.5 * np.log(det_true)

        # Gerar amostras gaussianas correlacionadas
        L = np.linalg.cholesky(R_true)
        Z_indep = self.rng.normal(size=(N, K))
        X_corr = Z_indep @ L.T

        tc_estimado = est.tc_gaussian_copula(X_corr)

        erro = abs(tc_estimado - tc_teorico)
        tolerancia = 0.03
        self.assertLess(
            erro,
            tolerancia,
            f"TC estimado ({tc_estimado:.4f}) difere do teórico ({tc_teorico:.4f}) além da tolerância ({erro:.4f} >= {tolerancia})"
        )

    def test_gaussian_copula_independent(self):
        """Variáveis independentes contínuas devem ter TC aproximada de zero (TC < 0.005)."""
        N = 50000
        K = 10
        X_indep = self.rng.normal(size=(N, K))

        tc_estimado = est.tc_gaussian_copula(X_indep)
        self.assertLess(tc_estimado, 0.005, f"TC para variáveis independentes deveria ser ~0, obtido {tc_estimado:.6f}")

    def test_pointwise_tc_mean_equals_population_tc(self):
        """A média da TC pontual deve coincidir com a TC populacional até precisão de máquina."""
        K = 4
        N = 20000
        R = np.array([
            [1.0, 0.4, 0.3, 0.2],
            [0.4, 1.0, 0.5, 0.3],
            [0.3, 0.5, 1.0, 0.6],
            [0.2, 0.3, 0.6, 1.0]
        ])
        L = np.linalg.cholesky(R)
        X = self.rng.normal(size=(N, K)) @ L.T

        tc_pop = est.tc_gaussian_copula(X)
        tc_pts = est.pointwise_tc_gaussian_copula(X)
        tc_pts_mean = float(np.mean(tc_pts))

        self.assertAlmostEqual(
            tc_pop,
            tc_pts_mean,
            places=8,
            msg=f"Média da TC pontual ({tc_pts_mean:.8f}) difere da TC populacional ({tc_pop:.8f})"
        )

    def test_oinfo_gaussian_copula_properties(self):
        """
        Propriedades da O-information:
        1) Independentes -> Omega approx 0
        2) Redundantes (correlacionadas positivamente) -> Omega > 0
        3) K=2 -> Omega identicamente 0
        """
        N = 50000

        # K=2
        X2 = self.rng.normal(size=(N, 2))
        self.assertEqual(est.oinfo_gaussian_copula(X2), 0.0)

        # Independentes K=4
        X_indep = self.rng.normal(size=(N, 4))
        oinfo_indep = est.oinfo_gaussian_copula(X_indep)
        self.assertLess(abs(oinfo_indep), 0.005)

        # Redundantes K=4 (rho = 0.6)
        R_red = np.full((4, 4), 0.6) + 0.4 * np.eye(4)
        X_red = self.rng.normal(size=(N, 4)) @ np.linalg.cholesky(R_red).T
        oinfo_red = est.oinfo_gaussian_copula(X_red)
        self.assertGreater(oinfo_red, 0.0, "O-information para variáveis redundantes deve ser positiva")

    def test_pairwise_mi_continuous(self):
        """MI bivariada: independentes -> ~0; correlacionadas -> -0.5*ln(1 - rho^2)."""
        N = 100000
        rho = 0.7
        cov = [[1.0, rho], [rho, 1.0]]
        xy = self.rng.multivariate_normal([0, 0], cov, size=N)

        mi_est = est.mi_pairwise_gaussian_copula(xy[:, 0], xy[:, 1])
        mi_teorica = -0.5 * np.log(1.0 - rho ** 2)

        self.assertAlmostEqual(mi_est, mi_teorica, delta=0.02)


class TestEstimadoresDiscretos(unittest.TestCase):
    """Testes dos estimadores discretos com correção de Miller-Madow (1955)."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_discrete_identical_copies(self):
        """
        Cópias idênticas discretas:
        X1 = X2 = ... = XK
        TC = sum_{j=1}^K H(Xj) - H(X1..XK) = K * H(X1) - H(X1) = (K - 1) * H(X1).
        Deve ser exato para qualquer correção consistente.
        """
        N = 5000
        K = 4
        # Variável com 4 classes de probabilidades desiguais
        x1 = self.rng.choice(["A", "L", "R", "W"], size=N, p=[0.1, 0.4, 0.3, 0.2])
        X_ident = np.column_stack([x1] * K)

        h_x1 = est.entropy_miller_madow(x1)
        tc_est = est.tc_discrete_miller_madow(X_ident)
        tc_esperado = (K - 1) * h_x1

        self.assertAlmostEqual(
            tc_est,
            tc_esperado,
            places=10,
            msg=f"TC de cópias idênticas ({tc_est}) difere de (K-1)*H(X1) ({tc_esperado})"
        )

    def test_discrete_independent(self):
        """Variáveis discretas independentes devem ter TC próxima de zero."""
        N = 50000
        K = 4
        X_indep = np.column_stack([
            self.rng.choice(["A", "B", "C"], size=N, p=[0.5, 0.3, 0.2])
            for _ in range(K)
        ])
        tc_indep = est.tc_discrete_miller_madow(X_indep)
        self.assertLess(tc_indep, 0.01)

    def test_discrete_pairwise_mi_identical(self):
        """MI de cópias idênticas discretas: MI(x, x) = H(x)."""
        N = 5000
        x = self.rng.choice(["Arc", "Loop", "Whorl"], size=N, p=[0.2, 0.5, 0.3])
        mi_xx = est.mi_pairwise_discrete_miller_madow(x, x)
        h_x = est.entropy_miller_madow(x)
        self.assertAlmostEqual(mi_xx, h_x, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
