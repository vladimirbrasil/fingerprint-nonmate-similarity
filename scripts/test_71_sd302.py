#!/usr/bin/env python3
# ==============================================================================
# scripts/test_71_sd302.py
# Testes unitários sintéticos para o pipeline do Estudo 03 (scripts/71-sd302-analise.py).
#
# Casos testados:
# (a) Simetria do CORPO:
#     Latente por pessoa x dedo compartilhado entre mãos e entre capturas + ruído por captura
#     -> S(c1, c2) cruzado > 0 e r > 0.7.
# (b) Simetria da CAPTURA:
#     Latente por pessoa x dedo x captura compartilhado só entre as mãos dentro da mesma captura
#     -> S(c1, c2) cruzado ~= 0 e r < 0.3.
# (c) Ordem de colunas:
#     Trocar FRGP mapeado destrói S.
# (d) Parser de errata:
#     Verifica extração correta nos formatos "Device X" e stems de arquivo.
# ==============================================================================

import importlib.util
import os
import tempfile
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Carregar 71 e 70 dinamicamente
def carregar_modulo(nome, rel_path):
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(nome, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

an71 = carregar_modulo("an71", os.path.join("scripts", "71-sd302-analise.py"))
feat70 = carregar_modulo("feat70", os.path.join("scripts", "70-sd302-features.py"))


def test_parser_errata():
    """(d) Parser de errata nos dois formatos: 'Device X' e stems de arquivo."""
    texto_errata = """
    Known Errata -- Special Database 302
    * 00002361, Device V, right ring is actually right index.
    * 00002534, Device S, thumbs swapped.
    * 00002480_6E_X_050_BT_D800_1103PPI_16BPC_1CH_LP20_1 source is finger 3, but is labeled as finger 2.
    * 00002420_U_500_roll_01.png mislabeled
    * The segmented slap thumb fingerprints for 00002420, device R were swapped.
    * 00002497, Device C, Finger 08: Unusable region
    """
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tf:
        tf.write(texto_errata)
        tf_name = tf.name

    try:
        errata = feat70.carregar_errata(tf_name)
        # Formato 1
        assert ("00002361", "V") in errata
        assert ("00002534", "S") in errata
        assert ("00002420", "R") in errata
        assert ("00002497", "C") in errata

        # Formato 2: stem de captura nossa (U)
        assert ("00002420", "U") in errata
        # Stem cujo 2º campo não é captura nossa (6E) não causa falso positivo em captura conhecida
        assert ("00002480", "U") not in errata
    finally:
        if os.path.exists(tf_name):
            os.remove(tf_name)


def test_simetria_do_corpo():
    """
    (a) Simetria do CORPO:
    Latente por pessoa x dedo compartilhado entre mãos e entre capturas + ruído por captura
    -> S(c1, c2) cruzado > 0 e r > 0.7.
    """
    rng = np.random.default_rng(101)
    N = 1000

    # 5 traços intrínsecos de dedos (polegar..mínimo) por pessoa
    traço_corpo = rng.normal(size=(N, 5))

    # Captura 1 (U): mãos L e R recebem o mesmo traço do corpo + ruído leve
    ruido_U = rng.normal(scale=0.3, size=(N, 10))
    # Colunas 0..4 = L1..L5, 5..9 = R1..R5
    mat_U = np.column_stack([traço_corpo, traço_corpo]) + ruido_U

    # Captura 2 (V): mãos L e R recebem o MESMO traço do corpo + ruído leve diferente
    ruido_V = rng.normal(scale=0.3, size=(N, 10))
    mat_V = np.column_stack([traço_corpo, traço_corpo]) + ruido_V

    pessoas = np.array([f"{i:08d}" for i in range(N)])
    cdata_U = {"pessoas": pessoas, "N": N, "o_entropia": mat_U}
    cdata_V = {"pessoas": pessoas, "N": N, "o_entropia": mat_V}

    s_uu, _, _, _, _ = an71.calcular_S_par(cdata_U, cdata_U, "o_entropia")
    s_vv, _, _, _, _ = an71.calcular_S_par(cdata_V, cdata_V, "o_entropia")
    s_uv, _, _, _, _ = an71.calcular_S_par(cdata_U, cdata_V, "o_entropia")
    s_vu, _, _, _, _ = an71.calcular_S_par(cdata_V, cdata_U, "o_entropia")

    s_cruz = 0.5 * (s_uv + s_vu)
    s_auto = 0.5 * (s_uu + s_vv)
    r = s_cruz / s_auto

    # S cruzado deve ser fortemente positivo e r próximo de 1 (> 0.7)
    assert s_cruz > 0.1, f"Esperado s_cruz > 0.1, obtido {s_cruz}"
    assert r > 0.7, f"Esperado r > 0.7, obtido {r}"


def test_simetria_da_captura():
    """
    (b) Simetria da CAPTURA:
    Latente compartilhado só entre as mãos dentro da mesma captura
    -> S cruzado ~= 0 e r < 0.3.
    """
    rng = np.random.default_rng(202)
    N = 1000

    # Latente específico da captura U
    traço_cap_U = rng.normal(size=(N, 5))
    ruido_U = rng.normal(scale=0.3, size=(N, 10))
    mat_U = np.column_stack([traço_cap_U, traço_cap_U]) + ruido_U

    # Latente específico da captura V (independente de U)
    traço_cap_V = rng.normal(size=(N, 5))
    ruido_V = rng.normal(scale=0.3, size=(N, 10))
    mat_V = np.column_stack([traço_cap_V, traço_cap_V]) + ruido_V

    pessoas = np.array([f"{i:08d}" for i in range(N)])
    cdata_U = {"pessoas": pessoas, "N": N, "o_entropia": mat_U}
    cdata_V = {"pessoas": pessoas, "N": N, "o_entropia": mat_V}

    s_uu, _, _, _, _ = an71.calcular_S_par(cdata_U, cdata_U, "o_entropia")
    s_vv, _, _, _, _ = an71.calcular_S_par(cdata_V, cdata_V, "o_entropia")
    s_uv, _, _, _, _ = an71.calcular_S_par(cdata_U, cdata_V, "o_entropia")
    s_vu, _, _, _, _ = an71.calcular_S_par(cdata_V, cdata_U, "o_entropia")

    s_cruz = 0.5 * (s_uv + s_vu)
    s_auto = 0.5 * (s_uu + s_vv)
    r = s_cruz / s_auto

    # Dentro de cada captura há simetria
    assert s_auto > 0.1, f"Esperado s_auto > 0.1, obtido {s_auto}"
    # Mas cruzado entre capturas cai para ~0 e r < 0.3
    assert abs(s_cruz) < 0.05, f"Esperado |s_cruz| < 0.05, obtido {s_cruz}"
    assert r < 0.3, f"Esperado r < 0.3, obtido {r}"


def test_ordem_de_colunas_destroi_S():
    """
    (c) Ordem de colunas: trocar FRGP mapeado destrói S.
    Se a ordem correta [L1..L5, R1..R5] for embaralhada ou invertida,
    a correspondência de homólogos é quebrada.
    """
    rng = np.random.default_rng(303)
    N = 1000

    # Simetria perfeita entre homólogos
    traço = rng.normal(size=(N, 5))
    mat_correta = np.column_stack([traço, traço]) + rng.normal(scale=0.1, size=(N, 10))

    s_correta, hom_c, cruz_c, _, _ = an71.calcular_S_e_classes(mat_correta)
    assert s_correta > 0.5, f"Esperado S alto com ordem correta, obtido {s_correta}"

    # Inverter ordem da mão direita: colunas 5..9 tornam-se 9..5 (R5..R1 em vez de R1..R5)
    mat_permutada = mat_correta.copy()
    mat_permutada[:, 5:] = mat_correta[:, [9, 8, 7, 6, 5]]

    s_permutada, hom_p, cruz_p, _, _ = an71.calcular_S_e_classes(mat_permutada)
    # Na ordem invertida, pares cruzados não-homólogos viram homólogos aparentes e vice-versa
    # S deve cair drasticamente em relação a s_correta
    assert s_permutada < 0.1, f"Esperado S destruído ao trocar ordem de colunas, obtido {s_permutada}"
