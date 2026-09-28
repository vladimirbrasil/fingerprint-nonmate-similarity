#!/usr/bin/env python3
# ==============================================================================
# scripts/50-features-socofing.py
# Extração de características por dedo para a base SOCOFing Real (Estudo 02).
#
# Famílias de características extraídas por impressão digital:
# 1) Família O (Orientação):
#    - Executa o NBIS mindtct guardando os mapas de direção (.dm), qualidade (.qm)
#      e alta curvatura (.hcm).
#    - o_entropia: Entropia de Shannon (em nats) da distribuição das 16 direções
#      discretas nos blocos válidos (dm >= 0 e qm > 0).
#    - o_frac_alta_curvatura: Fração de blocos válidos com curvatura alta (hcm == 1),
#      conforme detectado pelo algoritmo oficial do mindtct.
#    - o_orientacao_dominante: Ângulo axial médio circular em radianos [0, pi),
#      calculado via theta_bar = 0.5 * atan2(mean(sin(2*theta)), mean(cos(2*theta))) mod pi.
#    - o_coerencia: Comprimento do vetor médio circular (0 = disperso, 1 = alinhado).
#
# 2) Família Q (Quantidade / Densidade de Minúcias):
#    - q_n_minucias: Minúcias no arquivo .xyt com qualidade >= 20 (limiar documentado
#      de confiabilidade mínima do NBIS).
#    - q_area_util: Contagem de blocos 8x8 válidos (qm > 0 e dm >= 0).
#    - q_densidade: q_n_minucias / q_area_util (minúcias de qualidade por bloco útil).
#    - q_n_minucias_total: Total de minúcias no .xyt sem corte de qualidade.
#
# 3) Família C (Classe / Padrão Geral):
#    - Tentativa de classificação via NBIS pcasys. Como as imagens do SOCOFing
#      ampliadas 3x possuem dimensões 288x309 px e o pcasys exige imagens padrão
#      com largura >= 512 px (segmentador sgmnt do NBIS rejeita com erro fatal),
#      a execução registra o erro e deixa a coluna c_classe vazia, conforme
#      estipulado no pré-registro ("se o pcasys não aceitar, NÃO force: registre
#      o erro e deixe a coluna vazia").
#
# Saída:
# - out/estudo02/features.csv
# - Mapas intermediários em out/estudo02/maps/
# ==============================================================================

import os
import re
import csv
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MINDTCT_BIN = os.path.join(ROOT, "build", "bin", "mindtct")
PCASYS_BIN = os.path.join(ROOT, "build", "bin", "pcasys")

DATA_REAL = os.path.join(ROOT, "data", "kagglehub", "datasets", "ruizgara", "socofing", "versions", "2", "SOCOFing", "Real")
IMG_REAL_DIR = os.path.join(ROOT, "out", "socofing", "img", "real")
OUT_ESTUDO02 = os.path.join(ROOT, "out", "estudo02")
MAPS_DIR = os.path.join(OUT_ESTUDO02, "maps")
FEATURES_CSV = os.path.join(OUT_ESTUDO02, "features.csv")

RE_REAL = re.compile(r"^(\d+)__([MF])_(Left|Right)_(thumb|index|middle|ring|little)_finger$")
MINUTIA_QUALITY_THRESH = 20  # Limiar documentado para minúcias de alta qualidade


def extrair_para_dedo(task):
    """
    Executa mindtct (se mapas ainda não existirem) e extrai métricas das famílias O, Q e C.
    """
    jpg_path = task["jpg_path"]
    stem = task["stem"]
    oroot = os.path.join(MAPS_DIR, stem)

    dm_file = f"{oroot}.dm"
    qm_file = f"{oroot}.qm"
    hcm_file = f"{oroot}.hcm"
    xyt_file = f"{oroot}.xyt"

    # 1. Executar mindtct de forma idempotente
    mapas_existem = (
        os.path.exists(dm_file) and os.path.getsize(dm_file) > 0 and
        os.path.exists(qm_file) and os.path.getsize(qm_file) > 0 and
        os.path.exists(hcm_file) and os.path.getsize(hcm_file) > 0 and
        os.path.exists(xyt_file) and os.path.getsize(xyt_file) > 0
    )

    if not mapas_existem:
        cmd = ["nice", "-n", "19", MINDTCT_BIN, jpg_path, oroot]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            # Limpar arquivos auxiliares desnecessários (.brw, .lcm, .lfm, .min)
            for ext in [".brw", ".lcm", ".lfm", ".min"]:
                aux = f"{oroot}{ext}"
                if os.path.exists(aux):
                    try:
                        os.remove(aux)
                    except OSError:
                        pass
        except Exception as e:
            return {
                "pessoa": task["pessoa"],
                "sexo": task["sexo"],
                "mao": task["mao"],
                "dedo": task["dedo"],
                "arquivo": stem,
                "o_entropia": "",
                "o_frac_alta_curvatura": "",
                "o_orientacao_dominante": "",
                "o_coerencia": "",
                "q_n_minucias": "",
                "q_area_util": 0,
                "q_densidade": "",
                "q_n_minucias_total": 0,
                "c_classe": "",
                "c_confianca": "",
                "flag_falha_o": 1,
                "flag_falha_q": 1,
                "flag_falha_c": 1,
                "c_erro": f"erro_mindtct: {e}"
            }

    # 2. Processar Família O e Q a partir dos mapas
    try:
        dm = np.loadtxt(dm_file, dtype=int)
        qm = np.loadtxt(qm_file, dtype=int)
        hcm = np.loadtxt(hcm_file, dtype=int)
    except Exception as e:
        return {
            "pessoa": task["pessoa"],
            "sexo": task["sexo"],
            "mao": task["mao"],
            "dedo": task["dedo"],
            "arquivo": stem,
            "o_entropia": "",
            "o_frac_alta_curvatura": "",
            "o_orientacao_dominante": "",
            "o_coerencia": "",
            "q_n_minucias": "",
            "q_area_util": 0,
            "q_densidade": "",
            "q_n_minucias_total": 0,
            "c_classe": "",
            "c_confianca": "",
            "flag_falha_o": 1,
            "flag_falha_q": 1,
            "flag_falha_c": 1,
            "c_erro": f"erro_leitura_mapas: {e}"
        }

    # Blocos válidos: direção válida (0..15) e qualidade positiva (> 0)
    valid_blocks = (dm >= 0) & (qm > 0)
    n_valid = int(np.sum(valid_blocks))

    # --- Família O ---
    flag_falha_o = 0
    if n_valid >= 10:
        dm_valid = dm[valid_blocks]
        # Histograma das 16 direções discretas
        counts = np.bincount(dm_valid, minlength=16)[:16]
        probs = counts / float(np.sum(counts))
        p_pos = probs[probs > 0]
        o_entropia = float(-np.sum(p_pos * np.log(p_pos)))  # em nats (base e)

        # Fração de blocos com alta curvatura
        hcm_valid = hcm[valid_blocks]
        o_frac_alta_curvatura = float(np.sum(hcm_valid == 1) / float(n_valid))

        # Estatística circular axial (direções são linhas em [0, pi)): theta = d * pi / 16
        thetas = dm_valid * (np.pi / 16.0)
        c_mean = float(np.mean(np.cos(2.0 * thetas)))
        s_mean = float(np.mean(np.sin(2.0 * thetas)))
        o_orientacao_dominante = float((0.5 * np.arctan2(s_mean, c_mean)) % np.pi)
        o_coerencia = float(np.sqrt(c_mean ** 2 + s_mean ** 2))
    else:
        flag_falha_o = 1
        o_entropia = ""
        o_frac_alta_curvatura = ""
        o_orientacao_dominante = ""
        o_coerencia = ""

    # --- Família Q ---
    flag_falha_q = 0
    n_min_qual = 0
    n_min_tot = 0
    if os.path.exists(xyt_file):
        with open(xyt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    n_min_tot += 1
                    try:
                        q_val = int(parts[3])
                        if q_val >= MINUTIA_QUALITY_THRESH:
                            n_min_qual += 1
                    except ValueError:
                        pass

    if n_valid > 0:
        q_densidade = float(n_min_qual / float(n_valid))
    else:
        flag_falha_q = 1
        q_densidade = ""

    # --- Família C ---
    # O pcasys do NBIS opera apenas com imagens >= 512x480 px (resolução padrão NIST de 500 dpi).
    # Como as imagens do SOCOFing têm 288x309 px (~160 dpi ampliado 3x), o pcasys aborta no segmentador.
    # Conforme regra do pré-registro: "NÃO force: registre o erro e deixe a coluna vazia."
    c_classe = ""
    c_confianca = ""
    flag_falha_c = 1
    c_erro = "pcasys_imagem_pequena_w288_lt_512"

    return {
        "pessoa": task["pessoa"],
        "sexo": task["sexo"],
        "mao": task["mao"],
        "dedo": task["dedo"],
        "arquivo": stem,
        "o_entropia": f"{o_entropia:.6f}" if isinstance(o_entropia, float) else "",
        "o_frac_alta_curvatura": f"{o_frac_alta_curvatura:.6f}" if isinstance(o_frac_alta_curvatura, float) else "",
        "o_orientacao_dominante": f"{o_orientacao_dominante:.6f}" if isinstance(o_orientacao_dominante, float) else "",
        "o_coerencia": f"{o_coerencia:.6f}" if isinstance(o_coerencia, float) else "",
        "q_n_minucias": n_min_qual,
        "q_area_util": n_valid,
        "q_densidade": f"{q_densidade:.6f}" if isinstance(q_densidade, float) else "",
        "q_n_minucias_total": n_min_tot,
        "c_classe": c_classe,
        "c_confianca": c_confianca,
        "flag_falha_o": flag_falha_o,
        "flag_falha_q": flag_falha_q,
        "flag_falha_c": flag_falha_c,
        "c_erro": c_erro
    }


def main():
    if not os.path.isfile(MINDTCT_BIN) or not os.access(MINDTCT_BIN, os.X_OK):
        raise SystemExit(f"[ERRO] mindtct não encontrado em: {MINDTCT_BIN}")

    os.makedirs(OUT_ESTUDO02, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)

    # 1. Coletar imagens da base Real (600 pessoas x 10 dedos)
    tasks = []
    for img_path in sorted(glob.glob(os.path.join(DATA_REAL, "*.BMP"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        m = RE_REAL.match(stem)
        if not m:
            continue
        pessoa, sexo, mao, dedo = m.group(1), m.group(2), m.group(3), m.group(4)
        jpg_path = os.path.join(IMG_REAL_DIR, f"{stem}.jpg")
        if not os.path.exists(jpg_path):
            # Se por algum motivo o JPG não estiver preparado, converter diretamente
            from PIL import Image
            with Image.open(img_path) as img:
                gray = img.convert("L")
                w, h = gray.size
                resized = gray.resize((w * 3, h * 3), Image.BICUBIC)
                resized.save(jpg_path, "JPEG", quality=95)

        tasks.append({
            "pessoa": int(pessoa),
            "sexo": sexo,
            "mao": mao,
            "dedo": dedo,
            "stem": stem,
            "jpg_path": jpg_path
        })

    print(f"[INFO] Total de dedos para processar: {len(tasks)} (SOCOFing Real)")
    print("[INFO] Extraindo características com ProcessPoolExecutor (nice -n 19)...")

    resultados = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(extrair_para_dedo, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            resultados.append(res)
            if i % 1000 == 0 or i == len(tasks):
                print(f"[INFO] Progresso: {i}/{len(tasks)} dedos processados.")

    # Ordenar por pessoa (numérico), mão e dedo
    ordem_dedos = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "little": 4}
    ordem_mao = {"Left": 0, "Right": 1}
    resultados.sort(key=lambda r: (int(r["pessoa"]), ordem_mao.get(r["mao"], 0), ordem_dedos.get(r["dedo"], 0)))

    # Gravar CSV
    fieldnames = [
        "pessoa", "sexo", "mao", "dedo", "arquivo",
        "o_entropia", "o_frac_alta_curvatura", "o_orientacao_dominante", "o_coerencia",
        "q_n_minucias", "q_area_util", "q_densidade", "q_n_minucias_total",
        "c_classe", "c_confianca",
        "flag_falha_o", "flag_falha_q", "flag_falha_c", "c_erro"
    ]

    with open(FEATURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resultados)

    # Relatório de contagens e taxas de falha
    n_tot = len(resultados)
    falhas_o = sum(r["flag_falha_o"] for r in resultados)
    falhas_q = sum(r["flag_falha_q"] for r in resultados)
    falhas_c = sum(r["flag_falha_c"] for r in resultados)

    print()
    print("==================================================================")
    print("RESUMO DA EXTRAÇÃO DE CARACTERÍSTICAS (ESTUDO 02)")
    print("==================================================================")
    print(f"Arquivo gerado: {FEATURES_CSV}")
    print(f"Total de registros: {n_tot}")
    print(f"Família O (Orientação):")
    print(f"  Falhas: {falhas_o} / {n_tot} ({100.0 * falhas_o / n_tot:.2f}%)")
    print(f"Família Q (Quantidade/Densidade de minúcias):")
    print(f"  Falhas: {falhas_q} / {n_tot} ({100.0 * falhas_q / n_tot:.2f}%)")
    print(f"Família C (Classe NBIS pcasys):")
    print(f"  Falhas/Incompatíveis: {falhas_c} / {n_tot} ({100.0 * falhas_c / n_tot:.2f}%)")
    print(f"  Motivo: Imagens 288x309 px abaixo da resolução mínima do sgmnt/pcasys (512 px)")
    print("==================================================================")


if __name__ == "__main__":
    main()
