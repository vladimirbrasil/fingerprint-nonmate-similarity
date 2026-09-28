#!/usr/bin/env python3
# ==============================================================================
# scripts/56-qualidade.py
# Extração de variáveis de qualidade por impressão digital (Desvio D4a).
#
# Lê os mapas já extraídos pelo mindtct em out/estudo02/maps/ e as imagens em
# out/socofing/img/real/ para calcular de forma rápida e idempotente:
# 1) qualidade_media: média do mapa de qualidade .qm nos blocos válidos (dm >= 0 e qm > 0)
# 2) area_util: número de blocos válidos (mesma contagem de q_area_util)
# 3) contraste: desvio-padrão dos níveis de cinza da imagem original dentro da área útil
#
# Saída:
# - out/estudo02/features_qualidade.csv
# ==============================================================================

import os
import csv
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_CSV = os.path.join(ROOT, "out", "estudo02", "features.csv")
MAPS_DIR = os.path.join(ROOT, "out", "estudo02", "maps")
IMG_REAL_DIR = os.path.join(ROOT, "out", "socofing", "img", "real")
OUT_QUALIDADE_CSV = os.path.join(ROOT, "out", "estudo02", "features_qualidade.csv")


def processar_registro(row):
    stem = row["arquivo"]
    qm_file = os.path.join(MAPS_DIR, f"{stem}.qm")
    dm_file = os.path.join(MAPS_DIR, f"{stem}.dm")
    jpg_file = os.path.join(IMG_REAL_DIR, f"{stem}.jpg")

    if not (os.path.exists(qm_file) and os.path.exists(dm_file) and os.path.exists(jpg_file)):
        row_copy = dict(row)
        row_copy["qualidade_media"] = ""
        row_copy["area_util"] = "0"
        row_copy["contraste"] = ""
        return row_copy

    try:
        qm = np.loadtxt(qm_file, dtype=int)
        dm = np.loadtxt(dm_file, dtype=int)
        valid_blocks = (dm >= 0) & (qm > 0)
        n_valid = int(np.sum(valid_blocks))

        if n_valid > 0:
            qual_med = float(np.mean(qm[valid_blocks]))
            img = np.array(Image.open(jpg_file).convert("L"))
            mask = np.repeat(np.repeat(valid_blocks, 8, axis=0), 8, axis=1)[:img.shape[0], :img.shape[1]]
            valid_pixels = img[mask]
            contraste = float(np.std(valid_pixels)) if len(valid_pixels) > 0 else 0.0
        else:
            qual_med = 0.0
            contraste = 0.0

        row_copy = dict(row)
        row_copy["qualidade_media"] = f"{qual_med:.6f}"
        row_copy["area_util"] = str(n_valid)
        row_copy["contraste"] = f"{contraste:.6f}"
        return row_copy

    except Exception as e:
        row_copy = dict(row)
        row_copy["qualidade_media"] = ""
        row_copy["area_util"] = "0"
        row_copy["contraste"] = ""
        return row_copy


def main():
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Arquivo base não encontrado: {FEATURES_CSV}")

    with open(FEATURES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print(f"[INFO] Lendo {len(rows)} registros de {FEATURES_CSV}...")

    # Adicionar novos campos se não existirem
    novos_campos = ["qualidade_media", "area_util", "contraste"]
    for c in novos_campos:
        if c not in fieldnames:
            fieldnames.append(c)

    print(f"[INFO] Calculando qualidade_media, area_util e contraste com ProcessPoolExecutor...")
    resultados = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(processar_registro, r): idx for idx, r in enumerate(rows)}
        for i, fut in enumerate(as_completed(futures), 1):
            resultados.append(fut.result())
            if i % 1000 == 0 or i == len(rows):
                print(f"[INFO] Processados {i}/{len(rows)} dedos...")

    # Ordenar pelos mesmos critérios originais
    ordem_dedos = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "little": 4}
    ordem_mao = {"Left": 0, "Right": 1}
    resultados.sort(key=lambda r: (int(r["pessoa"]), ordem_mao.get(r["mao"], 0), ordem_dedos.get(r["dedo"], 0)))

    with open(OUT_QUALIDADE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resultados)

    print()
    print("==================================================================")
    print("EXTRAÇÃO DE VARIÁVEIS DE QUALIDADE CONCLUÍDA (Desvio D4a)")
    print("==================================================================")
    print(f"Arquivo gerado: {OUT_QUALIDADE_CSV}")
    print(f"Total de registros: {len(resultados)}")
    print(f"Campos adicionados: {novos_campos}")
    print("==================================================================")


if __name__ == "__main__":
    main()
