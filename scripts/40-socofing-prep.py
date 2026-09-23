#!/usr/bin/env python3
# ==============================================================================
# scripts/40-socofing-prep.py
# Preparação da base SOCOFing para os matchers NBIS (mindtct/bozorth3) e SourceAFIS.
#
# Processamento:
# 1) Converte cada imagem BMP (96x103 px, RGBA) para escala de cinza e amplia 3x
#    com Image.BICUBIC (resolução efetiva medida ~164 dpi -> ~490 dpi).
#    Salva PNG em out/socofing/img/<real|cr>/ e JPEG q=95 (lido pelo mindtct).
# 2) Executa build/bin/mindtct (3 processos, nice -n 19, idempotente) gerando
#    out/socofing/xyt/<real|cr>/<nome>.xyt e removendo arquivos auxiliares.
# 3) Gera out/socofing/index.csv e out/socofing/index_cr.csv ordenados por sujeito
#    numérico e nome, com colunas:
#    caminho_xyt,caminho_png,sujeito,sexo,mao,dedo,n_minucias
# 4) Imprime resumo estatístico (contagem, mediana, p5, p95 e contagem < 10 minúcias).
# ==============================================================================

import os
import re
import csv
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MINDTCT = os.path.join(ROOT, "build", "bin", "mindtct")

DATA_REAL = os.path.join(ROOT, "data", "kagglehub", "datasets", "ruizgara", "socofing", "versions", "2", "SOCOFing", "Real")
DATA_CR = os.path.join(ROOT, "data", "kagglehub", "datasets", "ruizgara", "socofing", "versions", "2", "SOCOFing", "Altered", "Altered-Easy")

OUT_SOCOFING = os.path.join(ROOT, "out", "socofing")
OUT_IMG_REAL = os.path.join(OUT_SOCOFING, "img", "real")
OUT_IMG_CR = os.path.join(OUT_SOCOFING, "img", "cr")
OUT_XYT_REAL = os.path.join(OUT_SOCOFING, "xyt", "real")
OUT_XYT_CR = os.path.join(OUT_SOCOFING, "xyt", "cr")

INDEX_REAL_CSV = os.path.join(OUT_SOCOFING, "index.csv")
INDEX_CR_CSV = os.path.join(OUT_SOCOFING, "index_cr.csv")

RE_REAL = re.compile(r"^(\d+)__([MF])_(Left|Right)_(thumb|index|middle|ring|little)_finger$")
RE_CR = re.compile(r"^(\d+)__([MF])_(Left|Right)_(thumb|index|middle|ring|little)_finger_CR$")


def process_task(task):
    """Converte imagem para escala de cinza 3x (PNG + JPEG) e extrai minúcias via mindtct."""
    png_path = task["png_path"]
    jpg_path = task["jpg_path"]
    xyt_path = task["xyt_path"]
    oroot = task["oroot"]
    img_path = task["img_path"]

    # 1) Gerar PNG e JPG se não existirem
    img_needed = not (os.path.exists(png_path) and os.path.getsize(png_path) > 0 and
                      os.path.exists(jpg_path) and os.path.getsize(jpg_path) > 0)
    if img_needed:
        try:
            with Image.open(img_path) as img:
                gray = img.convert("L")
                w, h = gray.size
                resized = gray.resize((w * 3, h * 3), Image.BICUBIC)
                resized.save(png_path, "PNG")
                resized.save(jpg_path, "JPEG", quality=95)
        except Exception as e:
            return task, f"erro_img: {e}", 0

    # 2) Executar mindtct sobre o JPEG (idempotente: pula se .xyt já existe e não está vazio)
    if not (os.path.exists(xyt_path) and os.path.getsize(xyt_path) > 0):
        cmd = ["nice", "-n", "19", MINDTCT, jpg_path, oroot]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            return task, f"erro_mindtct: {e}", 0
        finally:
            # Limpar arquivos auxiliares gerados pelo mindtct, mantendo o .xyt
            for aux_ext in [".brw", ".dm", ".hcm", ".lcm", ".lfm", ".min", ".qm"]:
                aux_file = f"{oroot}{aux_ext}"
                if os.path.exists(aux_file):
                    try:
                        os.remove(aux_file)
                    except OSError:
                        pass

    # 3) Contagem de minúcias
    num_minutiae = 0
    if os.path.exists(xyt_path):
        with open(xyt_path, "r") as f:
            num_minutiae = sum(1 for line in f if line.strip())

    return task, "ok", num_minutiae


def coletar_tarefas():
    """Coleta e valida as imagens reais e CR da SOCOFing."""
    if not os.path.isdir(DATA_REAL):
        raise SystemExit(f"[ERRO] Diretório Real não encontrado: {DATA_REAL}")
    if not os.path.isdir(DATA_CR):
        raise SystemExit(f"[ERRO] Diretório Altered-Easy não encontrado: {DATA_CR}")

    tasks_real = []
    for img_path in sorted(glob.glob(os.path.join(DATA_REAL, "*.BMP"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        m = RE_REAL.match(stem)
        if not m:
            continue
        sujeito, sexo, mao, dedo = m.group(1), m.group(2), m.group(3), m.group(4)
        tasks_real.append({
            "img_path": img_path,
            "stem": stem,
            "png_path": os.path.join(OUT_IMG_REAL, f"{stem}.png"),
            "jpg_path": os.path.join(OUT_IMG_REAL, f"{stem}.jpg"),
            "xyt_path": os.path.join(OUT_XYT_REAL, f"{stem}.xyt"),
            "oroot": os.path.join(OUT_XYT_REAL, stem),
            "sujeito": sujeito,
            "sexo": sexo,
            "mao": mao,
            "dedo": dedo,
            "tipo": "real"
        })

    tasks_cr = []
    for img_path in sorted(glob.glob(os.path.join(DATA_CR, "*_CR.BMP"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        m = RE_CR.match(stem)
        if not m:
            continue
        sujeito, sexo, mao, dedo = m.group(1), m.group(2), m.group(3), m.group(4)
        tasks_cr.append({
            "img_path": img_path,
            "stem": stem,
            "png_path": os.path.join(OUT_IMG_CR, f"{stem}.png"),
            "jpg_path": os.path.join(OUT_IMG_CR, f"{stem}.jpg"),
            "xyt_path": os.path.join(OUT_XYT_CR, f"{stem}.xyt"),
            "oroot": os.path.join(OUT_XYT_CR, stem),
            "sujeito": sujeito,
            "sexo": sexo,
            "mao": mao,
            "dedo": dedo,
            "tipo": "cr"
        })

    return tasks_real, tasks_cr


def gravar_indice(tasks_com_minucias, index_path):
    """Grava o índice CSV ordenado por sujeito numérico e nome."""
    rows = []
    for task, num_minutiae in tasks_com_minucias:
        rows.append({
            "caminho_xyt": os.path.relpath(task["xyt_path"], ROOT),
            "caminho_png": os.path.relpath(task["png_path"], ROOT),
            "sujeito": task["sujeito"],
            "sexo": task["sexo"],
            "mao": task["mao"],
            "dedo": task["dedo"],
            "n_minucias": num_minutiae
        })

    rows.sort(key=lambda r: (int(r["sujeito"]), r["caminho_xyt"]))

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["caminho_xyt", "caminho_png", "sujeito", "sexo", "mao", "dedo", "n_minucias"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def estatisticas(rows):
    """Calcula mediana, p5, p95 e contagem de itens com < 10 minúcias."""
    n = len(rows)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0
    mins = np.array([r["n_minucias"] for r in rows], dtype=np.int32)
    med = float(np.median(mins))
    p5 = float(np.percentile(mins, 5))
    p95 = float(np.percentile(mins, 95))
    poucas = int((mins < 10).sum())
    return n, med, p5, p95, poucas


def main():
    if not os.path.isfile(MINDTCT) or not os.access(MINDTCT, os.X_OK):
        raise SystemExit(f"[ERRO] mindtct não encontrado ou sem permissão de execução: {MINDTCT}")

    os.makedirs(OUT_IMG_REAL, exist_ok=True)
    os.makedirs(OUT_IMG_CR, exist_ok=True)
    os.makedirs(OUT_XYT_REAL, exist_ok=True)
    os.makedirs(OUT_XYT_CR, exist_ok=True)

    tasks_real, tasks_cr = coletar_tarefas()
    todas_tarefas = tasks_real + tasks_cr

    print(f"Total de tarefas: {len(todas_tarefas)} ({len(tasks_real)} real, {len(tasks_cr)} CR)")
    print("Processando imagens e extraindo minúcias com 3 processos...")

    resultados_real = {}
    resultados_cr = {}
    erros = 0

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_task, t): t for t in todas_tarefas}
        for future in as_completed(futures):
            task, status, num_min = future.result()
            if status != "ok":
                erros += 1
                print(f"[ERRO] {task['stem']}: {status}")
            if task["tipo"] == "real":
                resultados_real[task["stem"]] = (task, num_min)
            else:
                resultados_cr[task["stem"]] = (task, num_min)

    if erros > 0:
        print(f"[AVISO] Ocorreram {erros} erros durante o processamento.")

    # Gerar índices CSV
    rows_real = gravar_indice(list(resultados_real.values()), INDEX_REAL_CSV)
    rows_cr = gravar_indice(list(resultados_cr.values()), INDEX_CR_CSV)

    # Estatísticas
    n_r, med_r, p5_r, p95_r, poucas_r = estatisticas(rows_real)
    n_c, med_c, p5_c, p95_c, poucas_c = estatisticas(rows_cr)

    print()
    print("==================================================================")
    print("RESUMO DA PREPARAÇÃO DA BASE SOCOFing")
    print("==================================================================")
    print(f"Conjunto Real:")
    print(f"  Total de imagens: {n_r}")
    print(f"  Minúcias (mediana [p5, p95]): {med_r:.1f} [{p5_r:.1f}, {p95_r:.1f}]")
    print(f"  Arquivos .xyt com < 10 minúcias: {poucas_r} ({100.0 * poucas_r / n_r:.2f}%)")
    print()
    print(f"Conjunto Altered-Easy (CR):")
    print(f"  Total de imagens: {n_c}")
    print(f"  Minúcias (mediana [p5, p95]): {med_c:.1f} [{p5_c:.1f}, {p95_c:.1f}]")
    print(f"  Arquivos .xyt com < 10 minúcias: {poucas_c} ({100.0 * poucas_c / n_c:.2f}%)")
    print("==================================================================")
    print(f"Índice Real: {INDEX_REAL_CSV}")
    print(f"Índice CR:   {INDEX_CR_CSV}")


if __name__ == "__main__":
    main()
