#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# scripts/20-extract.sh
# Extração de minúcias com mindtct (3 processos, nice -n 19, idempotente).
# Converte imagens para JPEG grayscale e gera out/xyt/<base>/<arquivo>.xyt
# Cria out/index.csv com caminho_xyt, base, id_dedo, impressao.
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${REPO_ROOT}/build/bin"
MINDTCT="${BIN_DIR}/mindtct"
DATA_FVC="${REPO_ROOT}/data/fvc"
OUT_XYT="${REPO_ROOT}/out/xyt"
INDEX_CSV="${REPO_ROOT}/out/index.csv"

if [[ ! -x "${MINDTCT}" ]]; then
    echo "[ERRO] mindtct não encontrado em ${MINDTCT}. Execute scripts/00-build-nbis.sh primeiro."
    exit 1
fi

mkdir -p "${OUT_XYT}"

echo "=================================================================="
echo "EXTRAÇÃO DE MINÚCIAS (MINDTCT) - 3 NÚCLEOS (nice -n 19)"
echo "=================================================================="

python3 - << 'EOF'
import os
import re
import csv
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(), "."))
DATA_FVC = os.path.join(REPO_ROOT, "data", "fvc")
OUT_XYT = os.path.join(REPO_ROOT, "out", "xyt")
INDEX_CSV = os.path.join(REPO_ROOT, "out", "index.csv")
MINDTCT = os.path.join(REPO_ROOT, "build", "bin", "mindtct")

# Coletar todas as imagens das bases FVC
image_tasks = []
bases = sorted(os.listdir(DATA_FVC))
bases = [b for b in bases if b.startswith("FVC") and os.path.isdir(os.path.join(DATA_FVC, b))]

for base in bases:
    base_dir = os.path.join(DATA_FVC, base)
    for img_path in sorted(glob.glob(os.path.join(base_dir, "*.*"))):
        ext = os.path.splitext(img_path)[1].lower()
        if ext in [".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg"]:
            fname = os.path.basename(img_path)
            stem = os.path.splitext(fname)[0]
            out_base_dir = os.path.join(OUT_XYT, base)
            os.makedirs(out_base_dir, exist_ok=True)
            xyt_path = os.path.join(out_base_dir, f"{stem}.xyt")
            
            # Parse id_dedo e impressao
            m = re.match(r'^(\d+)_(\d+)$', stem)
            if m:
                id_dedo, impressao = m.group(1), m.group(2)
            else:
                id_dedo, impressao = stem, "0"
            
            image_tasks.append({
                "img_path": img_path,
                "base": base,
                "stem": stem,
                "out_base_dir": out_base_dir,
                "xyt_path": xyt_path,
                "id_dedo": id_dedo,
                "impressao": impressao
            })

print(f"Total de imagens encontradas: {len(image_tasks)}")

def process_image(task):
    xyt_path = task["xyt_path"]
    stem = task["stem"]
    out_base_dir = task["out_base_dir"]
    img_path = task["img_path"]
    oroot = os.path.join(out_base_dir, stem)

    # Idempotência: verificar se xyt existe e não está vazio
    if os.path.exists(xyt_path) and os.path.getsize(xyt_path) > 0:
        return task, "skipped", 0

    # Conversão de imagem para JPEG 8-bit grayscale temporário
    tmp_jpg = os.path.join(out_base_dir, f".tmp_{stem}.jpg")
    try:
        with Image.open(img_path) as img:
            gray = img.convert("L")
            gray.save(tmp_jpg, "JPEG", quality=100)
    except Exception as e:
        return task, f"error_img: {e}", 0

    # Execução do mindtct com prioridade baixa (nice -n 19)
    cmd = ["nice", "-n", "19", MINDTCT, tmp_jpg, oroot]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp_jpg):
            os.remove(tmp_jpg)
        return task, f"error_mindtct: {e}", 0
    finally:
        if os.path.exists(tmp_jpg):
            os.remove(tmp_jpg)
        # Limpar arquivos auxiliares gerados pelo mindtct, mantendo .xyt
        for aux_ext in [".brw", ".dm", ".hcm", ".lcm", ".lfm", ".min", ".qm"]:
            aux_file = f"{oroot}{aux_ext}"
            if os.path.exists(aux_file):
                os.remove(aux_file)

    num_minutiae = sum(1 for _ in open(xyt_path)) if os.path.exists(xyt_path) else 0
    return task, "extracted", num_minutiae

# Executar com 3 processos
num_extracted = 0
num_skipped = 0
num_errors = 0

with ProcessPoolExecutor(max_workers=3) as executor:
    futures = {executor.submit(process_image, task): task for task in image_tasks}
    for future in as_completed(futures):
        task, status, minutiae = future.result()
        if status == "extracted":
            num_extracted += 1
        elif status == "skipped":
            num_skipped += 1
        else:
            num_errors += 1
            print(f"[ERRO] {task['base']}/{task['stem']}: {status}")

print(f"Extração concluída: {num_extracted} extraídas, {num_skipped} puladas (já existiam), {num_errors} erros.")

# Gerar out/index.csv
index_rows = []
for task in image_tasks:
    # Usar caminho relativo normalizado (ex: out/xyt/FVC2000_DB1_B/101_1.xyt)
    rel_xyt = os.path.relpath(task["xyt_path"], REPO_ROOT)
    index_rows.append({
        "caminho_xyt": rel_xyt,
        "base": task["base"],
        "id_dedo": task["id_dedo"],
        "impressao": task["impressao"]
    })

# Ordenar por base, id_dedo, impressao
index_rows.sort(key=lambda r: (r["base"], int(r["id_dedo"]) if r["id_dedo"].isdigit() else r["id_dedo"], int(r["impressao"]) if r["impressao"].isdigit() else r["impressao"]))

with open(INDEX_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["caminho_xyt", "base", "id_dedo", "impressao"])
    writer.writeheader()
    writer.writerows(index_rows)

print(f"Índice gerado em {INDEX_CSV} com {len(index_rows)} registros.")
EOF

echo "=================================================================="
echo "EXTRAÇÃO CONCLUÍDA COM SUCESSO"
echo "=================================================================="
