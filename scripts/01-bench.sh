#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# scripts/01-bench.sh
# Benchmark de extração de minúcias (mindtct) e comparação 1:1 (bozorth3)
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${REPO_ROOT}/build/bin"
SAMPLE_DIR="${REPO_ROOT}/data/sample"
OUT_DIR="${REPO_ROOT}/out/bench"

MINDTCT="${BIN_DIR}/mindtct"
BOZORTH3="${BIN_DIR}/bozorth3"

if [[ ! -x "${MINDTCT}" || ! -x "${BOZORTH3}" ]]; then
    echo "[ERRO] Binários não encontrados em ${BIN_DIR}. Execute scripts/00-build-nbis.sh primeiro."
    exit 1
fi

if [[ ! -d "${SAMPLE_DIR}" || -z "$(ls -A "${SAMPLE_DIR}" 2>/dev/null)" ]]; then
    echo "[ERRO] Nenhuma imagem de amostra encontrada em ${SAMPLE_DIR}."
    exit 1
fi

mkdir -p "${OUT_DIR}"

python3 - << 'EOF'
import os
import sys
import time
import glob
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(), "."))
BIN_DIR = os.path.join(REPO_ROOT, "build", "bin")
SAMPLE_DIR = os.path.join(REPO_ROOT, "data", "sample")
OUT_DIR = os.path.join(REPO_ROOT, "out", "bench")

mindtct_bin = os.path.join(BIN_DIR, "mindtct")
bozorth3_bin = os.path.join(BIN_DIR, "bozorth3")

sample_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.wsq")))
if not sample_files:
    # Caso haja outros formatos
    sample_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.*")))

num_samples = len(sample_files)
print("=" * 65)
print("BENCHMARK NBIS: MINDTCT + BOZORTH3")
print("=" * 65)
print(f"Total de imagens na amostra: {num_samples}")
print("")

# ----------------------------------------------------------------------
# 1. Benchmark MINDTCT (Extração de Minúcias)
# ----------------------------------------------------------------------
print("[1/2] Executando MINDTCT em cada imagem da amostra...")
xyt_files = []
mindtct_times = []

for i, img_path in enumerate(sample_files, 1):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_root = os.path.join(OUT_DIR, base_name)
    xyt_path = f"{out_root}.xyt"
    
    t0 = time.perf_counter()
    subprocess.run([mindtct_bin, img_path, out_root], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t1 = time.perf_counter()
    
    elapsed_ms = (t1 - t0) * 1000.0
    mindtct_times.append(elapsed_ms)
    xyt_files.append(xyt_path)
    
    # Contar minúcias
    num_minutiae = sum(1 for _ in open(xyt_path)) if os.path.exists(xyt_path) else 0
    print(f"  [{i:02d}/{num_samples:02d}] {os.path.basename(img_path):20s} -> {elapsed_ms:6.1f} ms | minúcias: {num_minutiae}")

total_mindtct_time_s = sum(mindtct_times) / 1000.0
avg_mindtct_ms = sum(mindtct_times) / len(mindtct_times)

print(f"\nTempo total MINDTCT : {total_mindtct_time_s:.3f} s")
print(f"Média MINDTCT       : {avg_mindtct_ms:.2f} ms/imagem ({1000.0 / avg_mindtct_ms:.2f} imagens/s por núcleo)")
print("")

# ----------------------------------------------------------------------
# 2. Benchmark BOZORTH3 (Comparação 1:1)
# ----------------------------------------------------------------------
print("[2/2] Gerando pares de comparação para BOZORTH3...")
# Gerar pares combinando as imagens da amostra até >= 1000 pares
pairs = []
for f1 in xyt_files:
    for f2 in xyt_files:
        pairs.append((f1, f2))

# Multiplicar lista até atingir pelo menos 1000 pares
while len(pairs) < 1000:
    pairs.extend(pairs)
pairs = pairs[:1000]
target_pairs_count = len(pairs)

mates_file_path = os.path.join(OUT_DIR, "mates_bench.lis")
with open(mates_file_path, "w") as f:
    for p1, p2 in pairs:
        f.write(f"{p1}\n{p2}\n")

print(f"Lista de pares criada em {mates_file_path} com {target_pairs_count} pares.")
print("Executando BOZORTH3 em 1 núcleo...")

t0 = time.perf_counter()
res = subprocess.run([bozorth3_bin, "-M", mates_file_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
t1 = time.perf_counter()

bozorth3_elapsed_s = t1 - t0
pairs_per_sec_1core = target_pairs_count / bozorth3_elapsed_s
avg_bozorth3_ms_pair = (bozorth3_elapsed_s / target_pairs_count) * 1000.0

print(f"Tempo total BOZORTH3: {bozorth3_elapsed_s:.3f} s ({target_pairs_count} pares)")
print(f"Taxa por núcleo     : {pairs_per_sec_1core:.2f} pares/segundo ({avg_bozorth3_ms_pair:.3f} ms/par)")
print("")

# ----------------------------------------------------------------------
# 3. Projeção para 18.000.000 de pares em 3 núcleos
# ----------------------------------------------------------------------
total_pairs_target = 18_000_000
num_cores_proj = 3
throughput_3cores = pairs_per_sec_1core * num_cores_proj
time_seconds_proj = total_pairs_target / throughput_3cores

hours = int(time_seconds_proj // 3600)
minutes = int((time_seconds_proj % 3600) // 60)
seconds = time_seconds_proj % 60

print("=" * 65)
print("RESULTADO")
print("=" * 65)
print(f"MINDTCT  : {avg_mindtct_ms:.2f} ms por extração ({1000.0 / avg_mindtct_ms:.2f} extrações/s por núcleo)")
print(f"BOZORTH3 : {pairs_per_sec_1core:.2f} pares/segundo por núcleo ({avg_bozorth3_ms_pair:.3f} ms/par)")
print(f"PROJEÇÃO : 18.000.000 de pares em {num_cores_proj} núcleos:")
print(f"           Taxa combinada : {throughput_3cores:.2f} pares/segundo")
print(f"           Tempo estimado : {time_seconds_proj:.1f} s ({hours}h {minutes}m {seconds:.1f}s)")
print("=" * 65)
EOF
