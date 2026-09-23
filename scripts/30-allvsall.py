#!/usr/bin/env python3
import os
import sys
import csv
import json
import time
import heapq
import subprocess
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================================================================
# scripts/30-allvsall.py
# Comparação todos-contra-todos usando bozorth3 (-M) em 3 núcleos (nice -n 19).
# - Separação genuíno vs impostor
# - Histograma com bins de 1 para genuínos e impostores
# - Top 10.000 pares impostores mantidos em min-heap
# - Checkpointing por blocos em out/state.json para retomada transparente
# - Saídas: out/hist_genuine.csv, out/hist_impostor.csv, out/top_impostors.csv,
#           out/summary.json
# ==============================================================================

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN_DIR = os.path.join(REPO_ROOT, "build", "bin")
BOZORTH3 = os.path.join(BIN_DIR, "bozorth3")
OUT_DIR = os.path.join(REPO_ROOT, "out")
INDEX_CSV = os.path.join(OUT_DIR, "index.csv")
STATE_JSON = os.path.join(OUT_DIR, "state.json")
TMP_DIR = os.path.join(OUT_DIR, ".tmp_bozorth3")

HIST_GENUINE_CSV = os.path.join(OUT_DIR, "hist_genuine.csv")
HIST_IMPOSTOR_CSV = os.path.join(OUT_DIR, "hist_impostor.csv")
TOP_IMPOSTORS_CSV = os.path.join(OUT_DIR, "top_impostors.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "summary.json")

CHUNK_SIZE = 10000
TOP_K_IMPOSTORS = 10000
NUM_WORKERS = 3


def load_index():
    if not os.path.exists(INDEX_CSV):
        raise FileNotFoundError(f"Arquivo de índice não encontrado: {INDEX_CSV}. Execute scripts/20-extract.sh primeiro.")
    with open(INDEX_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
    return records


def generate_pairs(records):
    n = len(records)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


def worker_process_chunk(chunk_idx, chunk_pairs, records, repo_root, bozorth3_bin, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    lis_file = os.path.join(tmp_dir, f"mates_{chunk_idx}_{os.getpid()}_{time.time_ns()}.lis")
    
    # Criar lista de pares para o modo lote (-M) do bozorth3
    with open(lis_file, "w", encoding="utf-8") as f:
        for i, j in chunk_pairs:
            p1 = os.path.join(repo_root, records[i]["caminho_xyt"])
            p2 = os.path.join(repo_root, records[j]["caminho_xyt"])
            f.write(f"{p1}\n{p2}\n")

    cmd = ["nice", "-n", "19", bozorth3_bin, "-M", lis_file]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
        scores_raw = res.stdout.strip().splitlines()
        scores = [int(line.strip()) for line in scores_raw if line.strip()]
    finally:
        if os.path.exists(lis_file):
            os.remove(lis_file)

    if len(scores) != len(chunk_pairs):
        raise RuntimeError(f"Erro no chunk {chunk_idx}: esperado {len(chunk_pairs)} escores, recebido {len(scores)}")

    local_hist_gen = collections.defaultdict(int)
    local_hist_imp = collections.defaultdict(int)
    local_top_imp = []  # min-heap de tamanho até TOP_K_IMPOSTORS: (score, path1, path2)

    for (i, j), score in zip(chunk_pairs, scores):
        r1 = records[i]
        r2 = records[j]
        is_genuine = (r1["base"] == r2["base"] and r1["id_dedo"] == r2["id_dedo"])
        
        if is_genuine:
            local_hist_gen[score] += 1
        else:
            local_hist_imp[score] += 1
            p1 = r1["caminho_xyt"]
            p2 = r2["caminho_xyt"]
            item = (score, p1, p2)
            if len(local_top_imp) < TOP_K_IMPOSTORS:
                heapq.heappush(local_top_imp, item)
            elif score > local_top_imp[0][0]:
                heapq.heapreplace(local_top_imp, item)

    return chunk_idx, dict(local_hist_gen), dict(local_hist_imp), local_top_imp, len(chunk_pairs)


def compute_percentiles(hist_dict, percentiles_list):
    total = sum(hist_dict.values())
    if total == 0:
        return {f"p{p}": 0 for p in percentiles_list}
    
    sorted_scores = sorted(hist_dict.keys())
    results = {}
    
    for p in percentiles_list:
        target = total * (p / 100.0)
        acc = 0
        p_val = sorted_scores[-1]
        for s in sorted_scores:
            acc += hist_dict[s]
            if acc >= target:
                p_val = s
                break
        key_name = f"p{p}" if p == int(p) else f"p{p}"
        results[key_name] = p_val
        
    return results


def save_checkpoint(state_data):
    tmp_path = f"{STATE_JSON}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state_data, f)
    os.replace(tmp_path, STATE_JSON)


def main():
    if not os.path.exists(BOZORTH3):
        print(f"[ERRO] bozorth3 não encontrado em {BOZORTH3}. Execute scripts/00-build-nbis.sh primeiro.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    records = load_index()
    num_images = len(records)
    distinct_fingers = set((r["base"], r["id_dedo"]) for r in records)
    num_fingers = len(distinct_fingers)

    all_pairs = generate_pairs(records)
    total_pairs = len(all_pairs)

    chunks = [all_pairs[i : i + CHUNK_SIZE] for i in range(0, total_pairs, CHUNK_SIZE)]
    num_chunks = len(chunks)

    print("=" * 65)
    print("PIPELINE TODOS-CONTRA-TODOS: BOZORTH3 (3 PROCESSOS, nice -n 19)")
    print("=" * 65)
    print(f"Total de imagens         : {num_images}")
    print(f"Total de dedos distintos : {num_fingers}")
    print(f"Total de pares a avaliar : {total_pairs:,} em {num_chunks} blocos")
    print("")

    # Carregar estado anterior se existir
    completed_chunks = set()
    global_hist_gen = collections.defaultdict(int)
    global_hist_imp = collections.defaultdict(int)
    global_top_imp = []  # min-heap de tuplas (score, p1, p2)
    prior_elapsed = 0.0

    if os.path.exists(STATE_JSON):
        try:
            with open(STATE_JSON, "r", encoding="utf-8") as f:
                state = json.load(f)
            completed_chunks = set(state.get("completed_chunks", []))
            for k, v in state.get("hist_genuine", {}).items():
                global_hist_gen[int(k)] = v
            for k, v in state.get("hist_impostor", {}).items():
                global_hist_imp[int(k)] = v
            for item in state.get("top_impostors", []):
                heapq.heappush(global_top_imp, tuple(item))
            prior_elapsed = float(state.get("elapsed_time_s", 0.0))
            print(f"[CHECKPOINT] Retomando a partir de {len(completed_chunks)}/{num_chunks} blocos já concluídos.")
        except Exception as e:
            print(f"[AVISO] Falha ao ler checkpoint ({e}). Reiniciando do zero.")
            completed_chunks = set()
            global_hist_gen = collections.defaultdict(int)
            global_hist_imp = collections.defaultdict(int)
            global_top_imp = []
            prior_elapsed = 0.0

    pending_chunk_indices = [idx for idx in range(num_chunks) if idx not in completed_chunks]

    t_start = time.perf_counter()
    pairs_processed_session = 0

    if pending_chunk_indices:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_chunk = {
                executor.submit(
                    worker_process_chunk,
                    idx,
                    chunks[idx],
                    records,
                    REPO_ROOT,
                    BOZORTH3,
                    TMP_DIR
                ): idx
                for idx in pending_chunk_indices
            }

            for future in as_completed(future_to_chunk):
                chunk_idx, local_gen, local_imp, local_top, chunk_len = future.result()
                
                # Merge histogramas
                for score, count in local_gen.items():
                    global_hist_gen[score] += count
                for score, count in local_imp.items():
                    global_hist_imp[score] += count

                # Merge top impostores no heap global
                for item in local_top:
                    if len(global_top_imp) < TOP_K_IMPOSTORS:
                        heapq.heappush(global_top_imp, item)
                    elif item[0] > global_top_imp[0][0]:
                        heapq.heapreplace(global_top_imp, item)

                completed_chunks.add(chunk_idx)
                pairs_processed_session += chunk_len

                current_session_elapsed = time.perf_counter() - t_start
                total_elapsed = prior_elapsed + current_session_elapsed
                
                # Salvar checkpoint
                state_data = {
                    "completed_chunks": sorted(list(completed_chunks)),
                    "hist_genuine": {str(k): v for k, v in global_hist_gen.items()},
                    "hist_impostor": {str(k): v for k, v in global_hist_imp.items()},
                    "top_impostors": [list(item) for item in global_top_imp],
                    "elapsed_time_s": round(total_elapsed, 2),
                    "total_pairs_processed": sum(len(chunks[i]) for i in completed_chunks)
                }
                save_checkpoint(state_data)

                total_done_pairs = sum(len(chunks[i]) for i in completed_chunks)
                pct = (total_done_pairs / total_pairs) * 100.0
                rate = pairs_processed_session / current_session_elapsed if current_session_elapsed > 0 else 0
                remaining_pairs = total_pairs - total_done_pairs
                eta_s = remaining_pairs / rate if rate > 0 else 0
                print(f"  [Bloco {len(completed_chunks):02d}/{num_chunks:02d}] {total_done_pairs:,}/{total_pairs:,} pares ({pct:5.1f}%) | {rate:6.1f} pares/s | ETA: {eta_s:5.1f}s", flush=True)

    total_time_spent = prior_elapsed + (time.perf_counter() - t_start if pending_chunk_indices else 0)

    # 1. Gerar out/hist_genuine.csv
    with open(HIST_GENUINE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "count"])
        max_gen_score = max(global_hist_gen.keys()) if global_hist_gen else 0
        for s in range(max_gen_score + 1):
            writer.writerow([s, global_hist_gen.get(s, 0)])

    # 2. Gerar out/hist_impostor.csv
    with open(HIST_IMPOSTOR_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "count"])
        max_imp_score = max(global_hist_imp.keys()) if global_hist_imp else 0
        for s in range(max_imp_score + 1):
            writer.writerow([s, global_hist_imp.get(s, 0)])

    # 3. Gerar out/top_impostors.csv (ordenado descrescente pelo escore)
    sorted_top_impostors = sorted(global_top_imp, key=lambda x: x[0], reverse=True)
    with open(TOP_IMPOSTORS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "caminho_xyt_1", "caminho_xyt_2"])
        for score, p1, p2 in sorted_top_impostors:
            writer.writerow([score, p1, p2])

    # 4. Calcular métricas e percentis
    num_genuine_pairs = sum(global_hist_gen.values())
    num_impostor_pairs = sum(global_hist_imp.values())
    impostor_percentiles = compute_percentiles(global_hist_imp, [50, 75, 90, 95, 99, 99.9, 99.99])
    genuine_percentiles = compute_percentiles(global_hist_gen, [50, 75, 90, 95, 99, 99.9, 99.99])

    impostors_ge_40 = sum(c for s, c in global_hist_imp.items() if s >= 40)
    impostors_ge_100 = sum(c for s, c in global_hist_imp.items() if s >= 100)
    genuine_ge_40 = sum(c for s, c in global_hist_gen.items() if s >= 40)
    genuine_ge_100 = sum(c for s, c in global_hist_gen.items() if s >= 100)

    summary_data = {
        "num_images": num_images,
        "num_fingers": num_fingers,
        "num_genuine_pairs": num_genuine_pairs,
        "num_impostor_pairs": num_impostor_pairs,
        "num_total_pairs": num_genuine_pairs + num_impostor_pairs,
        "elapsed_time_seconds": round(total_time_spent, 2),
        "impostor_score_max": max(global_hist_imp.keys()) if global_hist_imp else 0,
        "impostor_score_percentiles": impostor_percentiles,
        "impostors_ge_40": impostors_ge_40,
        "impostors_ge_100": impostors_ge_100,
        "genuine_score_max": max(global_hist_gen.keys()) if global_hist_gen else 0,
        "genuine_score_percentiles": genuine_percentiles,
        "genuine_ge_40": genuine_ge_40,
        "genuine_ge_100": genuine_ge_100
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Limpar diretório temporário se vazio
    try:
        if os.path.exists(TMP_DIR) and not os.listdir(TMP_DIR):
            os.rmdir(TMP_DIR)
    except Exception:
        pass

    print("")
    print("=" * 65)
    print("RESUMO DA COMPARAÇÃO TODOS-CONTRA-TODOS")
    print("=" * 65)
    print(json.dumps(summary_data, indent=2))
    print("=" * 65)


if __name__ == "__main__":
    main()
