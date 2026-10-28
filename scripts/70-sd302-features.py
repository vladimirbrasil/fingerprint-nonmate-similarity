#!/usr/bin/env python3
# ==============================================================================
# scripts/70-sd302-features.py
# Extração de características por impressão digital para a base NIST SD302 (Estudo 03).
#
# Enumera as imagens de:
# - baseline: U, V (roll 500 ppi), R, S (slap-segmented 500 ppi)
# - challengers: A-H (roll)
# - auxiliary: K, L, M, P (flat 500 plain)
#
# Processamento por imagem:
# 1) Converte a escala de cinza ('L')
# 2) Obtém resolução original (info['dpi']) e reamostra para 500 ppi (LANCZOS)
#    se diferir mais de 1% em relação a 500.
# 3) Salva em JPEG quality=95 com dpi=(500, 500) em out/estudo03/jpg/<captura>/ (idempotente)
# 4) Executa extração de características via scripts/50-features-socofing.py
#    (função extrair_para_dedo importada dinamicamente) com mapas em
#    out/estudo03/maps/<captura>/
# 5) Inclui variáveis de qualidade e contraste reaproveitando a lógica de 50/56.
#
# Saída:
# - out/estudo03/features.csv
# ==============================================================================

import argparse
import csv
import glob
import importlib.util
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "nist-sd302")
ERRATA_PATH = os.path.join(DATA_DIR, "ERRATA_SD302.txt")
PARTICIPANTS_PATH = os.path.join(DATA_DIR, "participants.csv")
OUT_ESTUDO03 = os.path.join(ROOT, "out", "estudo03")
JPG_DIR = os.path.join(OUT_ESTUDO03, "jpg")
MAPS_DIR = os.path.join(OUT_ESTUDO03, "maps")
FEATURES_CSV = os.path.join(OUT_ESTUDO03, "features.csv")

RE_FILENAME = re.compile(r"^(\d{8})_([A-Za-z0-9]+)_(?:(\d+)_)?([a-z-]+)_(\d+)\.png$")

# Importar scripts/50-features-socofing.py
SPEC_50 = importlib.util.spec_from_file_location("feat50", os.path.join(ROOT, "scripts", "50-features-socofing.py"))
feat50 = importlib.util.module_from_spec(SPEC_50)
SPEC_50.loader.exec_module(feat50)


def carregar_errata(errata_path):
    """
    Extrai o conjunto de pares (pessoa, captura) citados na errata da SD302.
    Formatos aceitos:
    1) '... 00002361, Device V, ...' ou '... 00002420, device R ...'
    2) Stems de arquivo como '00002480_6E_...': stem cujo 2º campo seja a captura de interesse.
    """
    if not os.path.exists(errata_path):
        return set()

    with open(errata_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    exclusions = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Formato 1: Device X ou device X associado a uma pessoa de 8 dígitos
        # Ex: "00002361, Device V..." ou "segmented slap thumb ... for 00002420, device R were swapped"
        m_subj = re.search(r"\b(\d{8})\b", line)
        m_dev = re.search(r"\b[Dd]evice\s+([A-Za-z0-9]+)\b", line)
        if m_subj and m_dev:
            p = m_subj.group(1)
            dev = m_dev.group(1).upper()
            exclusions.add((p, dev))

        # Formato 2: stem de arquivo (\d{8})_([A-Za-z0-9]+)_
        m_stems = re.findall(r"\b(\d{8})_([A-Za-z0-9]+)_", line)
        for p, field2 in m_stems:
            exclusions.add((p, field2.upper()))

    return exclusions


def carregar_participantes(participants_path):
    """Carrega metadados de participants.csv (id -> {gender, age, ...})."""
    part = {}
    if not os.path.exists(participants_path):
        return part
    with open(participants_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            part[row["id"]] = row
    return part


def enumerar_arquivos(data_dir):
    """
    Enumera todos os arquivos PNG da SD302 de interesse:
    - baseline: U, V (roll 500 ppi), R, S (slap-segmented 500 ppi)
    - challengers: A-H (roll)
    - auxiliary: K, L, M, P (flat 500 plain)
    Retorna lista de tuplas: (caminho_png, captura, tipo)
    """
    itens = []
    # 1. Baseline roll: U, V (500)
    for cap in ["U", "V"]:
        pattern = os.path.join(data_dir, "extraido", "images", "baseline", cap, "500", "roll", "png", "*.png")
        for p in glob.glob(pattern):
            itens.append((p, cap, "roll"))

    # 2. Baseline slap-segmented: R, S (500)
    for cap in ["R", "S"]:
        pattern = os.path.join(data_dir, "extraido", "images", "baseline", cap, "500", "slap-segmented", "png", "*.png")
        for p in glob.glob(pattern):
            itens.append((p, cap, "slap"))

    # 3. Challengers roll: A..H
    for cap in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        pattern = os.path.join(data_dir, "extraido", "images", "challengers", cap, "roll", "png", "*.png")
        for p in glob.glob(pattern):
            itens.append((p, cap, "roll"))

    # 4. Auxiliary plain: K, L, M, P (a 302d não tem N)
    for cap in ["K", "L", "M", "P"]:
        pattern = os.path.join(data_dir, "extraido", "images", "auxiliary", "flat", cap, "500", "plain", "png", "*.png")
        for p in glob.glob(pattern):
            itens.append((p, cap, "plain"))

    return sorted(itens, key=lambda x: x[0])


def processar_imagem_individual(item_info):
    """
    Processa uma única imagem:
    1. Reamostragem se necessário e salvamento do JPEG 500 dpi.
    2. Extração de características via feat50.extrair_para_dedo.
    """
    png_path, captura, tipo, excluido_errata, sexo = item_info
    basename = os.path.basename(png_path)
    stem, _ = os.path.splitext(basename)

    m = RE_FILENAME.match(basename)
    if not m:
        return None

    pessoa = m.group(1)
    # m.group(2) é a captura
    # m.group(4) é o tipo do nome
    frgp = int(m.group(5))
    if frgp < 1 or frgp > 10:
        return None

    # Mão e dedo: 1-5 = Right thumb..little; 6-10 = Left thumb..little
    if 1 <= frgp <= 5:
        mao = "R"
        dedo = frgp
    else:
        mao = "L"
        dedo = frgp - 5

    # 1. Preparar JPEG
    cap_jpg_dir = os.path.join(JPG_DIR, captura)
    os.makedirs(cap_jpg_dir, exist_ok=True)
    jpg_path = os.path.join(cap_jpg_dir, f"{stem}.jpg")

    cap_maps_dir = os.path.join(MAPS_DIR, captura)
    os.makedirs(cap_maps_dir, exist_ok=True)

    ppi_x_orig, ppi_y_orig = 500.0, 500.0

    if not os.path.exists(jpg_path):
        with Image.open(png_path) as im:
            dpi = im.info.get("dpi")
            if dpi is not None and len(dpi) >= 2:
                ppi_x_orig = float(dpi[0])
                ppi_y_orig = float(dpi[1])

            im_gray = im.convert("L")
            scale_x = 500.0 / ppi_x_orig
            scale_y = 500.0 / ppi_y_orig

            new_w = round(im_gray.width * scale_x) if abs(ppi_x_orig - 500.0) / 500.0 > 0.01 else im_gray.width
            new_h = round(im_gray.height * scale_y) if abs(ppi_y_orig - 500.0) / 500.0 > 0.01 else im_gray.height

            if (new_w, new_h) != (im_gray.width, im_gray.height):
                resampled = im_gray.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                resampled = im_gray

            resampled.save(jpg_path, format="JPEG", quality=95, dpi=(500, 500))
    else:
        # Imagem já existe, ler dpi da original de forma rápida
        with Image.open(png_path) as im:
            dpi = im.info.get("dpi")
            if dpi is not None and len(dpi) >= 2:
                ppi_x_orig = float(dpi[0])
                ppi_y_orig = float(dpi[1])

    # 2. Configurar tarefa para extrair_para_dedo de 50
    feat50.MAPS_DIR = cap_maps_dir

    task = {
        "jpg_path": jpg_path,
        "stem": stem,
        "pessoa": pessoa,
        "sexo": sexo,
        "mao": "Right" if mao == "R" else "Left",
        "dedo": {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "little"}.get(dedo, "finger")
    }

    # Chamada idempotente à função extrair_para_dedo
    res50 = feat50.extrair_para_dedo(task)

    # 3. Montar linha de features
    # Ordem das colunas pedida:
    # captura, tipo, pessoa, frgp, mao, dedo, ppi_original_x, ppi_original_y, excluido_errata,
    # + colunas de 50 + qualidade_media, contraste
    row = {
        "captura": captura,
        "tipo": tipo,
        "pessoa": pessoa,
        "frgp": frgp,
        "mao": mao,
        "dedo": dedo,
        "ppi_original_x": f"{ppi_x_orig:.2f}",
        "ppi_original_y": f"{ppi_y_orig:.2f}",
        "excluido_errata": excluido_errata,
        "arquivo": stem,
        "o_entropia": res50.get("o_entropia", ""),
        "o_frac_alta_curvatura": res50.get("o_frac_alta_curvatura", ""),
        "o_orientacao_dominante": res50.get("o_orientacao_dominante", ""),
        "o_coerencia": res50.get("o_coerencia", ""),
        "q_n_minucias": res50.get("q_n_minucias", ""),
        "q_area_util": res50.get("q_area_util", 0),
        "q_densidade": res50.get("q_densidade", ""),
        "q_n_minucias_total": res50.get("q_n_minucias_total", 0),
        "c_classe": res50.get("c_classe", ""),
        "c_confianca": res50.get("c_confianca", ""),
        "flag_falha_o": res50.get("flag_falha_o", 1),
        "flag_falha_q": res50.get("flag_falha_q", 1),
        "flag_falha_c": res50.get("flag_falha_c", 1),
        "c_erro": res50.get("c_erro", ""),
        "qualidade_media": res50.get("qualidade_media", ""),
        "contraste": res50.get("contraste", "")
    }

    return row


def main():
    parser = argparse.ArgumentParser(description="Extração de features para NIST SD302 (Estudo 03)")
    parser.add_argument("--limite", type=int, default=None, help="Limite de imagens a processar por captura (para testes)")
    parser.add_argument("--workers", type=int, default=3, help="Número de workers paralelos (default=3)")
    args = parser.parse_args()

    # Aplica nice ao processo atual
    try:
        os.nice(19)
    except Exception:
        pass

    os.makedirs(JPG_DIR, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)

    errata = carregar_errata(ERRATA_PATH)
    participantes = carregar_participantes(PARTICIPANTS_PATH)

    todos_itens = enumerar_arquivos(DATA_DIR)
    print(f"[INFO] Total de imagens encontradas: {len(todos_itens)}")

    # Agrupar por captura para aplicar limite se solicitado
    itens_por_cap = {}
    for p, cap, tipo in todos_itens:
        itens_por_cap.setdefault(cap, []).append((p, cap, tipo))

    itens_selecionados = []
    for cap, lista in sorted(itens_por_cap.items()):
        if args.limite is not None:
            lista = lista[:args.limite]
        itens_selecionados.extend(lista)

    print(f"[INFO] Total de imagens a processar: {len(itens_selecionados)} (limite={args.limite})")

    # Montar lista de tarefas para os workers
    tarefas = []
    for p, cap, tipo in itens_selecionados:
        basename = os.path.basename(p)
        m = RE_FILENAME.match(basename)
        if not m:
            continue
        pessoa = m.group(1)
        excluido = 1 if (pessoa, cap.upper()) in errata else 0
        part_info = participantes.get(pessoa, {})
        sexo = part_info.get("gender", "")
        tarefas.append((p, cap, tipo, excluido, sexo))

    colunas = [
        "captura", "tipo", "pessoa", "frgp", "mao", "dedo",
        "ppi_original_x", "ppi_original_y", "excluido_errata",
        "arquivo", "o_entropia", "o_frac_alta_curvatura", "o_orientacao_dominante",
        "o_coerencia", "q_n_minucias", "q_area_util", "q_densidade", "q_n_minucias_total",
        "c_classe", "c_confianca", "flag_falha_o", "flag_falha_q", "flag_falha_c",
        "c_erro", "qualidade_media", "contraste"
    ]

    linhas_resultado = []
    total = len(tarefas)
    concluidos = 0

    print(f"[INFO] Processando com ProcessPoolExecutor(max_workers={args.workers})...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futuros = {executor.submit(processar_imagem_individual, t): t for t in tarefas}
        for fut in as_completed(futuros):
            res = fut.result()
            if res is not None:
                linhas_resultado.append(res)
            concluidos += 1
            if concluidos % 50 == 0 or concluidos == total:
                print(f"[PROGRESSO] {concluidos}/{total} imagens processadas ({100*concluidos/total:.1f}%)")

    # Ordenar resultados por captura, pessoa, frgp
    linhas_resultado.sort(key=lambda r: (r["captura"], r["pessoa"], int(r["frgp"])))

    with open(FEATURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        writer.writerows(linhas_resultado)

    print(f"[INFO] Concluído! Gravado em: {FEATURES_CSV}")


if __name__ == "__main__":
    main()
