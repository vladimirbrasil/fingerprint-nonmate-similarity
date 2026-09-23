#!/usr/bin/env python3
# bozorth3 sonda x galeria para qualquer índice com coluna caminho_xyt (e sujeito).
# Uso: 41-nbis-rect.py <galeria.csv> <sonda.csv|-> <saida.i32> [limite_sujeito]
# Sonda "-" = galeria contra ela mesma (só triângulo superior; diagonal e inferior = -1).
# Saída: int32 LE, linhas = sonda, colunas = galeria.
import os, sys, csv, tempfile, subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BOZ = os.path.join(ROOT, "build", "bin", "bozorth3")

def ler(csv_path, limite):
    with open(csv_path) as f:
        return [os.path.join(ROOT, r["caminho_xyt"]) for r in csv.DictReader(f)
                if limite <= 0 or int(r["sujeito"]) <= limite]

def linha(args):
    i, sonda, galeria, inicio = args
    alvo = galeria[inicio:]
    if not alvo:
        return i, []
    with tempfile.NamedTemporaryFile("w", suffix=".lis", delete=False) as f:
        for g in alvo:
            f.write(f"{sonda}\n{g}\n")
        lis = f.name
    try:
        res = subprocess.run(["nice", "-n", "19", BOZ, "-M", lis], capture_output=True, text=True, check=True)
    finally:
        os.remove(lis)
    s = [int(x) for x in res.stdout.split()]
    assert len(s) == len(alvo), (i, len(s), len(alvo))
    return i, s

def main():
    gal_csv, son_csv, saida = sys.argv[1:4]
    limite = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    galeria = ler(gal_csv, limite)
    mesma = son_csv == "-"
    sondas = galeria if mesma else ler(son_csv, limite)
    m = np.full((len(sondas), len(galeria)), -1, dtype=np.int32)
    tarefas = [(i, p, galeria, i + 1 if mesma else 0) for i, p in enumerate(sondas)]
    with ProcessPoolExecutor(3) as ex:
        for i, s in ex.map(linha, tarefas, chunksize=4):
            m[i, len(galeria) - len(s):] = s
    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    m.astype("<i4").tofile(saida)
    print(f"{len(sondas)}x{len(galeria)} -> {saida}")

if __name__ == "__main__":
    main()
