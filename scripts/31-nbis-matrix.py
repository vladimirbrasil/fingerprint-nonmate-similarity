#!/usr/bin/env python3
# Matriz completa NxN de escores bozorth3 (mesma ordem de out/index.csv), para comparar
# par a par com o SourceAFIS. O 30-allvsall.py guarda só histograma + top-10k.
# Saída: out/nbis/matrix.i32 (int32 little-endian, simétrica, diagonal = -1).
import os, csv, subprocess, tempfile
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BOZ = os.path.join(ROOT, "build", "bin", "bozorth3")
OUT = os.path.join(ROOT, "out", "nbis")

def linha(args):
    i, paths = args
    n = len(paths)
    if i == n - 1:
        return i, []
    with tempfile.NamedTemporaryFile("w", suffix=".lis", delete=False) as f:
        for j in range(i + 1, n):
            f.write(f"{paths[i]}\n{paths[j]}\n")
        lis = f.name
    try:
        res = subprocess.run(["nice", "-n", "19", BOZ, "-M", lis], capture_output=True, text=True, check=True)
    finally:
        os.remove(lis)
    s = [int(x) for x in res.stdout.split()]
    assert len(s) == n - 1 - i, (i, len(s))
    return i, s

def main():
    with open(os.path.join(ROOT, "out", "index.csv")) as f:
        paths = [os.path.join(ROOT, r["caminho_xyt"]) for r in csv.DictReader(f)]
    n = len(paths)
    m = np.full((n, n), -1, dtype=np.int32)
    with ProcessPoolExecutor(3) as ex:
        for i, s in ex.map(linha, [(i, paths) for i in range(n)], chunksize=8):
            m[i, i + 1:] = s
            m[i + 1:, i] = s
    os.makedirs(OUT, exist_ok=True)
    m.astype("<i4").tofile(os.path.join(OUT, "matrix.i32"))
    print(f"{n}x{n} gravada; max fora da diagonal = {m.max()}")

if __name__ == "__main__":
    main()
