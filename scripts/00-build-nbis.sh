#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# scripts/00-build-nbis.sh
# Compila mindtct e bozorth3 do NIST NBIS de forma idempotente e modular.
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NBIS_SRC="${REPO_ROOT}/nbis-src"
BUILD_DIR="${REPO_ROOT}/build"
BIN_DIR="${BUILD_DIR}/bin"

mkdir -p "${BIN_DIR}"

MINDTCT_BIN="${BIN_DIR}/mindtct"
BOZORTH3_BIN="${BIN_DIR}/bozorth3"

if [[ -x "${MINDTCT_BIN}" && -x "${BOZORTH3_BIN}" ]]; then
    echo "[INFO] Binários já compilados e instalados em ${BIN_DIR}."
else
    echo "[INFO] Iniciando preparação e compilação do NBIS..."

    # 1. Clonar repositório NBIS se não existir
    if [[ ! -d "${NBIS_SRC}" ]]; then
        echo "[INFO] Clonando NBIS de https://github.com/lessandro/nbis..."
        git clone --depth 1 https://github.com/lessandro/nbis "${NBIS_SRC}"
    fi

    cd "${NBIS_SRC}"

    # 2. Configuração do ambiente de build do NBIS
    if [[ ! -f "rules.mak" ]]; then
        echo "[INFO] Executando setup.sh..."
        ./setup.sh "${BUILD_DIR}" --without-X11 --STDLIBS --64
    fi

    if [[ ! -d "exports" ]]; then
        echo "[INFO] Executando make config..."
        make config
    fi

    # 3. Exportar headers dos módulos necessários
    echo "[INFO] Exportando headers..."
    for pkg in ijg commonnbis imgtools an2k bozorth3 mindtct; do
        (cd "${pkg}" && make cpheaders)
    done

    # 4. Compilar bibliotecas necessárias
    echo "[INFO] Compilando bibliotecas..."
    (cd bozorth3 && make libs)

    # commonnbis (apenas as libs necessárias para mindtct)
    (cd commonnbis/src/lib/cblas && make)
    (cd commonnbis/src/lib/clapck && make)
    (cd commonnbis/src/lib/f2c && make)
    (cd commonnbis/src/lib/fet && make)
    (cd commonnbis/src/lib/ioutil && make)
    (cd commonnbis/src/lib/util && make)

    # bibliotecas de imagem e detecção
    (cd ijg && make libs)
    (cd imgtools && make libs)
    (cd an2k && make libs)
    (cd mindtct && make libs)

    # 5. Compilar binários
    echo "[INFO] Compilando binários bozorth3 e mindtct..."
    (cd bozorth3 && make bins)
    (cd mindtct && make bins)
    (cd imgtools/src/bin/cwsq && make)

    # 6. Instalar em build/bin
    echo "[INFO] Instalando binários em ${BIN_DIR}..."
    cp "${NBIS_SRC}/bozorth3/bin/bozorth3" "${MINDTCT_BIN%/*}/"
    cp "${NBIS_SRC}/mindtct/bin/mindtct" "${MINDTCT_BIN%/*}/"
    if [[ -f "${NBIS_SRC}/imgtools/bin/cwsq" ]]; then
        cp "${NBIS_SRC}/imgtools/bin/cwsq" "${BIN_DIR}/"
    fi
fi

# 7. Imprimir caminhos e informações de versão/uso
echo "=================================================================="
echo "NBIS BUILD COMPLETO COM SUCESSO"
echo "=================================================================="
echo "mindtct : ${MINDTCT_BIN}"
echo "bozorth3: ${BOZORTH3_BIN}"
echo ""
echo "--- mindtct usage ---"
"${MINDTCT_BIN}" 2>&1 || true
echo ""
echo "--- bozorth3 usage ---"
"${BOZORTH3_BIN}" -h 2>&1 || true
