#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# scripts/10-download-fvc.sh
# Baixa e descompacta os conjuntos públicos DB1_B..DB4_B de FVC2000/2002/2004/2006.
# Idempotente: não rebaixa se já existir.
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_FVC="${REPO_ROOT}/data/fvc"
DOWNLOADS_DIR="${DATA_FVC}/.downloads"

mkdir -p "${DATA_FVC}" "${DOWNLOADS_DIR}"

YEARS=(2000 2002 2004 2006)
DBS=("DB1_B" "DB2_B" "DB3_B" "DB4_B")

echo "=================================================================="
echo "DOWNLOAD E DESCOMPACTAÇÃO DAS BASES FVC"
echo "=================================================================="

for year in "${YEARS[@]}"; do
    for db in "${DBS[@]}"; do
        base_name="FVC${year}_${db}"
        target_dir="${DATA_FVC}/${base_name}"
        zip_file="${DOWNLOADS_DIR}/${base_name}.zip"
        url="http://bias.csr.unibo.it/fvc${year}/Downloads/${db}.zip"

        # Verificar se já foi descompactado e contém imagens
        if [[ -d "${target_dir}" ]] && (( $(find "${target_dir}" -type f \( -iname "*.tif" -o -iname "*.bmp" -o -iname "*.png" -o -iname "*.jpg" \) 2>/dev/null | wc -l) > 0 )); then
            echo "[INFO] ${base_name}: Já descompactado em ${target_dir}."
            continue
        fi

        # Se o zip não existe, tentar baixar
        if [[ ! -f "${zip_file}" ]]; then
            echo "[INFO] Baixando ${base_name} de ${url}..."
            if curl -sSL -f -o "${zip_file}" "${url}"; then
                echo "[INFO] Download de ${base_name} concluído."
            else
                rm -f "${zip_file}"
                echo "[AVISO] ${base_name} não encontrado ou retornou 404 em ${url}. Pulando."
                continue
            fi
        fi

        # Descompactar
        mkdir -p "${target_dir}"
        echo "[INFO] Descompactando ${zip_file} para ${target_dir}..."
        unzip -q -o "${zip_file}" -d "${target_dir}"

        # Se descompactou dentro de uma subpasta, mover arquivos para a raiz de target_dir
        find "${target_dir}" -mindepth 2 -type f \( -iname "*.tif" -o -iname "*.bmp" -o -iname "*.png" -o -iname "*.jpg" \) -exec mv -t "${target_dir}" {} + 2>/dev/null || true
        # Limpar diretórios vazios remanescentes se houver
        find "${target_dir}" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    done
done

echo ""
echo "=================================================================="
echo "CONTAGEM DE IMAGENS POR CONJUNTO FVC"
echo "=================================================================="
total_imgs=0
for d in "${DATA_FVC}"/FVC*; do
    if [[ -d "${d}" ]]; then
        bname="$(basename "${d}")"
        count=$(find "${d}" -type f \( -iname "*.tif" -o -iname "*.bmp" -o -iname "*.png" -o -iname "*.jpg" \) | wc -l)
        echo "  - ${bname}: ${count} imagens"
        total_imgs=$((total_imgs + count))
    fi
done
echo "------------------------------------------------------------------"
echo "Total de imagens FVC disponíveis: ${total_imgs}"
echo "=================================================================="
