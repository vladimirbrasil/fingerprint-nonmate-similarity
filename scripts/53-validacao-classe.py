#!/usr/bin/env python3
# ==============================================================================
# scripts/53-validacao-classe.py
# Gera interface web local para validação cega das classes de 100 impressões digitais.
#
# Funcionalidades:
# - Sorteia 100 dedos de SOCOFing Real com semente fixa (seed=42).
# - Gera página HTML estática e auto-contida em out/estudo02/validacao/index.html.
# - Botões para as 5 classes: Arco, Presilha interna (I), Presilha externa (E) — convenção Vucetich, pelo observador, Verticilo, Não sei.
# - Atalhos de teclado ergonômicos (1..5, A/E/D/V/N, setas de navegação).
# - Persistência automática em localStorage (não perde o progresso ao fechar o navegador).
# - Exportação instantânea de CSV com download direto no navegador.
# - NÃO exibe nenhuma informação ou classe gerada pelo pcasys.
#
# Saídas:
# - out/estudo02/validacao/index.html
# - out/estudo02/validacao/amostra_100_dedos.json
# ==============================================================================

import os
import json
import csv
import glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_VALIDACAO = os.path.join(ROOT, "out", "estudo02", "validacao")
INDEX_REAL_CSV = os.path.join(ROOT, "out", "socofing", "index.csv")
HTML_OUTPUT = os.path.join(OUT_VALIDACAO, "index.html")
AMOSTRA_JSON = os.path.join(OUT_VALIDACAO, "amostra_100_dedos.json")


def selecionar_amostra(index_csv, n=100, seed=42):
    """Sorteia n dedos da base Real de forma determinística."""
    with open(index_csv, "r", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(linhas), size=n, replace=False)
    indices = sorted(indices)

    amostra = []
    for i, idx in enumerate(indices, 1):
        r = linhas[idx]
        stem = os.path.splitext(os.path.basename(r["caminho_png"]))[0]
        # Caminho relativo a partir de out/estudo02/validacao/index.html
        rel_img = f"../../socofing/img/real/{stem}.png"
        amostra.append({
            "id_amostra": i,
            "sujeito": int(r["sujeito"]),
            "sexo": r["sexo"],
            "mao": r["mao"],
            "dedo": r["dedo"],
            "arquivo": stem,
            "img_src": rel_img
        })

    return amostra


def gerar_html(amostra, output_html):
    """Gera a página HTML interativa de validação cega."""
    amostra_json_str = json.dumps(amostra, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validação Cega de Classes — Estudo 02 (SOCOFing)</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --accent-arc: #f59e0b;
            --accent-lloop: #10b981;
            --accent-rloop: #06b6d4;
            --accent-whorl: #8b5cf6;
            --accent-unknown: #64748b;
            --border: #334155;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}

        header {{
            background-color: var(--card-bg);
            border-bottom: 1px solid var(--border);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        header h1 {{
            font-size: 1.15rem;
            font-weight: 600;
        }}

        .stats-badge {{
            background: #0f172a;
            border: 1px solid var(--border);
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.9rem;
            font-weight: 500;
        }}

        .main-container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}

        .content-area {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            overflow-y: auto;
        }}

        .sidebar {{
            width: 280px;
            background-color: var(--card-bg);
            border-left: 1px solid var(--border);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            overflow-y: auto;
        }}

        .image-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
            max-width: 500px;
            width: 100%;
        }}

        .image-wrapper {{
            width: 320px;
            height: 340px;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid var(--border);
            margin-bottom: 16px;
        }}

        .image-wrapper img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            image-rendering: pixelated;
        }}

        .finger-meta {{
            width: 100%;
            display: flex;
            justify-content: space-between;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }}

        .buttons-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            width: 100%;
            margin-bottom: 12px;
        }}

        .btn-class {{
            padding: 14px 10px;
            border-radius: 8px;
            border: 2px solid transparent;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 4px;
            transition: all 0.15s ease;
            color: #fff;
        }}

        .btn-class .shortcut {{
            font-size: 0.75rem;
            opacity: 0.8;
            font-weight: normal;
        }}

        .btn-arc {{ background: #92400e; border-color: #b45309; }}
        .btn-arc:hover, .btn-arc.active {{ background: #d97706; border-color: #f59e0b; }}

        .btn-lloop {{ background: #065f46; border-color: #047857; }}
        .btn-lloop:hover, .btn-lloop.active {{ background: #059669; border-color: #10b981; }}

        .btn-rloop {{ background: #155e75; border-color: #0e7490; }}
        .btn-rloop:hover, .btn-rloop.active {{ background: #0891b2; border-color: #06b6d4; }}

        .btn-whorl {{ background: #5b21b6; border-color: #6d28d9; }}
        .btn-whorl:hover, .btn-whorl.active {{ background: #7c3aed; border-color: #8b5cf6; }}

        .btn-unknown {{
            grid-column: span 2;
            background: #334155;
            border-color: #475569;
        }}
        .btn-unknown:hover, .btn-unknown.active {{ background: #475569; border-color: #64748b; }}

        .nav-controls {{
            display: flex;
            justify-content: space-between;
            width: 100%;
            margin-top: 10px;
            gap: 12px;
        }}

        .btn-nav {{
            flex: 1;
            padding: 10px;
            background: #334155;
            border: 1px solid var(--border);
            color: var(--text-color);
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
        }}
        .btn-nav:hover {{ background: #475569; }}

        .btn-export {{
            width: 100%;
            padding: 12px;
            background: #16a34a;
            border: none;
            color: white;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            margin-top: auto;
        }}
        .btn-export:hover {{ background: #15803d; }}

        .progress-bar-container {{
            width: 100%;
            height: 8px;
            background: #0f172a;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 12px;
        }}

        .progress-bar {{
            height: 100%;
            background: var(--primary);
            width: 0%;
            transition: width 0.2s ease;
        }}

        .grid-overview {{
            display: grid;
            grid-template-columns: repeat(10, 1fr);
            gap: 4px;
            max-height: 250px;
            overflow-y: auto;
            background: #0f172a;
            padding: 8px;
            border-radius: 6px;
        }}

        .grid-item {{
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            background: #1e293b;
            border-radius: 3px;
            cursor: pointer;
            color: var(--text-muted);
        }}

        .grid-item.classified {{
            background: #2563eb;
            color: #fff;
            font-weight: bold;
        }}

        .grid-item.current {{
            border: 2px solid #f8fafc;
        }}

        .shortcuts-help {{
            font-size: 0.8rem;
            color: var(--text-muted);
            line-height: 1.4;
            background: #0f172a;
            padding: 10px;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Validação Cega de Classes (100 Dedos)</h1>
        <div class="stats-badge" id="stats-badge">Classificados: 0 / 100</div>
    </header>

    <div class="main-container">
        <div class="content-area">
            <div class="image-card">
                <div class="finger-meta">
                    <span id="finger-counter">Amostra 1 / 100</span>
                    <span id="finger-info">Dedo: ---</span>
                </div>

                <div class="image-wrapper">
                    <img id="finger-image" src="" alt="Impressão digital">
                </div>

                <div class="buttons-grid">
                    <button class="btn-class btn-arc" onclick="classificar('Arco')">
                        Arco
                        <span class="shortcut">[1] ou [A]</span>
                    </button>
                    <button class="btn-class btn-lloop" onclick="classificar('Presilha_esquerda')">
                        Presilha Esquerda
                        <span class="shortcut">[2] ou [E]</span>
                    </button>
                    <button class="btn-class btn-rloop" onclick="classificar('Presilha_direita')">
                        Presilha Direita
                        <span class="shortcut">[3] ou [D]</span>
                    </button>
                    <button class="btn-class btn-whorl" onclick="classificar('Verticilo')">
                        Verticilo
                        <span class="shortcut">[4] ou [V]</span>
                    </button>
                    <button class="btn-class btn-unknown" onclick="classificar('Nao_sei')">
                        Não Sei / Ilegível
                        <span class="shortcut">[5] ou [N]</span>
                    </button>
                </div>

                <div class="nav-controls">
                    <button class="btn-nav" onclick="navegar(-1)">Anterior (←)</button>
                    <button class="btn-nav" onclick="navegar(1)">Próximo (→)</button>
                </div>
            </div>
        </div>

        <div class="sidebar">
            <h3>Progresso</h3>
            <div class="progress-bar-container">
                <div class="progress-bar" id="progress-bar"></div>
            </div>

            <div class="grid-overview" id="grid-overview"></div>

            <div class="shortcuts-help">
                <strong>Atalhos de Teclado:</strong><br>
                • <code>1</code> ou <code>A</code>: Arco<br>
                • <code>2</code> ou <code>E</code>: Presilha Esq.<br>
                • <code>3</code> ou <code>D</code>: Presilha Dir.<br>
                • <code>4</code> ou <code>V</code>: Verticilo<br>
                • <code>5</code> ou <code>N</code>: Não sei<br>
                • <code>←</code> / <code>→</code>: Navegação
            </div>

            <button class="btn-export" onclick="exportarCSV()">Exportar Classificações (CSV)</button>
        </div>
    </div>

    <script>
        const amostra = {amostra_json_str};
        let currentIndex = 0;
        const STORAGE_KEY = "socofing_validacao_100_v2_vucetich";

        // Carregar estado salvo do localStorage
        function getRespostas() {{
            try {{
                return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {{}};
            }} catch (e) {{
                return {{}};
            }}
        }}

        function salvarResposta(id_amostra, classe) {{
            const respostas = getRespostas();
            respostas[id_amostra] = {{
                classe: classe,
                timestamp: new Date().toISOString()
            }};
            localStorage.setItem(STORAGE_KEY, JSON.stringify(respostas));
            atualizarUI();
        }}

        function classificar(classe) {{
            const item = amostra[currentIndex];
            salvarResposta(item.id_amostra, classe);
            if (currentIndex < amostra.length - 1) {{
                currentIndex++;
                carregarDedo();
            }}
        }}

        function navegar(delta) {{
            const novo = currentIndex + delta;
            if (novo >= 0 && novo < amostra.length) {{
                currentIndex = novo;
                carregarDedo();
            }}
        }}

        function carregarDedo() {{
            const item = amostra[currentIndex];
            document.getElementById("finger-counter").innerText = `Amostra ${{currentIndex + 1}} de ${{amostra.length}}`;
            document.getElementById("finger-info").innerText = `Sujeito #${{item.sujeito}} (${{item.mao}} ${{item.dedo}})`;
            document.getElementById("finger-image").src = item.img_src;

            // Destacar botão ativo se já classificado
            const respostas = getRespostas();
            const salvo = respostas[item.id_amostra];
            const botoes = document.querySelectorAll(".btn-class");
            botoes.forEach(b => b.classList.remove("active"));

            if (salvo) {{
                botoes.forEach(b => {{
                    if (b.innerText.toLowerCase().includes(salvo.classe.toLowerCase().replace("_", " "))) {{
                        b.classList.add("active");
                    }}
                }});
            }}

            atualizarUI();
        }}

        function atualizarUI() {{
            const respostas = getRespostas();
            const total = amostra.length;
            const preenchidos = Object.keys(respostas).length;

            document.getElementById("stats-badge").innerText = `Classificados: ${{preenchidos}} / ${{total}}`;
            const pct = (preenchidos / total) * 100;
            document.getElementById("progress-bar").style.width = `${{pct}}%`;

            // Atualizar grid
            const grid = document.getElementById("grid-overview");
            grid.innerHTML = "";
            amostra.forEach((item, idx) => {{
                const div = document.createElement("div");
                div.className = "grid-item";
                if (respostas[item.id_amostra]) div.classList.add("classified");
                if (idx === currentIndex) div.classList.add("current");
                div.innerText = idx + 1;
                div.onclick = () => {{
                    currentIndex = idx;
                    carregarDedo();
                }};
                grid.appendChild(div);
            }});
        }}

        function exportarCSV() {{
            const respostas = getRespostas();
            let csv = "arquivo,classe_vladimir\\n";
            amostra.forEach(item => {{
                const r = respostas[item.id_amostra] || {{ classe: "Nao_classificado", timestamp: "" }};
                csv += `${{item.arquivo}},${{r.classe}}\\n`;
            }});

            const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8;" }});
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = "classificacao_vladimir_100.csv";
            link.click();
        }}

        // Atalhos de teclado
        window.addEventListener("keydown", (e) => {{
            const key = e.key.toUpperCase();
            if (key === "1" || key === "A") classificar("Arco");
            else if (key === "2" || key === "E") classificar("Presilha_esquerda");
            else if (key === "3" || key === "D") classificar("Presilha_direita");
            else if (key === "4" || key === "V") classificar("Verticilo");
            else if (key === "5" || key === "N") classificar("Nao_sei");
            else if (e.key === "ArrowLeft") navegar(-1);
            else if (e.key === "ArrowRight") navegar(1);
        }});

        // Inicialização
        carregarDedo();
    </script>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    os.makedirs(OUT_VALIDACAO, exist_ok=True)
    amostra = selecionar_amostra(INDEX_REAL_CSV, n=100, seed=42)

    with open(AMOSTRA_JSON, "w", encoding="utf-8") as f:
        json.dump(amostra, f, indent=2, ensure_ascii=False)

    gerar_html(amostra, HTML_OUTPUT)

    print("==================================================================")
    print("GERAÇÃO DA VALIDAÇÃO CEGA DE CLASSES (ESTUDO 02)")
    print("==================================================================")
    print(f"Amostra selecionada: {len(amostra)} dedos (semente 42)")
    print(f"Arquivo JSON da amostra: {AMOSTRA_JSON}")
    print(f"Página HTML gerada: {HTML_OUTPUT}")
    print("A página está pronta para classificação cega pelo Vladimir.")
    print("==================================================================")


if __name__ == "__main__":
    main()
