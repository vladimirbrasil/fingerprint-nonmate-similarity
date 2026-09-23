# HANDOFF — estudo de similaridade entre dedos diferentes (pergunta do Michael Levin)

Atualizado: 2026-09-23. Repo público: https://github.com/vladimirbrasil/fingerprint-nonmate-similarity

## Objetivo

Michael Levin (Tufts / Allen Discovery Center) respondeu ao e-mail frio do Vladimir com um problema:
*"a digital de uma pessoa já bateu com a de alguém falecido? dá para procurar isso nas bases
disponíveis?"* — leitura provável: **o mesmo padrão reaparecendo em outro corpo**.

Entregar bem isto **pausa** o `~/git/alavanca/analises/emprego-missao/PLAYBOOK-POS-LEVIN.md`.
Contexto legal e da carreira: `~/git/alavanca/analises/emprego-missao/README.md`.

## Regra inegociável

Nada de AFIS, SINIC ou qualquer sistema da PF; o acesso profissional do Vladimir **não entra**.
Só dado público e software de domínio público. Isso está escrito na resposta ao Levin — é o que
torna o trabalho publicável.

## O que já está pronto (medido nesta máquina, Celeron N5105, 4 núcleos)

- `scripts/00-build-nbis.sh` → compila `build/bin/{mindtct,bozorth3,cwsq}` (NIST NBIS, domínio público).
- `scripts/01-bench.sh` → 75 ms/imagem (mindtct); 579 pares/s/núcleo (bozorth3).
- `scripts/10-download-fvc.sh` → baixa FVC2000/2002/2004 set B (público, sem conta).
- `scripts/20-extract.sh` → minúcias (.xyt), 3 núcleos, `nice`, idempotente.
- `scripts/30-allvsall.py` → todos-contra-todos com histograma + top-10k impostores + checkpoint.
- **Resultado:** 960 imagens, 120 dedos, 460.320 pares em 158 s. Mesmo dedo: mediana 58, p99 229.
  Dedos diferentes: mediana 6, p99 17, **máximo 59**, só **2 pares acima de 40 em 456.960**.
- **Achado principal:** os 129 "impostores" acima de 100 da rodada ingênua eram **o mesmo dedo em
  dois sensores** (FVC2000 DB1×DB2, todos os 10 índices) + identidades reaproveitadas nas bases
  sintéticas DB4. Registro duplicado é o que se disfarça de coincidência — e é como um acerto
  contra alguém dado como morto seria arquivado como erro administrativo.
- `REPORT.md` (inglês) e rascunho de resposta ao Levin **no Gmail, na thread certa** (aguardando
  revisão e envio do Vladimir).

## SourceAFIS (feito 23/09, commit 73b546b) — REPORT Results 4 e 5

- `sourceafis/` (Maven local em `build/apache-maven-3.9.9`), 70 s; `scripts/31` (matriz bozorth3),
  `scripts/32` (comparação) → `results/comparacao_matchers.json`.
- Discordam por IMPRESSÃO (top-10 sem interseção, Spearman 0,28); concordam mais por DEDO (0,59).
- Duplicatas conhecidas FVC2000 108 e 110 pontuam como impostores → escore não prova "diferente".
- ⚠️ Par FVC2002 DB3/108 × FVC2004 DB3/102: ambos parecem verticilos; top-10 nos dois matchers.
  A frase antiga "classe diferente, obviamente dedos diferentes" estava ERRADA e saiu do REPORT.
  **Pendência 👤: Vladimir olhar como perito** (`results/par_108x102_dois_matchers.png` + imagens em
  `data/fvc/`). Rascunho do Gmail ainda tem a frase errada.

## Próximos passos, na ordem

1. ~~SourceAFIS~~ feito. Na SOCOFing, rodar os DOIS matchers e só levar a humano o candidato que
   resistir nos dois e em várias impressões.
2. **SOCOFing** (600 pessoas, 6.000 imagens, ~18M pares, ~3 h aqui). ⚠️ **Excluir as pastas
   `Altered-Easy/Medium/Hard`** — são cópias alteradas de propósito das mesmas digitais e
   contaminariam a pilha de "pessoas diferentes" em massa. Conferir contagem antes de rodar.
   **Pendência 👤:** conta gratuita no Kaggle + `~/.kaggle/kaggle.json` (o dataset só sai por lá).
3. **NIST SD300–303** via formulário de solicitação (não é download direto).
4. **Lado "falecido":** não é computação. NamUs e sistemas nacionais guardam digital atrás de acesso
   policial. Resta revisão de literatura e de casos publicados (Mayfield 2004, McKie 1997, estudo
   black-box Ulery et al. PNAS 2011, modelo de individualidade Pankanti/Prabhakar/Jain 2002).
5. Atualizar `REPORT.md` e o repo público a cada rodada.

## Armadilhas já conhecidas (não repetir)

- **Rótulo ingênuo mente:** "mesmo id de dedo = mesma pessoa" falha entre bases; sempre checar se os
  pares de topo compartilham índice de dedo ou base.
- **`delegate` sai com exit 0 mesmo falhando** — conferir efeito no disco, nunca o status.
- **Não redistribuir as bases:** `data/` está no `.gitignore`; só resultados agregados e uma figura
  com citação de origem.
- Escore alto **não** é identificação; a decisão é humana (ACE-V). Isso precisa estar em todo texto.
