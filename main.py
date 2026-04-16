"""
main.py
Pipeline completo: limpeza → NER → grafo de co-ocorrência
                            ↓
                   metadados → grafo relacional acadêmico
                   (UNIV → DEPT → ORIENTADOR → AUTOR → TRABALHO)
"""

import os
import preprocessamento
import create_grafo
import visualizar_grafo

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
PASTA           = "./output"          # pasta com os .txt extraídos dos PDFs
K_CHARS         = 500
ESTRATEGIA      = "sentenca"
PASTA_FIGURAS   = "./figuras"


if __name__ == "__main__":
    os.makedirs(PASTA_FIGURAS, exist_ok=True)

    print("\n─── Inicializando o Programa ───")

    arquivos = sorted([
        f for f in os.listdir(PASTA)
        if os.path.isfile(os.path.join(PASTA, f))
    ])
    print(f"Encontrados {len(arquivos)} arquivo(s):")
    for a in arquivos:
        print(f"  {a}")

    # ── Coleta de janelas (co-ocorrência) e metadados ──────────────────────
    janelas_sentenca  = []
    janelas_paragrafo = []
    janelas_kchars    = []

    lista_metadados = []   # ← para o grafo relacional
    lista_docs      = []   # ← para associar áreas temáticas

    for nome in arquivos:
        caminho = os.path.join(PASTA, nome)
        print(f"\n{'═'*50}")
        print(f"  Arquivo: {nome}")
        print(f"{'═'*50}")

        # ── leitura bruta para metadados (antes do recorte do corpo) ──────
        texto_bruto = preprocessamento.ler_arquivo(caminho)
        meta = preprocessamento.extrair_metadados(texto_bruto, nome_arquivo=nome)
        lista_metadados.append(meta)

        # ── limpeza e NER ──────────────────────────────────────────────────
        texto_limpo = preprocessamento.preprocessar(caminho, diagnostico=False)
        print(f"  Caracteres limpos : {len(texto_limpo)}")
        print(f"  Parágrafos        : {texto_limpo.count(chr(10)*2) + 1}")
        preprocessamento.diagnosticar(texto_limpo)

        print("\n  Processando NER...")
        doc = preprocessamento.processar_spacy(texto_limpo)
        lista_docs.append(doc)

        janelas_sentenca.extend(create_grafo.janela_sentenca(doc))
        janelas_paragrafo.extend(create_grafo.janela_paragrafo(texto_limpo, doc))
        janelas_kchars.extend(create_grafo.janela_k_caracteres(texto_limpo, doc, k=K_CHARS))

    print(f"\n  Janelas coletadas:")
    print(f"    sentença     → {len(janelas_sentenca)}")
    print(f"    parágrafo    → {len(janelas_paragrafo)}")
    print(f"    k={K_CHARS}chars → {len(janelas_kchars)}")

    # ── Grafos de co-ocorrência ────────────────────────────────────────────
    print("\n  Construindo grafos de co-ocorrência...")
    G_sentenca  = create_grafo.criar_grafo(janelas_sentenca,  estrategia="sentenca")
    G_paragrafo = create_grafo.criar_grafo(janelas_paragrafo, estrategia="paragrafo")
    G_kchars    = create_grafo.criar_grafo(janelas_kchars,    estrategia=f"k{K_CHARS}chars")

    grafos = {
        "sentenca":         G_sentenca,
        "paragrafo":        G_paragrafo,
        f"k{K_CHARS}chars": G_kchars,
    }

    # ── Grafo relacional acadêmico ─────────────────────────────────────────
    print("\n  Construindo grafo relacional acadêmico...")
    G_relacional = create_grafo.criar_grafo_relacional(lista_metadados, lista_docs)
    create_grafo.analisar_grafo_relacional(G_relacional)

    # ── Análise co-ocorrência ──────────────────────────────────────────────
    resultados = {}
    for nome_e, G in grafos.items():
        metricas = create_grafo.analisar_grafo(G)
        resultados[nome_e] = {"grafo": G, "metricas": metricas}

    # ── Figuras estáticas ──────────────────────────────────────────────────
    print("\n  Gerando figuras estáticas...")
    for nome_e, G in grafos.items():
        visualizar_grafo.figura_grafo_completo(
            G,
            salvar=os.path.join(PASTA_FIGURAS, f"grafo_{nome_e}.png"),
            top_nos=80,
        )
        visualizar_grafo.figura_distribuicao_grau(
            G,
            salvar=os.path.join(PASTA_FIGURAS, f"dist_grau_{nome_e}.png"),
        )

    visualizar_grafo.figura_comparativa(
        resultados,
        salvar=os.path.join(PASTA_FIGURAS, "comparacao_estrategias.png"),
    )

    # ── Figura do grafo relacional ─────────────────────────────────────────
    print("\n  Gerando figura do grafo relacional...")
    visualizar_grafo.figura_grafo_relacional(
        G_relacional,
        salvar=os.path.join(PASTA_FIGURAS, "grafo_relacional.png"),
    )

    # ── HTMLs interativos ──────────────────────────────────────────────────
    print("\n  Gerando HTMLs interativos...")
    for nome_e, G in grafos.items():
        visualizar_grafo.visualizar_grafo_interativo(
            G,
            salvar_html=os.path.join(PASTA_FIGURAS, f"grafo_{nome_e}.html"),
            filtro_peso_min=2,
        )

    # HTML interativo do grafo relacional
    visualizar_grafo.visualizar_grafo_relacional_interativo(
        G_relacional,
        salvar_html=os.path.join(PASTA_FIGURAS, "grafo_relacional.html"),
    )

    # ── Ego-grafo do nó mais central ───────────────────────────────────────
    G_principal = grafos[ESTRATEGIA]
    no_top = max(dict(G_principal.degree()).items(), key=lambda x: x[1])[0]
    print(f"\n  Nó mais central em '{ESTRATEGIA}': {no_top}")

    visualizar_grafo.visualizar_ego_interativo(
        G_principal, no_top, raio=2,
        salvar_html=os.path.join(PASTA_FIGURAS, f"ego_{no_top}.html"),
        abrir_browser=True,
    )
    visualizar_grafo.figura_ego(
        G_principal, no_top, raio=2,
        salvar=os.path.join(PASTA_FIGURAS, f"ego_{no_top}.png"),
    )

    # ── Busca interativa ───────────────────────────────────────────────────
    print("\n  Entrando no modo de busca interativa...")
    create_grafo.buscar_termo(
        G_principal,
        visualizar_ego_fn=lambda G, termo, raio: (
            visualizar_grafo.visualizar_ego_interativo(
                G, termo, raio=raio,
                salvar_html=os.path.join(PASTA_FIGURAS, f"ego_{termo}.html"),
                abrir_browser=True,
            ),
            visualizar_grafo.figura_ego(
                G, termo, raio=raio,
                salvar=os.path.join(PASTA_FIGURAS, f"ego_{termo}.png"),
            )
        )
    )

    print(f"\n  ✓ Tudo salvo em '{PASTA_FIGURAS}/'")