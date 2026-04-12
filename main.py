"""
Arquivo MAIN
Pipeline completo: limpeza → NER → grafo de co-ocorrência → análise → visualização
"""

import os
import regex
import create_grafo
import visualizar_grafo

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
PASTA           = "./output"               # pasta com os .txt extraídos dos PDFs
K_CHARS         = 500                      # tamanho da janela k-caracteres
ESTRATEGIA      = "sentenca"               # estratégia para visualização principal
PASTA_FIGURAS   = "./figuras"              # onde salvar as imagens


if __name__ == "__main__":
    os.makedirs(PASTA_FIGURAS, exist_ok=True)

    print("\n─── Inicializando o Programa ───")

    # 1. Lista arquivos
    arquivos = sorted([
        f for f in os.listdir(PASTA)
        if os.path.isfile(os.path.join(PASTA, f))
    ])
    print(f"Encontrados {len(arquivos)} arquivo(s):")
    for a in arquivos:
        print(f"  {a}")

    # 2. Processa cada arquivo — limpeza + NER + coleta de janelas
    janelas_sentenca  = []
    janelas_paragrafo = []
    janelas_kchars    = []

    for nome in arquivos:
        caminho = os.path.join(PASTA, nome)
        print(f"\n{'═'*50}")
        print(f"  Arquivo: {nome}")
        print(f"{'═'*50}")

        texto_limpo = regex.preprocessar(caminho, diagnostico=False)
        print(f"  Caracteres limpos : {len(texto_limpo)}")
        print(f"  Parágrafos        : {texto_limpo.count(chr(10)*2) + 1}")
        print("\n  ── Amostra (500 chars) ──")
        print(texto_limpo[:500])
        regex.diagnosticar(texto_limpo)

        print("\n  Processando NER...")
        doc = regex.processar_spacy(texto_limpo)

        janelas_sentenca.extend(create_grafo.janela_sentenca(doc))
        janelas_paragrafo.extend(create_grafo.janela_paragrafo(texto_limpo, doc))
        janelas_kchars.extend(create_grafo.janela_k_caracteres(texto_limpo, doc, k=K_CHARS))

    print(f"\n  Janelas coletadas:")
    print(f"    sentença     → {len(janelas_sentenca)}")
    print(f"    parágrafo    → {len(janelas_paragrafo)}")
    print(f"    k={K_CHARS}chars → {len(janelas_kchars)}")

    # 3. Constrói os 3 grafos
    print("\n  Construindo grafos...")
    G_sentenca  = create_grafo.criar_grafo(janelas_sentenca,  estrategia="sentenca")
    G_paragrafo = create_grafo.criar_grafo(janelas_paragrafo, estrategia="paragrafo")
    G_kchars    = create_grafo.criar_grafo(janelas_kchars,    estrategia=f"k{K_CHARS}chars")

    grafos = {
        "sentenca"        : G_sentenca,
        "paragrafo"       : G_paragrafo,
        f"k{K_CHARS}chars": G_kchars,
    }

    # 4. Análise e métricas
    resultados = {}
    for nome_e, G in grafos.items():
        metricas = create_grafo.analisar_grafo(G)
        resultados[nome_e] = {"grafo": G, "metricas": metricas}

    # 5. Figuras estáticas — uma por estratégia + comparativa + distribuição de grau
    print("\n\n  Gerando figuras estáticas...")

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

    # 6. HTMLs interativos — um por estratégia
    print("\n  Gerando HTMLs interativos...")
    for nome_e, G in grafos.items():
        visualizar_grafo.visualizar_grafo_interativo(
            G,
            salvar_html=os.path.join(PASTA_FIGURAS, f"grafo_{nome_e}.html"),
            filtro_peso_min=2,
        )

    # 7. Ego-grafo do nó mais central — interativo + figura
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

    # 8. Busca interativa por termo
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