"""
main.py — Pipeline completo NER → Grafo de Co-ocorrência
"""

import os
import regex
import create_grafo
import visualizar_grafo

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
PASTA         = "./output"       # .txt extraídos dos PDFs
K_CHARS       = 500              # tamanho da janela k-chars
ESTRATEGIA    = "sentenca"       # estratégia principal para ego-grafo
PASTA_FIGURAS = "./figuras"
FILTRO_PESO   = 2                # arestas com peso < isso são removidas do HTML


if __name__ == "__main__":
    os.makedirs(PASTA_FIGURAS, exist_ok=True)

    print("\n─── Inicializando Pipeline NER → Grafo ───")

    # 1. Lista arquivos
    arquivos = sorted([
        f for f in os.listdir(PASTA)
        if os.path.isfile(os.path.join(PASTA, f))
    ])
    print(f"Encontrados {len(arquivos)} arquivo(s):")
    for a in arquivos:
        print(f"  {a}")

    # 2. Limpeza + NER + janelas
    janelas_sentenca  = []
    janelas_paragrafo = []
    janelas_kchars    = []

    for nome in arquivos:
        caminho = os.path.join(PASTA, nome)
        print(f"\n{'═'*55}\n  {nome}\n{'═'*55}")

        texto_limpo = regex.preprocessar(caminho, diagnostico=False)
        print(f"  Caracteres : {len(texto_limpo)}")
        print(f"  Parágrafos : {texto_limpo.count(chr(10)*2) + 1}")
        print(f"\n  Amostra:\n{texto_limpo[:400]}\n")

        doc = regex.processar_spacy(texto_limpo)

        janelas_sentenca.extend(create_grafo.janela_sentenca(doc))
        janelas_paragrafo.extend(create_grafo.janela_paragrafo(texto_limpo, doc))
        janelas_kchars.extend(create_grafo.janela_k_caracteres(texto_limpo, doc, k=K_CHARS))

    print(f"\n  Janelas coletadas:")
    print(f"    sentença   → {len(janelas_sentenca)}")
    print(f"    parágrafo  → {len(janelas_paragrafo)}")
    print(f"    k{K_CHARS}chars → {len(janelas_kchars)}")

    # 3. Grafos
    print("\n  Construindo grafos...")
    G_sentenca  = create_grafo.criar_grafo(janelas_sentenca,  "sentenca")
    G_paragrafo = create_grafo.criar_grafo(janelas_paragrafo, "paragrafo")
    G_kchars    = create_grafo.criar_grafo(janelas_kchars,    f"k{K_CHARS}chars")

    grafos = {
        "sentenca"       : G_sentenca,
        "paragrafo"      : G_paragrafo,
        f"k{K_CHARS}chars": G_kchars,
    }

    # 4. Análise individual + resumo comparativo
    resultados = {}
    for nome_e, G in grafos.items():
        metricas = create_grafo.analisar_grafo(G)
        resultados[nome_e] = {"grafo": G, "metricas": metricas}

    create_grafo.resumo_comparativo(resultados)

    # 5. Figuras estáticas
    print("\n  Gerando figuras estáticas...")
    for nome_e, G in grafos.items():
        visualizar_grafo.figura_grafo_completo(
            G, salvar=os.path.join(PASTA_FIGURAS, f"grafo_{nome_e}.png"), top_nos=80
        )
        visualizar_grafo.figura_distribuicao_grau(
            G, salvar=os.path.join(PASTA_FIGURAS, f"dist_grau_{nome_e}.png")
        )

    visualizar_grafo.figura_comparativa(
        resultados,
        salvar=os.path.join(PASTA_FIGURAS, "comparacao_estrategias.png"),
    )

    # 6. HTMLs interativos
    print("\n  Gerando HTMLs interativos...")
    for nome_e, G in grafos.items():
        visualizar_grafo.visualizar_grafo_interativo(
            G,
            salvar_html=os.path.join(PASTA_FIGURAS, f"grafo_{nome_e}.html"),
            filtro_peso_min=FILTRO_PESO,
        )

    # 7. Ego-grafo do nó mais central
    G_principal = grafos[ESTRATEGIA]
    no_top = max(dict(G_principal.degree()).items(), key=lambda x: x[1])[0]
    print(f"\n  Nó mais central ({ESTRATEGIA}): {no_top}")

    visualizar_grafo.visualizar_ego_interativo(
        G_principal, no_top, raio=2,
        salvar_html=os.path.join(PASTA_FIGURAS, f"ego_{no_top}.html"),
        abrir_browser=True,
    )
    visualizar_grafo.figura_ego(
        G_principal, no_top, raio=2,
        salvar=os.path.join(PASTA_FIGURAS, f"ego_{no_top}.png"),
    )

    # 8. Busca interativa
    print("\n  Modo de busca interativa (digite 'sair' para encerrar)...")
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
            ),
        ),
    )

    print(f"\n  ✓ Tudo salvo em '{PASTA_FIGURAS}/'")