"""
main.py

Arquivo usado para centralizar todas as funções de analise e limpeza
dos textos
"""

import os
import preprocessamento as pp
import create_grafo     as cg
import visualizar_grafo as vg

from grafo_orientadores import (
    construir_grafo_orientadores,
    analisar_grafo_orientadores,
    visualizar_orientadores_interativo
)

PASTA         = "./output"
K_CHARS       = 500
ESTRATEGIA    = "sentenca"
PASTA_FIGURAS = "./figuras"
FILTRO_PESO   = 2


if __name__ == "__main__":
    os.makedirs(PASTA_FIGURAS, exist_ok=True)

    print("\n" + "═"*65)
    print("Criando as Co-ocorrencias baseadas nos materiais academcios")
    print("═"*65)

    arquivos = sorted([
        f for f in os.listdir(PASTA)
        if os.path.isfile(os.path.join(PASTA, f))
    ])
    print(f"\n{len(arquivos)} arquivo(s): {', '.join(arquivos)}")

    # ── ETAPA 1: coleta ──────────────────────────────────────────
    janelas_sent = []
    janelas_par  = []
    janelas_kch  = []
    lista_meta   = []
    lista_docs   = []

    for nome in arquivos:
        caminho = os.path.join(PASTA, nome)
        print(f"\n{'─'*60}\n  {nome}\n{'─'*60}")

        # metadados: lê o texto bruto ANTES do recorte do corpo
        texto_bruto = pp.ler_arquivo(caminho)
        texto_bruto = pp.normalizar_caracteres(texto_bruto)
        meta = pp.extrair_metadados(texto_bruto, nome_arquivo=nome)

        # limpeza + NER
        texto_limpo = pp.preprocessar(caminho, diagnostico=False)
        print(f"\n  Chars: {len(texto_limpo)}  |  "
              f"Parágrafos: {texto_limpo.count(chr(10)*2)+1}")

        doc = pp.processar_spacy(texto_limpo)

        # palavras-chave do TCC → incluídas nos metadados
        meta["palavras_chave"] = pp.extrair_palavras_chave(doc)
        print(f"  Palavras-chave ({len(meta['palavras_chave'])}): "
              f"{', '.join(meta['palavras_chave'][:8])}")

        lista_meta.append(meta)
        lista_docs.append(doc)

        janelas_sent.extend(cg.janela_sentenca(doc))
        janelas_par.extend(cg.janela_paragrafo(texto_limpo, doc))
        janelas_kch.extend(cg.janela_k_caracteres(texto_limpo, doc, k=K_CHARS))

    print(f"\n  Janelas: sentença={len(janelas_sent)} | "
          f"parágrafo={len(janelas_par)} | k{K_CHARS}={len(janelas_kch)}")

    # ── ETAPA 2: grafos de co-ocorrência ─────────────────────────
    print("\n\n" + "═"*65)
    print("  GRAFOS DE CO-OCORRÊNCIA NER")
    print("═"*65)

    G_sent = cg.criar_grafo(janelas_sent, "sentenca")
    G_par  = cg.criar_grafo(janelas_par,  "paragrafo")
    G_kch  = cg.criar_grafo(janelas_kch,  f"k{K_CHARS}chars")

    grafos_ner = {
        "sentenca":         G_sent,
        "paragrafo":        G_par,
        f"k{K_CHARS}chars": G_kch,
    }

    resultados = {}
    for nome_e, G in grafos_ner.items():
        m = cg.analisar_grafo(G)
        resultados[nome_e] = {"grafo": G, "metricas": m}

    cg.resumo_comparativo(resultados)

    # ── ETAPA 3: grafo relacional ─────────────────────────────────
    print("\n\n" + "═"*65)
    print("  GRAFO RELACIONAL ACADÊMICO")
    print("═"*65)

    G_rel = cg.criar_grafo_relacional(lista_meta, lista_docs)
    analise_rel = cg.analisar_relacional(G_rel)

    # ── ETAPA 4: figuras estáticas ────────────────────────────────
    print("\n\n  Gerando figuras...")

    for nome_e, G in grafos_ner.items():
        vg.figura_grafo_completo(
            G,
            salvar=os.path.join(PASTA_FIGURAS, f"grafo_{nome_e}.png"),
            top_nos=80,
        )
        vg.figura_distribuicao_grau(
            G,
            salvar=os.path.join(PASTA_FIGURAS, f"dist_grau_{nome_e}.png"),
        )

    vg.figura_comparativa(
        resultados,
        salvar=os.path.join(PASTA_FIGURAS, "comparacao_estrategias.png"),
    )

    # figura do grafo relacional (função em visualizar_grafo.py)
    vg.figura_grafo_relacional(
        G_rel,
        salvar=os.path.join(PASTA_FIGURAS, "grafo_relacional.png"),
    )

    # ── ETAPA 5: HTMLs interativos ────────────────────────────────
    print("\n  Gerando HTMLs...")

    for nome_e, G in grafos_ner.items():
        vg.visualizar_grafo_interativo(
            G,
            salvar_html=os.path.join(PASTA_FIGURAS, f"grafo_{nome_e}.html"),
            filtro_peso_min=FILTRO_PESO,
        )

    vg.visualizar_grafo_relacional_interativo(
        G_rel,
        salvar_html=os.path.join(PASTA_FIGURAS, "grafo_relacional.html"),
    )

    # ego-grafo do nó mais central
    G_p    = grafos_ner[ESTRATEGIA]
    no_top = max(dict(G_p.degree()).items(), key=lambda x: x[1])[0]
    print(f"\n  Nó mais central ({ESTRATEGIA}): {no_top}")

    vg.visualizar_ego_interativo(
        G_p, no_top, raio=2,
        salvar_html=os.path.join(PASTA_FIGURAS, f"ego_{no_top}.html"),
        abrir_browser=True,
    )
    vg.figura_ego(
        G_p, no_top, raio=2,
        salvar=os.path.join(PASTA_FIGURAS, f"ego_{no_top}.png"),
    )

    # ── ETAPA 6: busca interativa ─────────────────────────────────
    print("\n  Busca interativa NER (sair para encerrar)...")
    cg.buscar_termo(
        G_p,
        visualizar_ego_fn=lambda G, termo, raio: (
            vg.visualizar_ego_interativo(
                G, termo, raio=raio,
                salvar_html=os.path.join(PASTA_FIGURAS, f"ego_{termo}.html"),
                abrir_browser=True,
            ),
            vg.figura_ego(
                G, termo, raio=raio,
                salvar=os.path.join(PASTA_FIGURAS, f"ego_{termo}.png"),
            ),
        ),
    )

    print(f"\n  ✓ Concluído. Arquivos em '{PASTA_FIGURAS}/'")

    