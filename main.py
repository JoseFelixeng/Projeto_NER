"""
Pipeline de pré-processamento com Regex para uso com spaCy
"""
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import regex
import os
# No topo do main.py, adicione:
import visualizar_grafo
from collections import Counter


def extrair_entidades(doc):
    lista = []
    for sent in doc.sents:
        ents = list(set([ent.text.strip() for ent in sent.ents if len(ent.text) > 2]))
        if len(ents) > 1:
            lista.append(ents)
    return lista


def visualizar_ego_galaxia(G, termo_central):
    if termo_central not in G:
        print(f"Termo '{termo_central}' não encontrado no grafo.")
        return

    ego = nx.ego_graph(G, termo_central, radius=2)
    centralidade = nx.degree_centrality(ego)
    pos = nx.spring_layout(ego, center=(0, 0), k=0.5)

    node_sizes = []
    node_colors = []
    for node in ego.nodes():
        if node == termo_central:
            node_sizes.append(8000)
            node_colors.append(1.0)
        else:
            valor = centralidade[node]
            node_sizes.append(3000 * valor)
            node_colors.append(valor)

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(ego, pos, node_size=node_sizes,
                           node_color=node_colors, cmap=plt.cm.plasma, alpha=0.9)
    nx.draw_networkx_edges(ego, pos, alpha=0.2)

    labels = {n: n for n in ego.nodes()
              if centralidade[n] > 0.05 or n == termo_central}
    nx.draw_networkx_labels(ego, pos, labels, font_size=9)

    plt.title(f"Ego Graph - {termo_central}")
    plt.axis("off")
    plt.show()


def analisar_grafo(G):
    print("\n─── INFORMAÇÕES DO GRAFO ───")
    print("Nós:", G.number_of_nodes())
    print("Arestas:", G.number_of_edges())

    centralidade = nx.degree_centrality(G)
    print("\nTop 10 entidades mais importantes:")
    top = sorted(centralidade.items(), key=lambda x: x[1], reverse=True)[:10]
    for no, valor in top:
        print(f"  {no} → {valor:.4f}")


def buscar_termo(G):
    while True:
        termo = input("\nDigite um termo (ou 'sair'): ").strip()
        if termo.lower() == "sair":
            break
        if termo not in G:
            print(f"Termo '{termo}' não encontrado. Tente novamente.")
            continue
        visualizar_grafo.visualizar_ego_interativo(G, termo, raio=2)


def criar_grafo(lista_entidades):
    G = nx.Graph()
    for ents in lista_entidades:
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                if G.has_edge(ents[i], ents[j]):
                    G[ents[i]][ents[j]]["weight"] += 1
                else:
                    G.add_edge(ents[i], ents[j], weight=1)
    return G


if __name__ == "__main__":
    PASTA = "./output"
    arquivos = sorted(os.listdir(PASTA))

    print(f"Encontrados {len(arquivos)} arquivos:")
    for a in arquivos:
        print(f"  {a}")

    todas_entidades = []

    for nome in arquivos:                          # itera sobre nomes, não range(23)
        CAMINHO = os.path.join(PASTA, nome)        # caminho completo como string
        print(f"\n─── {nome} ───")

        texto_limpo = regex.preprocessar(CAMINHO)  # passa string, não lista
        print(f"Caracteres limpos: {len(texto_limpo)}")
        print(f"Parágrafos: {texto_limpo.count(chr(10) * 2) + 1}")
        print("\n── Amostra ──")
        print(texto_limpo[:500])

        regex.diagnosticar(texto_limpo)
        doc = regex.processar_spacy(texto_limpo)

        print("Extraindo entidades...")
        lista_entidades = extrair_entidades(doc)
        todas_entidades.extend(lista_entidades)

    print("\nCriando grafo consolidado...")
    G = criar_grafo(todas_entidades)

    print("\nAnalisando grafo...")
    analisar_grafo(G)

# No lugar de nx.ego_graph(...) e visualizar_ego_galaxia(...):
    print("\nCriando visualização interativa...")
    visualizar_grafo.visualizar_grafo_interativo(G, salvar_html="grafo_entidades.html")

    print("Entrando no modo de busca...")
    buscar_termo(G)