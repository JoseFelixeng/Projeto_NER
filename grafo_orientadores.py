"""
grafo_orientadores.py
Constrói e analisa o grafo de co-autoria/orientação a partir dos
metadados extraídos dos TCCs.

Nós:
  - Orientador  (PER)
  - Autor       (PER)
  - Trabalho    (WORK)
  - Departamento (ORG)

Arestas:
  - orientador → autor      : "orientou"
  - orientador → trabalho   : "orientou_trabalho"
  - autor      → trabalho   : "escreveu"
  - orientador → depto      : "pertence_a"
  - orientador → orientador : "co-orientou" (quando há coorientador)
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pyvis.network import Network
import os


# ─────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────
COR_NO = {
    "PER_orientador" : "#FFD700",   # dourado
    "PER_autor"      : "#4CAF50",   # verde
    "WORK"           : "#2196F3",   # azul
    "ORG"            : "#FF5722",   # laranja
}
FORMA_NO = {
    "PER_orientador" : "star",
    "PER_autor"      : "dot",
    "WORK"           : "square",
    "ORG"            : "triangle",
}


# ─────────────────────────────────────────────
# CONSTRUÇÃO DO GRAFO
# ─────────────────────────────────────────────
def construir_grafo_orientadores(lista_metadados: list) -> nx.DiGraph:
    """
    Recebe lista de dicts retornados por regex.extrair_metadados().
    Constrói DiGraph (dirigido) com 4 tipos de nó e 5 tipos de aresta.

    Normaliza nomes para evitar duplicatas por capitalização.
    """
    G = nx.DiGraph()
    G.graph["tipo"] = "orientadores"

    for meta in lista_metadados:
        orientador  = _norm(meta.get("orientador"))
        autor       = _norm(meta.get("autor"))
        trabalho    = _norm(meta.get("titulo") or meta.get("arquivo"))
        departamento = _norm(meta.get("departamento"))
        coorientador = _norm(meta.get("coorientador"))
        ano         = meta.get("ano", "?")

        # — nós —
        if orientador:
            _add_no(G, orientador, "PER_orientador", ano=ano)
        if autor:
            _add_no(G, autor, "PER_autor", ano=ano)
        if trabalho:
            _add_no(G, trabalho, "WORK", ano=ano,
                    autor=autor or "?", orientador=orientador or "?")
        if departamento:
            _add_no(G, departamento, "ORG")
        if coorientador:
            _add_no(G, coorientador, "PER_orientador", ano=ano)

        # — arestas —
        if orientador and autor:
            _add_aresta(G, orientador, autor, "orientou")
        if orientador and trabalho:
            _add_aresta(G, orientador, trabalho, "orientou_trabalho")
        if autor and trabalho:
            _add_aresta(G, autor, trabalho, "escreveu")
        if orientador and departamento:
            _add_aresta(G, orientador, departamento, "pertence_a")
        if coorientador and orientador:
            _add_aresta(G, orientador, coorientador, "co-orientou")
        if coorientador and autor:
            _add_aresta(G, coorientador, autor, "co-orientou_aluno")

    return G


def _norm(texto: str | None) -> str | None:
    if not texto:
        return None
    return " ".join(texto.strip().title().split())


def _add_no(G: nx.DiGraph, nome: str, tipo: str, **attrs):
    if not G.has_node(nome):
        G.add_node(nome, tipo=tipo, count=0, **attrs)
    G.nodes[nome]["count"] = G.nodes[nome].get("count", 0) + 1


def _add_aresta(G: nx.DiGraph, origem: str, destino: str, relacao: str):
    if G.has_edge(origem, destino):
        G[origem][destino]["weight"] = G[origem][destino].get("weight", 0) + 1
    else:
        G.add_edge(origem, destino, relacao=relacao, weight=1)


# ─────────────────────────────────────────────
# ANÁLISE CIENTÍFICA DO GRAFO DE ORIENTADORES
# ─────────────────────────────────────────────
def analisar_grafo_orientadores(G: nx.DiGraph) -> dict:
    """
    Métricas específicas para rede de co-autoria/orientação:
    - produtividade por orientador (nº de orientações)
    - centralidade de intermediação (quem conecta mais grupos)
    - comunidades de pesquisa (Louvain no grafo não-dirigido)
    - orientadores que transitam entre departamentos
    """
    print("\n" + "═"*60)
    print("  ANÁLISE DO GRAFO DE ORIENTADORES")
    print("═"*60)

    n_nos     = G.number_of_nodes()
    n_arestas = G.number_of_edges()
    print(f"  Nós     : {n_nos}")
    print(f"  Arestas : {n_arestas}")

    # contagens por tipo de nó
    tipos = defaultdict(int)
    for _, d in G.nodes(data=True):
        tipos[d.get("tipo","?")] += 1
    print("\n  Composição da rede:")
    labels = {"PER_orientador":"Orientadores","PER_autor":"Autores",
               "WORK":"Trabalhos","ORG":"Departamentos"}
    for tipo, qtd in sorted(tipos.items(), key=lambda x: x[1], reverse=True):
        print(f"    {labels.get(tipo, tipo):<20} → {qtd}")

    # produtividade: orientadores ordenados por nº de orientações
    orientadores = [n for n, d in G.nodes(data=True)
                    if d.get("tipo") == "PER_orientador"]
    prod = {}
    for ori in orientadores:
        alunos     = [v for u, v, d in G.out_edges(ori, data=True)
                      if d.get("relacao") == "orientou"]
        trabalhos  = [v for u, v, d in G.out_edges(ori, data=True)
                      if d.get("relacao") == "orientou_trabalho"]
        prod[ori]  = {"alunos": len(alunos), "trabalhos": len(trabalhos)}

    print("\n  Produtividade dos orientadores (top 15):")
    print(f"  {'Orientador':<40} {'Alunos':>7} {'Trabalhos':>10}")
    print("  " + "-"*60)
    for ori, p in sorted(prod.items(),
                         key=lambda x: x[1]["trabalhos"], reverse=True)[:15]:
        print(f"  {ori:<40} {p['alunos']:>7} {p['trabalhos']:>10}")

    # centralidade de intermediação (no grafo não-dirigido)
    G_und = G.to_undirected()
    between = nx.betweenness_centrality(G_und, weight="weight", normalized=True)

    print("\n  Top 10 por betweenness (pontes entre grupos):")
    top_bet = sorted(between.items(), key=lambda x: x[1], reverse=True)[:10]
    for no, val in top_bet:
        tipo = G.nodes[no].get("tipo", "?")
        print(f"    {no:<40} [{labels.get(tipo,tipo)}]  {val:.4f}")

    # in-degree dos orientadores (quantos trabalhos chegam até eles)
    print("\n  In-degree dos orientadores (trabalhos recebidos):")
    in_deg = {n: G.in_degree(n) for n in orientadores}
    for no, deg in sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {no:<40}  in-degree={deg}")

    # detecção de comunidades (Louvain)
    try:
        from networkx.algorithms.community import louvain_communities
        comunidades = louvain_communities(G_und, seed=42)
        print(f"\n  Comunidades detectadas (Louvain): {len(comunidades)}")
        for i, com in enumerate(
            sorted(comunidades, key=len, reverse=True)[:5]
        ):
            membros_ori = [n for n in com
                           if G.nodes[n].get("tipo") == "PER_orientador"]
            print(f"    Comunidade {i+1}: {len(com)} nós  "
                  f"| orientadores: {', '.join(list(membros_ori)[:4])}")
    except Exception as e:
        comunidades = []
        print(f"\n  Comunidades: não disponível ({e})")

    # co-orientações (rede entre orientadores)
    co_ori = [(u, v) for u, v, d in G.edges(data=True)
              if d.get("relacao") == "co-orientou"]
    print(f"\n  Co-orientações registradas: {len(co_ori)}")
    for u, v in co_ori[:10]:
        print(f"    {u}  ↔  {v}")

    return {
        "nos"           : n_nos,
        "arestas"       : n_arestas,
        "tipos"         : dict(tipos),
        "produtividade" : prod,
        "betweenness"   : dict(top_bet),
        "comunidades"   : len(comunidades),
        "co_orientacoes": co_ori,
    }


# ─────────────────────────────────────────────
# VISUALIZAÇÃO INTERATIVA (pyvis)
# ─────────────────────────────────────────────
def visualizar_orientadores_interativo(
    G: nx.DiGraph,
    salvar_html: str = "grafo_orientadores.html",
) -> str:
    net = Network(
        height="800px", width="100%",
        bgcolor="#0f0f23", font_color="#e0e0e0",
        directed=True, notebook=False,
    )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.005,
          "springLength": 150,
          "springConstant": 0.06,
          "damping": 0.4,
          "avoidOverlap": 0.8
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "enabled": true, "iterations": 400, "fit": true }
      },
      "interaction": {
        "hover": true, "navigationButtons": true,
        "dragNodes": true, "zoomView": true, "tooltipDelay": 80
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
        "smooth": { "type": "curvedCW", "roundness": 0.2 }
      }
    }
    """)

    for no, dados in G.nodes(data=True):
        tipo  = dados.get("tipo", "?")
        freq  = dados.get("count", 1)
        cor   = COR_NO.get(tipo, "#9E9E9E")
        forma = FORMA_NO.get(tipo, "dot")
        tam   = 30 if tipo == "PER_orientador" else \
                20 if tipo == "PER_autor" else \
                12 if tipo == "WORK" else 18

        titulo_html = f"<b>{no}</b><br>Tipo: {tipo}<br>Freq: {freq}"
        if tipo == "WORK":
            autor_info = dados.get("autor", "?")
            ori_info   = dados.get("orientador", "?")
            titulo_html += f"<br>Autor: {autor_info}<br>Orientador: {ori_info}"

        net.add_node(
            no, label=no if tipo != "WORK" else no[:30] + "…" if len(no) > 30 else no,
            title=titulo_html, color=cor, shape=forma, size=tam,
            font={"size": 11 if tipo == "PER_orientador" else 8,
                  "color": "#ffffff"},
        )

    cores_aresta = {
        "orientou"          : "#FFD700",
        "orientou_trabalho" : "#FFD70088",
        "escreveu"          : "#4CAF5088",
        "pertence_a"        : "#FF572288",
        "co-orientou"       : "#E91E6388",
        "co-orientou_aluno" : "#E91E6344",
    }

    for u, v, dados in G.edges(data=True):
        rel = dados.get("relacao", "")
        peso = dados.get("weight", 1)
        net.add_edge(
            u, v,
            title=f"{rel} (peso={peso})",
            color=cores_aresta.get(rel, "#555555"),
            width=min(1 + peso, 6),
        )

    net.save_graph(salvar_html)
    print(f"  ✓ Grafo orientadores → {salvar_html}")
    return salvar_html


# ─────────────────────────────────────────────
# FIGURA ESTÁTICA
# ─────────────────────────────────────────────
def figura_orientadores(
    G: nx.DiGraph,
    salvar: str = "grafo_orientadores.png",
    dpi: int = 300,
    apenas_orientadores: bool = False,
) -> str:
    """
    Figura estática focada nas relações entre orientadores.
    Se apenas_orientadores=True, mostra só a rede orientador↔orientador
    via co-orientações (útil para análise de colaboração).
    """
    if apenas_orientadores:
        nos_ori = [n for n, d in G.nodes(data=True)
                   if d.get("tipo") == "PER_orientador"]
        G = G.subgraph(nos_ori).copy()
        titulo = "Rede de Co-orientação entre Orientadores"
    else:
        # limita: mostra orientadores + autores (remove trabalhos longos)
        nos_viz = [n for n, d in G.nodes(data=True)
                   if d.get("tipo") in ("PER_orientador","PER_autor","ORG")]
        G = G.subgraph(nos_viz).copy()
        titulo = "Rede Orientador ↔ Autor ↔ Departamento"

    if G.number_of_nodes() == 0:
        print("  Grafo vazio — nenhum metadado extraído.")
        return ""

    pos = nx.spring_layout(G, seed=42, k=2.5)

    cores  = [COR_NO.get(G.nodes[n].get("tipo","?"), "#9E9E9E") for n in G.nodes()]
    tam    = [40 if G.nodes[n].get("tipo")=="PER_orientador" else 20 for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(18, 13), facecolor="#0f0f23")
    ax.set_facecolor("#0f0f23")

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#4a4a6a",
                           alpha=0.5, arrows=True,
                           arrowstyle="-|>", arrowsize=12)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cores,
                           node_size=tam, alpha=0.92,
                           linewidths=0.5, edgecolors="#ffffff")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7,
                            font_color="#ffffff", font_weight="bold")

    handles = [mpatches.Patch(color=cor, label=tipo.replace("PER_",""))
               for tipo, cor in COR_NO.items()]
    ax.legend(handles=handles, loc="upper left", framealpha=0.3,
              facecolor="#1a1a2e", edgecolor="#ffffff",
              labelcolor="#ffffff", fontsize=10)

    ax.set_title(titulo, color="#e0e0e0", fontsize=14, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(salvar, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Figura orientadores → {salvar}")
    return salvar