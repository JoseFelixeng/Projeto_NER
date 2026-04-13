"""
visualizar_grafo.py
Visualização do grafo de co-ocorrência NER usando:
  - pyvis  → HTML interativo (exploração no navegador)
  - matplotlib + networkx → figuras estáticas (PNG/PDF para o relatório)
"""

import os
import math
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pyvis.network import Network

# ─────────────────────────────────────────────
# PALETA DE CORES POR TIPO NER
# ─────────────────────────────────────────────
COR_TIPO = {
    "PER":  "#4CAF50",   # verde  — pessoas
    "ORG":  "#2196F3",   # azul   — organizações
    "LOC":  "#FF5722",   # laranja— locais
    "MISC": "#9C27B0",   # roxo   — miscelânea
    "?":    "#9E9E9E",   # cinza  — desconhecido
}

FORMA_TIPO = {
    "PER":  "dot",
    "ORG":  "square",
    "LOC":  "triangle",
    "MISC": "diamond",
    "?":    "dot",
}

# ─────────────────────────────────────────────
# UTILITÁRIO — escala de tamanho por frequência
# ─────────────────────────────────────────────
def _escalar(valor: float, vmin: float, vmax: float,
             saida_min: float = 10, saida_max: float = 60) -> float:
    if vmax == vmin:
        return (saida_min + saida_max) / 2
    return saida_min + (valor - vmin) / (vmax - vmin) * (saida_max - saida_min)


def _tamanhos_nos(G: nx.Graph, saida_min=10, saida_max=60):
    counts = {n: G.nodes[n].get("count", 1) for n in G.nodes()}
    vmin, vmax = min(counts.values()), max(counts.values())
    return {n: _escalar(v, vmin, vmax, saida_min, saida_max) for n, v in counts.items()}


def _espessuras_arestas(G: nx.Graph, saida_min=1, saida_max=10):
    pesos = {(u, v): d.get("weight", 1) for u, v, d in G.edges(data=True)}
    if not pesos:
        return pesos
    vmin, vmax = min(pesos.values()), max(pesos.values())
    return {k: _escalar(v, vmin, vmax, saida_min, saida_max) for k, v in pesos.items()}


# ═════════════════════════════════════════════
# 1. GRAFO INTERATIVO COMPLETO (pyvis)
# ═════════════════════════════════════════════
def visualizar_grafo_interativo(
    G: nx.Graph,
    salvar_html: str = "grafo_entidades.html",
    altura: str = "750px",
    largura: str = "100%",
    fisica: bool = True,
    filtro_peso_min: int = 1,
) -> str:
    """
    Gera um HTML interativo com pyvis.

    Recursos:
    - nós coloridos e com forma por tipo NER
    - tamanho proporcional à frequência do nó
    - espessura da aresta proporcional ao peso
    - painel de filtros e física habilitado
    - tooltip com tipo e frequência ao passar o mouse

    Parâmetros
    ----------
    G               : grafo NetworkX
    salvar_html     : caminho de saída do arquivo HTML
    altura / largura: dimensões do canvas
    fisica          : ativa simulação física (Barnes-Hut)
    filtro_peso_min : remove arestas com peso abaixo deste valor

    Retorna o caminho do HTML gerado.
    """
    net = Network(
        height=altura,
        width=largura,
        bgcolor="#1a1a2e",       # fundo escuro para contraste
        font_color="#e0e0e0",
        directed=False,
        notebook=False,
        select_menu=True,        # menu de seleção de nós/arestas
        filter_menu=True,        # filtros por atributo
    )

    tamanhos  = _tamanhos_nos(G)
    espessuras = _espessuras_arestas(G)

    # — nós —
    for no, dados in G.nodes(data=True):
        tipo  = dados.get("tipo", "?")
        freq  = dados.get("count", 1)
        cor   = COR_TIPO.get(tipo, COR_TIPO["?"])
        forma = FORMA_TIPO.get(tipo, "dot")
        tam   = tamanhos.get(no, 15)

        net.add_node(
            no,
            label=no,
            title=f"<b>{no}</b><br>Tipo: {tipo}<br>Freq: {freq}",
            color=cor,
            shape=forma,
            size=tam,
            font={"size": max(10, int(tam * 0.4)), "color": "#ffffff"},
        )

    # — arestas —
    for u, v, dados in G.edges(data=True):
        peso = dados.get("weight", 1)
        if peso < filtro_peso_min:
            continue
        esp = espessuras.get((u, v), espessuras.get((v, u), 1))
        net.add_edge(
            u, v,
            value=peso,
            title=f"Co-ocorrências: {peso}",
            width=esp,
            color={"color": "#4a4a6a", "highlight": "#ffffff"},
        )

    # — física Barnes-Hut —
    if fisica:
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 120,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "stabilization": { "iterations": 200 }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)

    net.save_graph(salvar_html)
    print(f"  ✓ Grafo interativo salvo → {salvar_html}")
    return salvar_html


# ═════════════════════════════════════════════
# 2. EGO GRAPH INTERATIVO (pyvis)
# ═════════════════════════════════════════════
def visualizar_ego_interativo(
    G: nx.Graph,
    no_central: str,
    raio: int = 2,
    salvar_html: str = None,
    abrir_browser: bool = True,
) -> str:
    """
    Gera HTML interativo focado no ego-grafo de um nó.

    O nó central fica destacado em amarelo dourado.
    Vizinhos de 1º grau em cor plena; 2º grau em versão mais clara.
    """
    if no_central not in G:
        print(f"  Nó '{no_central}' não encontrado no grafo.")
        return ""

    subg = nx.ego_graph(G, no_central, radius=raio)

    if salvar_html is None:
        salvar_html = f"ego_{no_central.replace(' ', '_')}.html"

    net = Network(
        height="700px",
        width="100%",
        bgcolor="#0f0f23",
        font_color="#e0e0e0",
        directed=False,
        notebook=False,
    )

    tamanhos  = _tamanhos_nos(subg)
    espessuras = _espessuras_arestas(subg)
    vizinhos_diretos = set(G.neighbors(no_central))

    for no, dados in subg.nodes(data=True):
        tipo = dados.get("tipo", "?")
        freq = dados.get("count", 1)

        if no == no_central:
            cor, tam, borda = "#FFD700", 50, {"width": 4, "color": "#FF8C00"}
        elif no in vizinhos_diretos:
            cor, tam, borda = COR_TIPO.get(tipo, "#9E9E9E"), tamanhos.get(no, 15), {"width": 2, "color": "#ffffff"}
        else:
            cor_base = COR_TIPO.get(tipo, "#9E9E9E")
            cor = cor_base + "88"   # transparência para 2º grau
            tam, borda = tamanhos.get(no, 10) * 0.7, {"width": 1, "color": "#555555"}

        net.add_node(
            no,
            label=no,
            title=f"<b>{no}</b><br>Tipo: {tipo}<br>Freq: {freq}",
            color=cor,
            shape=FORMA_TIPO.get(tipo, "dot"),
            size=tam,
            borderWidth=borda["width"],
            borderWidthSelected=borda["width"] + 2,
        )

    for u, v, dados in subg.edges(data=True):
        peso = dados.get("weight", 1)
        esp  = espessuras.get((u, v), espessuras.get((v, u), 1))
        destaque = u == no_central or v == no_central
        net.add_edge(
            u, v,
            value=peso,
            title=f"Peso: {peso}",
            width=esp,
            color={"color": "#FFD700" if destaque else "#3a3a5a",
                   "highlight": "#ffffff"},
        )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based"
      },
      "interaction": { "hover": true, "navigationButtons": true }
    }
    """)

    net.save_graph(salvar_html)
    print(f"  ✓ Ego-grafo '{no_central}' salvo → {salvar_html}")

    if abrir_browser:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(salvar_html)}")

    return salvar_html


# ═════════════════════════════════════════════
# 3. FIGURAS ESTÁTICAS (matplotlib) — para relatório
# ═════════════════════════════════════════════

def _layout_inteligente(G: nx.Graph):
    """Escolhe layout baseado no tamanho do grafo."""
    n = G.number_of_nodes()
    if n <= 30:
        return nx.spring_layout(G, seed=42, k=2.5)
    elif n <= 100:
        return nx.kamada_kawai_layout(G)
    else:
        return nx.spring_layout(G, seed=42, k=1.5, iterations=30)


def figura_grafo_completo(
    G: nx.Graph,
    salvar: str = "figura_grafo.png",
    dpi: int = 300,
    top_nos: int = None,
) -> str:
    """
    Figura estática do grafo completo (ou top_nos nós por grau).
    Salva em PNG/PDF dependendo da extensão de `salvar`.
    """
    if top_nos:
        # filtra apenas os top_nos nós mais centrais
        deg = dict(G.degree())
        nos_top = sorted(deg, key=deg.get, reverse=True)[:top_nos]
        G = G.subgraph(nos_top).copy()

    tamanhos   = _tamanhos_nos(G, saida_min=100, saida_max=1500)
    espessuras = _espessuras_arestas(G, saida_min=0.5, saida_max=5)
    pos        = _layout_inteligente(G)

    cores_nos = [COR_TIPO.get(G.nodes[n].get("tipo", "?"), "#9E9E9E") for n in G.nodes()]
    tam_nos   = [tamanhos[n] for n in G.nodes()]
    larg_arestas = [espessuras.get((u, v), espessuras.get((v, u), 1))
                    for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(16, 12), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=larg_arestas,
        edge_color="#4a4a6a",
        alpha=0.6,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=cores_nos,
        node_size=tam_nos,
        alpha=0.92,
        linewidths=0.5,
        edgecolors="#ffffff",
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=7,
        font_color="#ffffff",
        font_weight="bold",
    )

    # legenda por tipo
    handles = [
        mpatches.Patch(color=cor, label=tipo)
        for tipo, cor in COR_TIPO.items()
        if tipo != "?"
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        framealpha=0.3,
        facecolor="#2a2a4e",
        edgecolor="#ffffff",
        labelcolor="#ffffff",
        fontsize=10,
    )

    estrategia = G.graph.get("estrategia", "")
    titulo = f"Grafo de Co-ocorrência NER"
    if estrategia:
        titulo += f"  |  estratégia: {estrategia}"
    ax.set_title(titulo, color="#e0e0e0", fontsize=14, pad=15)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(salvar, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Figura salva → {salvar}")
    return salvar


def figura_ego(
    G: nx.Graph,
    no_central: str,
    raio: int = 2,
    salvar: str = None,
    dpi: int = 300,
) -> str:
    """
    Figura estática do ego-grafo de um nó específico.
    """
    if no_central not in G:
        print(f"  Nó '{no_central}' não encontrado.")
        return ""

    subg = nx.ego_graph(G, no_central, radius=raio)
    if salvar is None:
        salvar = f"ego_{no_central.replace(' ', '_')}.png"

    tamanhos   = _tamanhos_nos(subg, saida_min=200, saida_max=2000)
    espessuras = _espessuras_arestas(subg, saida_min=0.5, saida_max=6)
    pos        = nx.spring_layout(subg, seed=42, k=2.0)

    vizinhos_diretos = set(G.neighbors(no_central))

    cores = []
    for n in subg.nodes():
        if n == no_central:
            cores.append("#FFD700")
        elif n in vizinhos_diretos:
            cores.append(COR_TIPO.get(subg.nodes[n].get("tipo", "?"), "#9E9E9E"))
        else:
            cores.append("#555577")

    tam_nos      = [tamanhos[n] for n in subg.nodes()]
    larg_arestas = [espessuras.get((u, v), espessuras.get((v, u), 1))
                    for u, v in subg.edges()]

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0f0f23")
    ax.set_facecolor("#0f0f23")

    edge_colors = [
        "#FFD700" if (u == no_central or v == no_central) else "#3a3a5a"
        for u, v in subg.edges()
    ]

    nx.draw_networkx_edges(subg, pos, ax=ax,
                           width=larg_arestas, edge_color=edge_colors, alpha=0.7)
    nx.draw_networkx_nodes(subg, pos, ax=ax,
                           node_color=cores, node_size=tam_nos,
                           linewidths=1, edgecolors="#ffffff")
    nx.draw_networkx_labels(subg, pos, ax=ax,
                            font_size=8, font_color="#ffffff", font_weight="bold")

    ax.set_title(
        f"Ego-grafo: '{no_central}'  (raio={raio})",
        color="#e0e0e0", fontsize=13, pad=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(salvar, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Ego-figura salva → {salvar}")
    return salvar


def figura_comparativa(
    resultados: dict,
    salvar: str = "comparacao_estrategias.png",
    dpi: int = 300,
) -> str:
    """
    Figura com 4 subplots comparando métricas das 3 estratégias:
    nós, arestas, densidade e clustering médio.

    `resultados` é o dict retornado por grafo_ner.comparar_estrategias().
    """
    nomes     = list(resultados.keys())
    nos       = [resultados[e]["metricas"].get("nos", 0)              for e in nomes]
    arestas   = [resultados[e]["metricas"].get("arestas", 0)          for e in nomes]
    densidade = [resultados[e]["metricas"].get("densidade", 0)        for e in nomes]
    cluster   = [resultados[e]["metricas"].get("clustering_medio", 0) for e in nomes]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#1a1a2e")
    fig.suptitle("Comparação de Estratégias de Janela — NER Co-ocorrência",
                 color="#e0e0e0", fontsize=14, y=1.01)

    cores_bar = ["#4CAF50", "#2196F3", "#FF5722"]

    pares = [
        (axes[0, 0], nos,       "Nós",              "#4CAF50"),
        (axes[0, 1], arestas,   "Arestas",           "#2196F3"),
        (axes[1, 0], densidade, "Densidade",         "#FF5722"),
        (axes[1, 1], cluster,   "Clustering Médio",  "#9C27B0"),
    ]

    for ax, valores, titulo, cor in pares:
        ax.set_facecolor("#16213e")
        bars = ax.bar(nomes, valores, color=cores_bar, edgecolor="#ffffff",
                      linewidth=0.5, alpha=0.85)
        ax.set_title(titulo, color="#e0e0e0", fontsize=11)
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333355")
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.4f}" if isinstance(val, float) else str(val),
                    ha="center", va="bottom", color="#ffffff", fontsize=9)

    plt.tight_layout()
    plt.savefig(salvar, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Figura comparativa salva → {salvar}")
    return salvar


def figura_distribuicao_grau(
    G: nx.Graph,
    salvar: str = "distribuicao_grau.png",
    dpi: int = 300,
) -> str:
    """
    Histograma da distribuição de grau dos nós.
    Útil para identificar se a rede segue lei de potência (scale-free).
    """
    graus = [d for _, d in G.degree()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333355")

    # histograma linear
    axes[0].hist(graus, bins=30, color="#2196F3", edgecolor="#0a0a20", alpha=0.85)
    axes[0].set_title("Distribuição de Grau (linear)", color="#e0e0e0", fontsize=11)
    axes[0].set_xlabel("Grau", color="#aaaaaa")
    axes[0].set_ylabel("Frequência", color="#aaaaaa")

    # histograma log-log
    import numpy as np
    graus_np = np.array(graus)
    graus_np = graus_np[graus_np > 0]
    axes[1].hist(graus_np, bins=30, color="#FF5722", edgecolor="#0a0a20",
                 alpha=0.85, log=True)
    axes[1].set_xscale("log")
    axes[1].set_title("Distribuição de Grau (log-log)", color="#e0e0e0", fontsize=11)
    axes[1].set_xlabel("Grau (log)", color="#aaaaaa")
    axes[1].set_ylabel("Frequência (log)", color="#aaaaaa")

    estrategia = G.graph.get("estrategia", "")
    fig.suptitle(f"Distribuição de Grau — {estrategia}", color="#e0e0e0", fontsize=13)

    plt.tight_layout()
    plt.savefig(salvar, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Distribuição de grau salva → {salvar}")
    return salvar