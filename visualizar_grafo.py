"""
visualizar_grafo.py
Visualização do grafo de co-ocorrência NER:
  - pyvis      → HTML interativo (física dinâmica)
  - matplotlib → figuras estáticas PNG/PDF para o relatório
"""

import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyvis.network import Network

# ─────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────
COR_TIPO = {
    "PER":  "#4CAF50",
    "ORG":  "#2196F3",
    "LOC":  "#FF5722",
    "MISC": "#9C27B0",
    "?":    "#9E9E9E",
}
FORMA_TIPO = {
    "PER":  "dot",
    "ORG":  "square",
    "LOC":  "triangle",
    "MISC": "diamond",
    "?":    "dot",
}


# ─────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────
def _escalar(v, vmin, vmax, smin, smax):
    if vmax == vmin:
        return (smin + smax) / 2
    return smin + (v - vmin) / (vmax - vmin) * (smax - smin)

def _tamanhos_nos(G, smin=10, smax=60):
    counts = {n: G.nodes[n].get("count", 1) for n in G.nodes()}
    vmin, vmax = min(counts.values()), max(counts.values())
    return {n: _escalar(v, vmin, vmax, smin, smax) for n, v in counts.items()}

def _espessuras_arestas(G, smin=1, smax=10):
    pesos = {(u, v): d.get("weight", 1) for u, v, d in G.edges(data=True)}
    if not pesos:
        return pesos
    vmin, vmax = min(pesos.values()), max(pesos.values())
    return {k: _escalar(v, vmin, vmax, smin, smax) for k, v in pesos.items()}

def _opcoes_fisica_barnes():
    return """
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 130,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.2
        },
        "stabilization": {
          "enabled": true,
          "iterations": 300,
          "updateInterval": 25,
          "fit": true
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": true,
        "keyboard": true,
        "dragNodes": true,
        "zoomView": true
      },
      "edges": { "smooth": { "type": "continuous" } }
    }
    """

def _opcoes_fisica_atlas():
    return """
    {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.005,
          "springLength": 120,
          "springConstant": 0.08,
          "damping": 0.4,
          "avoidOverlap": 0.5
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 300,
          "updateInterval": 25,
          "fit": true
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": true,
        "dragNodes": true,
        "zoomView": true
      },
      "edges": { "smooth": { "type": "continuous" } }
    }
    """


# ═════════════════════════════════════════════
# 1. GRAFO INTERATIVO COMPLETO (pyvis)
# ═════════════════════════════════════════════
def visualizar_grafo_interativo(
    G: nx.Graph,
    salvar_html: str = "grafo_entidades.html",
    altura: str = "750px",
    largura: str = "100%",
    filtro_peso_min: int = 2,
) -> str:
    """
    HTML interativo com física dinâmica Barnes-Hut.
    set_options é chamado ANTES de add_node para garantir física ativa.
    """
    net = Network(
        height=altura,
        width=largura,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        directed=False,
        notebook=False,
    )

    # ← física ANTES dos dados
    net.set_options(_opcoes_fisica_barnes())

    tamanhos   = _tamanhos_nos(G)
    espessuras = _espessuras_arestas(G)

    for no, dados in G.nodes(data=True):
        tipo = dados.get("tipo", "?")
        freq = dados.get("count", 1)
        tam  = tamanhos.get(no, 15)
        net.add_node(
            no,
            label=no,
            title=f"<b>{no}</b><br>Tipo: {tipo}<br>Freq: {freq}",
            color=COR_TIPO.get(tipo, COR_TIPO["?"]),
            shape=FORMA_TIPO.get(tipo, "dot"),
            size=tam,
            font={"size": max(9, int(tam * 0.38)), "color": "#ffffff"},
        )

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

    net.save_graph(salvar_html)
    print(f"  ✓ Grafo interativo → {salvar_html}")
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
    HTML interativo do ego-grafo.
    Nó central em dourado; 1º grau em cor plena; 2º grau transparente.
    set_options chamado ANTES dos dados.
    """
    if no_central not in G:
        print(f"  Nó '{no_central}' não encontrado.")
        return ""

    subg = nx.ego_graph(G, no_central, radius=raio)
    if salvar_html is None:
        salvar_html = f"ego_{no_central.replace(' ', '_')}.html"

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#0f0f23",
        font_color="#e0e0e0",
        directed=False,
        notebook=False,
    )

    # ← física ANTES dos dados
    net.set_options(_opcoes_fisica_atlas())

    tamanhos         = _tamanhos_nos(subg)
    espessuras       = _espessuras_arestas(subg)
    vizinhos_diretos = set(G.neighbors(no_central))

    for no, dados in subg.nodes(data=True):
        tipo = dados.get("tipo", "?")
        freq = dados.get("count", 1)

        if no == no_central:
            cor, tam = "#FFD700", 55
            borda = {"width": 4, "color": "#FF8C00"}
        elif no in vizinhos_diretos:
            cor  = COR_TIPO.get(tipo, "#9E9E9E")
            tam  = tamanhos.get(no, 20)
            borda = {"width": 2, "color": "#ffffff"}
        else:
            cor  = COR_TIPO.get(tipo, "#9E9E9E") + "88"
            tam  = max(8, tamanhos.get(no, 12) * 0.65)
            borda = {"width": 1, "color": "#555555"}

        net.add_node(
            no,
            label=no,
            title=f"<b>{no}</b><br>Tipo: {tipo}<br>Freq: {freq}",
            color=cor,
            shape=FORMA_TIPO.get(tipo, "dot"),
            size=tam,
            borderWidth=borda["width"],
            borderWidthSelected=borda["width"] + 2,
            font={"size": max(9, int(tam * 0.35)), "color": "#ffffff"},
        )

    for u, v, dados in subg.edges(data=True):
        peso     = dados.get("weight", 1)
        esp      = espessuras.get((u, v), espessuras.get((v, u), 1))
        destaque = (u == no_central or v == no_central)
        net.add_edge(
            u, v,
            value=peso,
            title=f"Peso: {peso}",
            width=esp,
            color={
                "color":     "#FFD700" if destaque else "#3a3a5a",
                "highlight": "#ffffff",
            },
        )

    net.save_graph(salvar_html)
    print(f"  ✓ Ego-grafo '{no_central}' → {salvar_html}")

    if abrir_browser:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(salvar_html)}")

    return salvar_html


# ═════════════════════════════════════════════
# 3. FIGURAS ESTÁTICAS (matplotlib)
# ═════════════════════════════════════════════

def _layout_inteligente(G):
    n = G.number_of_nodes()
    if n <= 30:
        return nx.spring_layout(G, seed=42, k=2.5)
    elif n <= 120:
        return nx.kamada_kawai_layout(G)
    return nx.spring_layout(G, seed=42, k=1.5, iterations=40)


def figura_grafo_completo(
    G: nx.Graph,
    salvar: str = "figura_grafo.png",
    dpi: int = 300,
    top_nos: int = 80,
) -> str:
    """Figura estática do grafo (top_nos nós por grau)."""
    if top_nos and G.number_of_nodes() > top_nos:
        deg     = dict(G.degree())
        nos_top = sorted(deg, key=deg.get, reverse=True)[:top_nos]
        G = G.subgraph(nos_top).copy()

    tamanhos   = _tamanhos_nos(G, 150, 1800)
    espessuras = _espessuras_arestas(G, 0.4, 5)
    pos        = _layout_inteligente(G)

    cores_nos    = [COR_TIPO.get(G.nodes[n].get("tipo","?"), "#9E9E9E") for n in G.nodes()]
    tam_nos      = [tamanhos[n] for n in G.nodes()]
    larg_arestas = [espessuras.get((u,v), espessuras.get((v,u), 1)) for u,v in G.edges()]

    fig, ax = plt.subplots(figsize=(18, 13), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    nx.draw_networkx_edges(G, pos, ax=ax, width=larg_arestas,
                           edge_color="#4a4a6a", alpha=0.55)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cores_nos,
                           node_size=tam_nos, alpha=0.93,
                           linewidths=0.5, edgecolors="#ffffff")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7,
                            font_color="#ffffff", font_weight="bold")

    handles = [mpatches.Patch(color=cor, label=tipo)
               for tipo, cor in COR_TIPO.items() if tipo != "?"]
    ax.legend(handles=handles, loc="upper left", framealpha=0.3,
              facecolor="#2a2a4e", edgecolor="#ffffff",
              labelcolor="#ffffff", fontsize=10)

    estrategia = G.graph.get("estrategia", "")
    ax.set_title(f"Grafo de Co-ocorrência NER  |  estratégia: {estrategia}",
                 color="#e0e0e0", fontsize=14, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(salvar, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Figura → {salvar}")
    return salvar


def figura_ego(
    G: nx.Graph,
    no_central: str,
    raio: int = 2,
    salvar: str = None,
    dpi: int = 300,
) -> str:
    """Figura estática do ego-grafo."""
    if no_central not in G:
        print(f"  Nó '{no_central}' não encontrado.")
        return ""

    subg = nx.ego_graph(G, no_central, radius=raio)
    if salvar is None:
        salvar = f"ego_{no_central.replace(' ','_')}.png"

    tamanhos         = _tamanhos_nos(subg, 250, 2200)
    espessuras       = _espessuras_arestas(subg, 0.5, 6)
    pos              = nx.spring_layout(subg, seed=42, k=2.2)
    vizinhos_diretos = set(G.neighbors(no_central))

    cores = []
    for n in subg.nodes():
        if n == no_central:
            cores.append("#FFD700")
        elif n in vizinhos_diretos:
            cores.append(COR_TIPO.get(subg.nodes[n].get("tipo","?"), "#9E9E9E"))
        else:
            cores.append("#444466")

    tam_nos      = [tamanhos[n] for n in subg.nodes()]
    larg_arestas = [espessuras.get((u,v), espessuras.get((v,u), 1))
                    for u,v in subg.edges()]
    edge_colors  = ["#FFD700" if (u==no_central or v==no_central) else "#3a3a5a"
                    for u,v in subg.edges()]

    fig, ax = plt.subplots(figsize=(13, 11), facecolor="#0f0f23")
    ax.set_facecolor("#0f0f23")
    nx.draw_networkx_edges(subg, pos, ax=ax, width=larg_arestas,
                           edge_color=edge_colors, alpha=0.72)
    nx.draw_networkx_nodes(subg, pos, ax=ax, node_color=cores,
                           node_size=tam_nos, linewidths=1, edgecolors="#ffffff")
    nx.draw_networkx_labels(subg, pos, ax=ax, font_size=8,
                            font_color="#ffffff", font_weight="bold")
    ax.set_title(f"Ego-grafo: '{no_central}'  (raio={raio})",
                 color="#e0e0e0", fontsize=13, pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(salvar, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Ego-figura → {salvar}")
    return salvar


def figura_comparativa(
    resultados: dict,
    salvar: str = "comparacao_estrategias.png",
    dpi: int = 300,
) -> str:
    """
    6 subplots: nós, arestas, densidade, clustering, componentes, diâmetro.
    Inclui linha de referência de densidade adequada (0.005–0.15).
    """
    nomes     = list(resultados.keys())
    def _m(k): return [resultados[e]["metricas"].get(k, 0) for e in nomes]

    nos        = _m("nos")
    arestas    = _m("arestas")
    densidade  = _m("densidade")
    cluster    = _m("clustering_medio")
    comp       = _m("componentes")
    diametro   = [v if v else 0 for v in _m("diametro")]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#1a1a2e")
    fig.suptitle("Comparação de Estratégias de Janela — NER Co-ocorrência",
                 color="#e0e0e0", fontsize=15, y=1.02)

    cores_bar = ["#4CAF50", "#2196F3", "#FF5722"]

    dados_plots = [
        (axes[0,0], nos,       "Nós",               None),
        (axes[0,1], arestas,   "Arestas",            None),
        (axes[0,2], densidade, "Densidade",          (0.005, 0.15)),  # faixa ideal
        (axes[1,0], cluster,   "Clustering Médio",   None),
        (axes[1,1], comp,      "Componentes",        None),
        (axes[1,2], diametro,  "Diâmetro",           None),
    ]

    for ax, valores, titulo, faixa in dados_plots:
        ax.set_facecolor("#16213e")
        bars = ax.bar(nomes, valores, color=cores_bar,
                      edgecolor="#ffffff", linewidth=0.5, alpha=0.85)
        ax.set_title(titulo, color="#e0e0e0", fontsize=11)
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333355")

        # faixa de referência de densidade
        if faixa:
            ax.axhspan(faixa[0], faixa[1], alpha=0.15,
                       color="#00ff88", label="faixa ideal")
            ax.legend(fontsize=8, labelcolor="#aaaaaa",
                      facecolor="#1a1a2e", edgecolor="#333355")

        for bar, val in zip(bars, valores):
            label = f"{val:.4f}" if isinstance(val, float) else str(val)
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.02,
                    label, ha="center", va="bottom",
                    color="#ffffff", fontsize=9)

    plt.tight_layout()
    plt.savefig(salvar, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Comparativa → {salvar}")
    return salvar


def figura_distribuicao_grau(
    G: nx.Graph,
    salvar: str = "distribuicao_grau.png",
    dpi: int = 300,
) -> str:
    """
    3 subplots: histograma linear, log-log e CCDF (Complementary CDF).
    CCDF é mais robusto que log-log para detectar scale-free.
    """
    graus = sorted([d for _, d in G.degree()])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333355")

    # 1. Linear
    axes[0].hist(graus, bins=30, color="#2196F3", edgecolor="#0a0a20", alpha=0.85)
    axes[0].set_title("Distribuição de Grau (linear)", color="#e0e0e0", fontsize=11)
    axes[0].set_xlabel("Grau", color="#aaaaaa")
    axes[0].set_ylabel("Frequência", color="#aaaaaa")

    # 2. Log-log
    graus_pos = np.array([g for g in graus if g > 0])
    axes[1].hist(graus_pos, bins=30, color="#FF5722",
                 edgecolor="#0a0a20", alpha=0.85, log=True)
    axes[1].set_xscale("log")
    axes[1].set_title("Distribuição de Grau (log-log)", color="#e0e0e0", fontsize=11)
    axes[1].set_xlabel("Grau (log)", color="#aaaaaa")
    axes[1].set_ylabel("Frequência (log)", color="#aaaaaa")

    # 3. CCDF — melhor para identificar scale-free
    unique, counts = np.unique(graus_pos, return_counts=True)
    cdf  = np.cumsum(counts) / counts.sum()
    ccdf = 1 - cdf
    axes[2].loglog(unique, ccdf, "o-", color="#9C27B0",
                   markersize=4, linewidth=1.5, alpha=0.85)
    axes[2].set_title("CCDF de Grau (log-log)\nline reta → scale-free",
                      color="#e0e0e0", fontsize=11)
    axes[2].set_xlabel("Grau (log)", color="#aaaaaa")
    axes[2].set_ylabel("P(X > k)", color="#aaaaaa")
    axes[2].grid(True, alpha=0.2, color="#555555")

    estrategia = G.graph.get("estrategia", "")
    fig.suptitle(f"Distribuição de Grau — {estrategia}",
                 color="#e0e0e0", fontsize=13)
    plt.tight_layout()
    plt.savefig(salvar, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Distribuição de grau → {salvar}")
    return salvar