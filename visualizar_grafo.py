"""
ADIÇÃO ao visualizar_grafo.py
Novas funções para o grafo relacional acadêmico.

Cole estas funções no final do seu visualizar_grafo.py existente,
ou importe deste arquivo separado.
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyvis.network import Network

# ─────────────────────────────────────────────
# PALETA DO GRAFO RELACIONAL
# ─────────────────────────────────────────────
COR_TIPO_REL = {
    "UNIV":       "#1565C0",   # azul escuro  — universidade
    "DEPT":       "#0288D1",   # azul médio   — departamento
    "ORIENTADOR": "#2E7D32",   # verde escuro — orientador/professor
    "AUTOR":      "#F57C00",   # laranja      — aluno/autor
    "TRABALHO":   "#6A1B9A",   # roxo         — trabalho/TCC
    "AREA":       "#AD1457",   # rosa         — área temática
}

FORMA_TIPO_REL = {
    "UNIV":       "square",
    "DEPT":       "square",
    "ORIENTADOR": "dot",
    "AUTOR":      "dot",
    "TRABALHO":   "diamond",
    "AREA":       "triangle",
}

TAMANHO_TIPO_REL = {
    "UNIV":       60,
    "DEPT":       45,
    "ORIENTADOR": 40,
    "AUTOR":      30,
    "TRABALHO":   25,
    "AREA":       20,
}


# ═════════════════════════════════════════════
# FIGURA ESTÁTICA — GRAFO RELACIONAL
# ═════════════════════════════════════════════
def figura_grafo_relacional(
    G_rel: nx.DiGraph,
    salvar: str = "grafo_relacional.png",
    dpi: int = 300,
) -> str:
    """
    Figura estática hierárquica do grafo relacional acadêmico.
    Usa layout por camadas: UNIV → DEPT → ORIENTADOR → AUTOR → TRABALHO
    """
    if G_rel.number_of_nodes() == 0:
        print("  [Grafo relacional vazio]")
        return ""

    # Layout por tipo (multi_partite se possível)
    ordem_camadas = ["UNIV", "DEPT", "ORIENTADOR", "AUTOR", "TRABALHO", "AREA"]
    for no in G_rel.nodes():
        t = G_rel.nodes[no].get("tipo", "?")
        G_rel.nodes[no]["subset"] = ordem_camadas.index(t) if t in ordem_camadas else 6

    try:
        pos = nx.multipartite_layout(G_rel, subset_key="subset", scale=3)
    except Exception:
        pos = nx.spring_layout(G_rel, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=(18, 12), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Arestas por tipo de relação
    cores_aresta = {
        "tem_dept":   "#4FC3F7",
        "tem_prof":   "#81C784",
        "orientou":   "#FFB74D",
        "coorientou": "#FF8A65",
        "produziu":   "#CE93D8",
        "tema":       "#F48FB1",
        "aborda":     "#80CBC4",
    }
    for u, v, dados in G_rel.edges(data=True):
        rel = dados.get("relacao", "?")
        cor = cores_aresta.get(rel, "#555577")
        nx.draw_networkx_edges(
            G_rel, pos, edgelist=[(u, v)], ax=ax,
            edge_color=cor, alpha=0.6, width=1.5,
            arrows=True, arrowsize=15,
            connectionstyle="arc3,rad=0.1",
        )

    # Nós por tipo
    for tipo in ordem_camadas:
        nos_tipo = [n for n in G_rel.nodes() if G_rel.nodes[n].get("tipo") == tipo]
        if not nos_tipo:
            continue
        nx.draw_networkx_nodes(
            G_rel, pos, nodelist=nos_tipo, ax=ax,
            node_color=COR_TIPO_REL.get(tipo, "#9E9E9E"),
            node_size=TAMANHO_TIPO_REL.get(tipo, 25) * 20,
            alpha=0.92, linewidths=0.5, edgecolors="#ffffff",
        )

    # Labels (labels legíveis, não os IDs internos)
    labels = {n: G_rel.nodes[n].get("label", n.split("::")[-1]) for n in G_rel.nodes()}
    # Trunca labels longos
    labels = {k: (v[:20] + "…" if len(v) > 22 else v) for k, v in labels.items()}
    nx.draw_networkx_labels(
        G_rel, pos, labels=labels, ax=ax,
        font_size=7, font_color="#ffffff", font_weight="bold",
    )

    # Legenda de nós
    handles_nos = [
        mpatches.Patch(color=cor, label=tipo)
        for tipo, cor in COR_TIPO_REL.items()
    ]
    # Legenda de arestas
    handles_arestas = [
        mpatches.Patch(color=cor, label=rel)
        for rel, cor in cores_aresta.items()
    ]
    leg1 = ax.legend(
        handles=handles_nos, loc="upper left",
        title="Tipo de nó", title_fontsize=9,
        framealpha=0.3, facecolor="#2a2a4e",
        edgecolor="#ffffff", labelcolor="#ffffff", fontsize=8,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=handles_arestas, loc="lower left",
        title="Relação", title_fontsize=9,
        framealpha=0.3, facecolor="#2a2a4e",
        edgecolor="#ffffff", labelcolor="#ffffff", fontsize=8,
    )

    ax.set_title(
        "Grafo Relacional Acadêmico  |  UNIV → DEPT → ORIENTADOR → AUTOR → TRABALHO",
        color="#e0e0e0", fontsize=13, pad=15,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(salvar, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Grafo relacional salvo → {salvar}")
    return salvar


# ═════════════════════════════════════════════
# HTML INTERATIVO — GRAFO RELACIONAL (pyvis)
# ═════════════════════════════════════════════
def visualizar_grafo_relacional_interativo(
    G_rel: nx.DiGraph,
    salvar_html: str = "grafo_relacional.html",
    altura: str = "800px",
) -> str:
    """
    HTML interativo do grafo relacional com pyvis.
    Nós coloridos por tipo; arestas com label da relação.
    """
    net = Network(
        height=altura, width="100%",
        bgcolor="#1a1a2e", font_color="#e0e0e0",
        directed=True, notebook=False,
        select_menu=True, filter_menu=True,
    )

    for no, dados in G_rel.nodes(data=True):
        tipo  = dados.get("tipo", "?")
        label = dados.get("label", no.split("::")[-1])
        freq  = dados.get("count", 1)
        cor   = COR_TIPO_REL.get(tipo, "#9E9E9E")
        forma = FORMA_TIPO_REL.get(tipo, "dot")
        tam   = TAMANHO_TIPO_REL.get(tipo, 20)

        net.add_node(
            no,
            label=label[:30],
            title=f"<b>{label}</b><br>Tipo: {tipo}<br>Freq: {freq}",
            color=cor, shape=forma, size=tam,
            font={"size": max(9, tam // 3), "color": "#ffffff"},
        )

    cores_aresta_hex = {
        "tem_dept":   "#4FC3F7",
        "tem_prof":   "#81C784",
        "orientou":   "#FFB74D",
        "coorientou": "#FF8A65",
        "produziu":   "#CE93D8",
        "tema":       "#F48FB1",
        "aborda":     "#80CBC4",
    }

    for u, v, dados in G_rel.edges(data=True):
        rel  = dados.get("relacao", "?")
        peso = dados.get("peso", 1)
        cor  = cores_aresta_hex.get(rel, "#555577")
        net.add_edge(
            u, v,
            title=f"{rel}  (peso={peso})",
            label=rel,
            color={"color": cor, "highlight": "#ffffff"},
            width=max(1, peso),
            arrows="to",
            font={"size": 8, "color": cor},
        )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "hierarchicalRepulsion": {
          "centralGravity": 0.5,
          "springLength": 150,
          "nodeDistance": 200
        },
        "solver": "hierarchicalRepulsion",
        "stabilization": { "iterations": 300 }
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 200,
          "nodeSpacing": 120
        }
      },
      "interaction": {
        "hover": true, "navigationButtons": true,
        "tooltipDelay": 100, "keyboard": true
      },
      "edges": { "smooth": { "type": "curvedCW", "roundness": 0.2 } }
    }
    """)

    net.save_graph(salvar_html)
    print(f"  ✓ Grafo relacional interativo salvo → {salvar_html}")
    return salvar_html