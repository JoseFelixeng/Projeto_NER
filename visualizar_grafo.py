"""
visualizar_grafo.py
Visualização interativa profissional do grafo de entidades.
Uso: importar e chamar visualizar_grafo_interativo(G) no main.py
"""

import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio
from collections import defaultdict
import math


# ── Paleta de cores ────────────────────────────────────────────────────────────
COR_FUNDO        = "#0d0f14"
COR_PAINEL       = "#13161e"
COR_ARESTA       = "#2a2f3d"
COR_TEXTO        = "#e8eaf0"
COR_TEXTO_SUAVE  = "#6b7280"
COR_DESTAQUE     = "#f59e0b"   # âmbar — nó central / hover

ESCALA_NOS = [
    [0.00, "#1e3a5f"],
    [0.25, "#1d4ed8"],
    [0.50, "#7c3aed"],
    [0.75, "#c026d3"],
    [1.00, "#f59e0b"],
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _calcular_metricas(G):
    """Retorna dicts de centralidade, grau e comunidades."""
    centralidade = nx.degree_centrality(G)
    grau         = dict(G.degree())

    # Comunidades via greedy modularity
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comunidades_raw = greedy_modularity_communities(G)
        mapa_comunidade = {}
        for idx, comunidade in enumerate(comunidades_raw):
            for no in comunidade:
                mapa_comunidade[no] = idx
    except Exception:
        mapa_comunidade = {no: 0 for no in G.nodes()}

    return centralidade, grau, mapa_comunidade


def _layout(G, seed=42):
    """Spring layout com parâmetros otimizados para grafos densos."""
    n = G.number_of_nodes()
    k = 1.5 / math.sqrt(n) if n > 0 else 1.0
    return nx.spring_layout(G, k=k, iterations=80, seed=seed)


def _normalizar(valores):
    """Normaliza um dict de valores para [0, 1]."""
    mn, mx = min(valores.values()), max(valores.values())
    if mx == mn:
        return {k: 0.5 for k in valores}
    return {k: (v - mn) / (mx - mn) for k, v in valores.items()}


# ── Construção das traces ──────────────────────────────────────────────────────
def _trace_arestas(G, pos, centralidade_norm):
    """Gera uma única trace de arestas com opacidade proporcional ao peso."""
    xs, ys, opacidades = [], [], []

    pesos = {(u, v): d.get("weight", 1) for u, v, d in G.edges(data=True)}
    p_max = max(pesos.values()) if pesos else 1

    edge_traces = []
    for (u, v), peso in pesos.items():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        opacidade = 0.05 + 0.35 * (peso / p_max)
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=0.6, color=f"rgba(100,116,139,{opacidade:.2f})"),
                hoverinfo="none",
                showlegend=False,
            )
        )
    return edge_traces


def _trace_nos(G, pos, centralidade, grau, mapa_comunidade):
    """Trace principal dos nós com tamanho, cor e tooltip ricos."""
    cent_norm = _normalizar(centralidade)

    nos      = list(G.nodes())
    xs       = [pos[n][0] for n in nos]
    ys       = [pos[n][1] for n in nos]
    tamanhos = [6 + 40 * cent_norm[n] for n in nos]
    cores    = [cent_norm[n] for n in nos]

    # Tooltip com métricas
    hover = []
    for n in nos:
        vizinhos = sorted(
            G.neighbors(n),
            key=lambda v: G[n][v].get("weight", 1),
            reverse=True,
        )[:5]
        viz_str = "<br>".join(f"  • {v}" for v in vizinhos) or "—"
        hover.append(
            f"<b>{n}</b><br>"
            f"Grau: {grau[n]}<br>"
            f"Centralidade: {centralidade[n]:.4f}<br>"
            f"Comunidade: {mapa_comunidade.get(n, '?')}<br>"
            f"<br><b>Top conexões:</b><br>{viz_str}"
        )

    trace = go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        marker=dict(
            size=tamanhos,
            color=cores,
            colorscale=ESCALA_NOS,
            colorbar=dict(
                title=dict(text="Centralidade", font=dict(color=COR_TEXTO_SUAVE, size=11)),
                tickfont=dict(color=COR_TEXTO_SUAVE, size=10),
                thickness=12,
                len=0.5,
                x=1.01,
            ),
            line=dict(width=0.8, color="rgba(255,255,255,0.15)"),
            opacity=0.92,
        ),
        text=[n if cent_norm[n] > 0.12 else "" for n in nos],
        textposition="top center",
        textfont=dict(size=9, color=COR_TEXTO, family="JetBrains Mono, monospace"),
        hovertext=hover,
        hoverinfo="text",
        hoverlabel=dict(
            bgcolor=COR_PAINEL,
            bordercolor="#374151",
            font=dict(color=COR_TEXTO, size=12, family="JetBrains Mono, monospace"),
        ),
        showlegend=False,
    )
    return trace


# ── Figura final ───────────────────────────────────────────────────────────────
def _montar_figura(edge_traces, node_trace, G, centralidade):
    """Monta a figura Plotly com layout refinado."""
    top10 = sorted(centralidade.items(), key=lambda x: x[1], reverse=True)[:10]
    ranking_html = "<br>".join(
        f"<span style='color:{COR_DESTAQUE}'>{i+1:02d}.</span> {no}"
        for i, (no, _) in enumerate(top10)
    )

    fig = go.Figure(data=[*edge_traces, node_trace])

    fig.update_layout(
        title=dict(
            text="<b>Grafo de Entidades — TCCs UFRN</b>",
            font=dict(size=22, color=COR_TEXTO, family="Georgia, serif"),
            x=0.5,
            y=0.97,
        ),
        paper_bgcolor=COR_FUNDO,
        plot_bgcolor=COR_FUNDO,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=180, t=70, b=20),
        hovermode="closest",
        annotations=[
            # Painel de estatísticas
            dict(
                x=1.13, y=1.0,
                xref="paper", yref="paper",
                xanchor="left", yanchor="top",
                text=(
                    f"<b style='color:{COR_DESTAQUE}'>ESTATÍSTICAS</b><br>"
                    f"<span style='color:{COR_TEXTO_SUAVE}'>Nós</span> "
                    f"<b style='color:{COR_TEXTO}'>{G.number_of_nodes()}</b><br>"
                    f"<span style='color:{COR_TEXTO_SUAVE}'>Arestas</span> "
                    f"<b style='color:{COR_TEXTO}'>{G.number_of_edges()}</b><br>"
                    f"<span style='color:{COR_TEXTO_SUAVE}'>Arquivos</span> "
                    f"<b style='color:{COR_TEXTO}'>22</b><br>"
                    f"<br><b style='color:{COR_DESTAQUE}'>TOP 10</b><br>"
                    f"<span style='color:{COR_TEXTO};font-size:11px'>{ranking_html}</span>"
                ),
                showarrow=False,
                align="left",
                font=dict(size=12, color=COR_TEXTO, family="JetBrains Mono, monospace"),
                bgcolor=COR_PAINEL,
                bordercolor="#374151",
                borderwidth=1,
                borderpad=10,
            ),
            # Dica de uso
            dict(
                x=0.5, y=-0.02,
                xref="paper", yref="paper",
                xanchor="center", yanchor="top",
                text=(
                    "<span style='color:#4b5563;font-size:10px'>"
                    "🖱 Scroll para zoom  •  Arraste para mover  •  "
                    "Hover para detalhes  •  Duplo clique para resetar"
                    "</span>"
                ),
                showarrow=False,
                font=dict(size=10),
            ),
        ],
        dragmode="pan",
        uirevision="grafo",
    )

    # Botões de controle
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0, y=1.08,
                xanchor="left",
                buttons=[
                    dict(
                        label="⟳  Reset View",
                        method="relayout",
                        args=["uirevision", "reset"],
                    ),
                ],
                bgcolor=COR_PAINEL,
                bordercolor="#374151",
                font=dict(color=COR_TEXTO, size=11),
            )
        ]
    )

    return fig


# ── API pública ────────────────────────────────────────────────────────────────
def visualizar_grafo_interativo(G: nx.Graph, salvar_html: str = "grafo_entidades.html"):
    """
    Gera e abre uma visualização interativa do grafo no navegador.

    Parâmetros
    ----------
    G : nx.Graph
        Grafo de entidades criado pelo criar_grafo().
    salvar_html : str
        Caminho do arquivo HTML gerado (padrão: grafo_entidades.html).
    """
    print("  Calculando métricas do grafo...")
    centralidade, grau, mapa_comunidade = _calcular_metricas(G)

    print("  Calculando layout (pode levar alguns segundos)...")
    pos = _layout(G)

    print("  Construindo traces...")
    edge_traces = _trace_arestas(G, pos, _normalizar(centralidade))
    node_trace  = _trace_nos(G, pos, centralidade, grau, mapa_comunidade)

    print("  Montando figura...")
    fig = _montar_figura(edge_traces, node_trace, G, centralidade)

    print(f"  Salvando → {salvar_html}")
    pio.write_html(
        fig,
        file=salvar_html,
        auto_open=True,
        include_plotlyjs="cdn",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {
                "format": "png",
                "width": 1920,
                "height": 1080,
                "filename": "grafo_entidades_ufrn",
            },
        },
    )
    print(f"  Visualização aberta no navegador.")

def visualizar_ego_interativo(G, termo_central, raio=2, salvar_html=None):
    """
    Gera uma visualização interativa do ego-graph centrado em termo_central.
    
    Parâmetros
    ----------
    G : nx.Graph
        Grafo completo.
    termo_central : str
        Nó a partir do qual o subgrafo será extraído.
    raio : int
        Raio da vizinhança (padrão 2).
    salvar_html : str, opcional
        Nome do arquivo HTML. Se None, usa f"ego_{termo_central}.html".
    """
    if termo_central not in G:
        print(f"Termo '{termo_central}' não encontrado no grafo.")
        return

    # Extrai subgrafo
    ego = nx.ego_graph(G, termo_central, radius=raio)

    # Reaproveita as funções do visualizar_grafo_interativo
    # mas aplicadas ao subgrafo 'ego'
    from visualizar_grafo import (
        _calcular_metricas,
        _layout,
        _trace_arestas,
        _trace_nos,
        _montar_figura,
        COR_FUNDO, COR_PAINEL, COR_TEXTO, COR_DESTAQUE,
        pio
    )

    print(f"  Gerando ego-graph para '{termo_central}' com {ego.number_of_nodes()} nós e {ego.number_of_edges()} arestas.")

    # Métricas do subgrafo
    centralidade, grau, mapa_comunidade = _calcular_metricas(ego)

    # Layout (usar semente fixa para consistência)
    pos = _layout(ego, seed=42)

    # Traces
    edge_traces = _trace_arestas(ego, pos, _normalizar(centralidade))
    node_trace = _trace_nos(ego, pos, centralidade, grau, mapa_comunidade)

    # Figura (personalizada para o ego)
    fig = _montar_figura(edge_traces, node_trace, ego, centralidade)
    fig.update_layout(
        title=dict(
            text=f"<b>Ego‑Graph: {termo_central}</b>",
            font=dict(size=22, color=COR_TEXTO, family="Georgia, serif"),
            x=0.5,
            y=0.97,
        ),
        annotations=[
            # Painel de estatísticas adaptado
            dict(
                x=1.13, y=1.0,
                xref="paper", yref="paper",
                xanchor="left", yanchor="top",
                text=(
                    f"<b style='color:{COR_DESTAQUE}'>EGO‑GRAPH</b><br>"
                    f"<span style='color:#6b7280'>Centro</span> "
                    f"<b style='color:{COR_TEXTO}'>{termo_central}</b><br>"
                    f"<span style='color:#6b7280'>Nós</span> "
                    f"<b style='color:{COR_TEXTO}'>{ego.number_of_nodes()}</b><br>"
                    f"<span style='color:#6b7280'>Arestas</span> "
                    f"<b style='color:{COR_TEXTO}'>{ego.number_of_edges()}</b><br>"
                    f"<br><b style='color:{COR_DESTAQUE}'>VIZINHOS DIRETOS</b><br>"
                    f"<span style='color:{COR_TEXTO};font-size:11px'>" +
                    "<br>".join(sorted(ego.neighbors(termo_central))[:10]) +
                    "</span>"
                ),
                showarrow=False,
                align="left",
                font=dict(size=12, color=COR_TEXTO, family="JetBrains Mono, monospace"),
                bgcolor=COR_PAINEL,
                bordercolor="#374151",
                borderwidth=1,
                borderpad=10,
            ),
        ],
    )

    # Nome do arquivo
    if salvar_html is None:
        salvar_html = f"ego_{termo_central.replace(' ', '_')}.html"

    print(f"  Salvando → {salvar_html}")
    pio.write_html(
        fig,
        file=salvar_html,
        auto_open=True,
        include_plotlyjs="cdn",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {
                "format": "png",
                "width": 1920,
                "height": 1080,
                "filename": f"ego_{termo_central}",
            },
        },
    )