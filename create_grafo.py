"""
create_grafo.py
Criação, análise e busca interativa do grafo de co-ocorrência NER.
"""

import time
import networkx as nx
from itertools import combinations
from regex import filtrar_entidade

# ─────────────────────────────────────────────
# UTILITÁRIO INTERNO
# ─────────────────────────────────────────────

def _limpar_janela(ents: list) -> list:
    """
    Aplica filtrar_entidade() e normaliza texto.
    - descarta entidades com > 3 tokens (reduzido de 5 para cortar mais ruído)
    - normaliza para lowercase
    - deduplica dentro da janela
    """
    vistos = set()
    resultado = []
    for nome, tipo in ents:
        nome_norm = nome.lower().strip()
        # filtra vazios, muito curtos e duplicados
        if not nome_norm or len(nome_norm) <= 2:
            continue
        if not filtrar_entidade(nome_norm, tipo):
            continue
        if nome_norm not in vistos:
            vistos.add(nome_norm)
            resultado.append((nome_norm, tipo))
    return resultado


# ─────────────────────────────────────────────
# ESTRATÉGIAS DE JANELA
# ─────────────────────────────────────────────

def janela_sentenca(doc) -> list:
    """
    Sentença como janela de co-ocorrência.
    Mais preciso semanticamente — captura relações diretas.
    """
    janelas = []
    for sent in doc.sents:
        ents = [(ent.text.strip(), ent.label_) for ent in sent.ents]
        ents = _limpar_janela(ents)
        if len(ents) >= 2:
            janelas.append(ents)
    return janelas


def janela_paragrafo(texto_limpo: str, doc) -> list:
    """
    Parágrafo como janela de co-ocorrência.
    Captura mais contexto mas gera mais ruído em parágrafos longos.
    """
    paragrafos = texto_limpo.split("\n\n")
    janelas = []
    offset = 0

    for par in paragrafos:
        inicio = offset
        fim    = offset + len(par)
        ents   = [
            (ent.text.strip(), ent.label_)
            for ent in doc.ents
            if ent.start_char >= inicio and ent.end_char <= fim
        ]
        ents = _limpar_janela(ents)
        if len(ents) >= 2:
            janelas.append(ents)
        offset = fim + 2

    return janelas


def janela_k_caracteres(texto_limpo: str, doc, k: int = 500) -> list:
    """
    Janela deslizante de k caracteres.
    Equilíbrio entre precisão (sentença) e contexto (parágrafo).
    """
    janelas = []
    total   = len(texto_limpo)

    for inicio in range(0, total, k):
        fim  = min(inicio + k, total)
        ents = [
            (ent.text.strip(), ent.label_)
            for ent in doc.ents
            if ent.start_char >= inicio and ent.end_char <= fim
        ]
        ents = _limpar_janela(ents)
        if len(ents) >= 2:
            janelas.append(ents)

    return janelas


# ─────────────────────────────────────────────
# CRIAÇÃO DO GRAFO
# ─────────────────────────────────────────────

def criar_grafo(lista_entidades: list, estrategia: str = "") -> nx.Graph:
    """
    Constrói grafo ponderado de co-ocorrência NER.
      nó    = entidade normalizada
      aresta = co-ocorrência dentro da mesma janela
      peso  = frequência de co-ocorrência
    """
    G = nx.Graph()
    G.graph["estrategia"] = estrategia

    for janela in lista_entidades:
        for nome, tipo in janela:
            if not G.has_node(nome):
                G.add_node(nome, tipo=tipo, count=0)
            G.nodes[nome]["count"] += 1

        for (n1, _), (n2, _) in combinations(janela, 2):
            if G.has_edge(n1, n2):
                G[n1][n2]["weight"] += 1
            else:
                G.add_edge(n1, n2, weight=1)

    return G


# ─────────────────────────────────────────────
# ANÁLISE DO GRAFO
# ─────────────────────────────────────────────

def analisar_grafo(G: nx.Graph, top_n: int = 10) -> dict:
    """
    Calcula métricas estruturais e retorna dict para comparação.
    Imprime análise crítica por estratégia.
    """
    estrategia = G.graph.get("estrategia", "?")
    print(f"\n{'═'*55}")
    print(f"  ESTRATÉGIA: {estrategia.upper()}")
    print(f"{'═'*55}")

    n_nos     = G.number_of_nodes()
    n_arestas = G.number_of_edges()
    print(f"  Nós      : {n_nos}")
    print(f"  Arestas  : {n_arestas}")

    if n_nos == 0:
        print("  [Grafo vazio — nenhuma entidade passou pelos filtros]")
        return {}

    densidade = nx.density(G)
    print(f"  Densidade: {densidade:.4f}", end="")
    # interpretação automática da densidade
    if densidade > 0.15:
        print("  ⚠ ALTA — janela provavelmente grande demais")
    elif densidade < 0.005:
        print("  ⚠ BAIXA — janela provavelmente pequena demais / grafo fragmentado")
    else:
        print("  ✓ adequada")

    componentes = list(nx.connected_components(G))
    maior = max(componentes, key=len)
    print(f"  Componentes conectados : {len(componentes)}")
    print(f"  Maior componente       : {len(maior)} nós "
          f"({100*len(maior)/n_nos:.1f}% do total)")

    clustering_medio = nx.average_clustering(G, weight="weight")
    print(f"  Clustering médio       : {clustering_medio:.4f}")

    subgrafo = G.subgraph(maior)
    try:
        diametro = nx.diameter(subgrafo)
        print(f"  Diâmetro               : {diametro}")
    except Exception:
        diametro = None
        print(f"  Diâmetro               : N/A")

    # centralidades
    centralidade_grau  = nx.degree_centrality(G)
    centralidade_entre = nx.betweenness_centrality(G, weight="weight")
    centralidade_prox  = nx.closeness_centrality(G)

    print(f"\n  Top {top_n} por grau (hubs — mais conectados):")
    top_grau = sorted(centralidade_grau.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for no, val in top_grau:
        tipo = G.nodes[no].get("tipo", "?")
        freq = G.nodes[no].get("count", 0)
        print(f"    {no:<35} [{tipo}]  grau={val:.4f}  freq={freq}")

    print(f"\n  Top {top_n} por betweenness (pontes entre comunidades):")
    top_entre = sorted(centralidade_entre.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for no, val in top_entre:
        tipo = G.nodes[no].get("tipo", "?")
        print(f"    {no:<35} [{tipo}]  between={val:.4f}")

    print(f"\n  Top {top_n} por closeness (acesso rápido à rede):")
    top_prox = sorted(centralidade_prox.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for no, val in top_prox:
        tipo = G.nodes[no].get("tipo", "?")
        print(f"    {no:<35} [{tipo}]  close={val:.4f}")

    print(f"\n  Distribuição por tipo NER:")
    tipos = {}
    for _, dados in G.nodes(data=True):
        t = dados.get("tipo", "?")
        tipos[t] = tipos.get(t, 0) + 1
    for tipo, qtd in sorted(tipos.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * qtd / n_nos
        print(f"    {tipo:<10} → {qtd:4d} nós  ({pct:.1f}%)")

    # arestas mais pesadas (relações mais fortes)
    print(f"\n  Top 10 arestas mais frequentes (relações mais fortes):")
    top_arestas = sorted(
        G.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True
    )[:10]
    for u, v, d in top_arestas:
        print(f"    {u:<25} ↔ {v:<25}  peso={d.get('weight',1)}")

    return {
        "estrategia"        : estrategia,
        "nos"               : n_nos,
        "arestas"           : n_arestas,
        "densidade"         : densidade,
        "componentes"       : len(componentes),
        "maior_componente"  : len(maior),
        "clustering_medio"  : clustering_medio,
        "diametro"          : diametro,
        "top_grau"          : top_grau,
        "top_betweenness"   : top_entre,
        "top_closeness"     : top_prox,
        "distribuicao_tipos": tipos,
        "top_arestas"       : [(u, v, d["weight"]) for u, v, d in top_arestas],
    }


# ─────────────────────────────────────────────
# RESUMO COMPARATIVO
# ─────────────────────────────────────────────

def resumo_comparativo(resultados: dict) -> None:
    """
    Imprime tabela comparativa e conclusão automática entre estratégias.
    """
    print("\n" + "═"*70)
    print("  RESUMO COMPARATIVO DAS ESTRATÉGIAS")
    print(f"  {'Estratégia':<15} {'Nós':>6} {'Arestas':>8} "
          f"{'Densidade':>10} {'Clustering':>11} {'Componentes':>13}")
    print("  " + "-"*68)

    melhor = None
    melhor_score = -1

    for nome, r in resultados.items():
        m = r.get("metricas", {})
        nos       = m.get("nos", 0)
        arestas   = m.get("arestas", 0)
        dens      = m.get("densidade", 0)
        clust     = m.get("clustering_medio", 0)
        comp      = m.get("componentes", 0)

        # score heurístico: penaliza densidade extrema e fragmentação
        if nos > 0:
            score = clust - abs(dens - 0.05) * 5 - (comp / nos)
            if score > melhor_score:
                melhor_score = score
                melhor = nome

        flag = "  ←" if nome == melhor else ""
        print(f"  {nome:<15} {nos:>6} {arestas:>8} "
              f"{dens:>10.4f} {clust:>11.4f} {comp:>13}{flag}")

    print("\n  CONCLUSÃO AUTOMÁTICA:")
    print(f"  Estratégia com melhor equilíbrio densidade/clustering: "
          f"'{melhor}'")
    print("  (score = clustering - penalidade_densidade - fragmentação)")
    print("═"*70)


# ─────────────────────────────────────────────
# BUSCA INTERATIVA
# ─────────────────────────────────────────────

def buscar_termo(G: nx.Graph, visualizar_ego_fn=None) -> None:
    """
    Busca interativa por termo com informações detalhadas do nó.
    Suporta busca parcial e exibe vizinhos ordenados por peso.
    """
    nos_disponiveis = sorted(G.nodes())
    print(f"\n  Grafo com {len(nos_disponiveis)} entidades.")
    print("  Digite parte do nome para buscar, ou 'sair' para encerrar.\n")

    while True:
        termo = input("  Termo: ").strip().lower()
        if termo == "sair":
            break

        matches = [n for n in nos_disponiveis if termo in n]

        if not matches:
            print(f"  ✗ Nenhuma entidade contém '{termo}'.")
            continue

        if len(matches) > 1:
            print(f"  Encontrados {len(matches)}:")
            for i, m in enumerate(matches[:15]):
                print(f"    [{i}] {m}")
            escolha = input("  Digite o nome exato (ou número): ").strip()
            if escolha.isdigit() and int(escolha) < len(matches):
                termo = matches[int(escolha)]
            elif escolha in G:
                termo = escolha
            else:
                print("  Não encontrado.")
                continue
        else:
            termo = matches[0]

        dados    = G.nodes[termo]
        vizinhos = sorted(
            G[termo].items(), key=lambda x: x[1].get("weight", 0), reverse=True
        )

        print(f"\n  ┌─ [{dados.get('tipo','?')}] {termo}")
        print(f"  │  Frequência : {dados.get('count', 0)}")
        print(f"  │  Grau       : {G.degree(termo)}")
        print(f"  └─ Top vizinhos ({min(10, len(vizinhos))}/{len(vizinhos)}):")
        for viz, attrs in vizinhos[:10]:
            tipo_viz = G.nodes[viz].get("tipo", "?")
            print(f"       → [{tipo_viz}] {viz:<35}  peso={attrs.get('weight',1)}")

        if visualizar_ego_fn:
            visualizar_ego_fn(G, termo, raio=2)