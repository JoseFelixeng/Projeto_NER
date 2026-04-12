import time
import networkx as nx
from itertools import combinations

# ─────────────────────────────────────────────
# ESTRATÉGIAS DE JANELA (sentença / parágrafo / k-caracteres)
# ─────────────────────────────────────────────

def janela_sentenca(doc):
    """
    Cada sentença detectada pelo spaCy é uma janela de co-ocorrência.
    Retorna lista de listas de (texto, label).
    """
    return [
        [(ent.text.strip(), ent.label_) for ent in sent.ents]
        for sent in doc.sents
        if len(list(sent.ents)) >= 2   # só janelas com ao menos 2 entidades
    ]


def janela_paragrafo(texto_limpo, doc):
    """
    Cada parágrafo (separado por \n\n) é uma janela de co-ocorrência.
    Reanota as entidades usando o doc completo, filtrando pelo char offset.
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
        if len(ents) >= 2:
            janelas.append(ents)
        offset = fim + 2   # +2 pelo \n\n

    return janelas


def janela_k_caracteres(texto_limpo, doc, k: int = 500):
    """
    Janelas deslizantes de k caracteres (sem sobreposição).
    Útil para capturar co-ocorrências além do limite de sentença/parágrafo.
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
        if len(ents) >= 2:
            janelas.append(ents)

    return janelas


# ─────────────────────────────────────────────
# CRIAÇÃO DO GRAFO
# ─────────────────────────────────────────────

def criar_grafo(lista_entidades: list, estrategia: str = "") -> nx.Graph:
    """
    Recebe lista de janelas:
        [ [("UFRN","ORG"), ("Natal","LOC")], ... ]

    Constrói grafo ponderado onde:
        - nó   = entidade (texto normalizado)
        - aresta = co-ocorrência dentro da mesma janela
        - peso = frequência de co-ocorrência

    Atributos do nó: tipo (label NER), count (freq. total), estrategia
    """
    G = nx.Graph()
    G.graph["estrategia"] = estrategia

    for janela in lista_entidades:
        # normaliza e deduplica dentro da janela
        ents_unicas = list({
            (nome.lower().strip(), tipo)
            for nome, tipo in janela
            if nome.strip()
        })

        # adiciona / atualiza nós
        for nome, tipo in ents_unicas:
            if not G.has_node(nome):
                G.add_node(nome, tipo=tipo, count=0)
            G.nodes[nome]["count"] += 1

        # cria arestas (combinações par a par)
        for (n1, _), (n2, _) in combinations(ents_unicas, 2):
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
    Calcula e exibe métricas estruturais do grafo.
    Retorna dicionário com os resultados para comparação entre estratégias.
    """
    estrategia = G.graph.get("estrategia", "?")
    print(f"\n{'═'*50}")
    print(f"  ESTRATÉGIA: {estrategia.upper()}")
    print(f"{'═'*50}")

    n_nos     = G.number_of_nodes()
    n_arestas = G.number_of_edges()
    print(f"  Nós      : {n_nos}")
    print(f"  Arestas  : {n_arestas}")

    if n_nos == 0:
        print("  [Grafo vazio]")
        return {}

    # densidade
    densidade = nx.density(G)
    print(f"  Densidade: {densidade:.4f}")

    # componentes conectados
    componentes = list(nx.connected_components(G))
    print(f"  Componentes conectados: {len(componentes)}")
    maior = max(componentes, key=len)
    print(f"  Maior componente      : {len(maior)} nós")

    # clustering
    clustering_medio = nx.average_clustering(G, weight="weight")
    print(f"  Clustering médio      : {clustering_medio:.4f}")

    # diâmetro (apenas no maior componente para evitar erro)
    subgrafo = G.subgraph(maior)
    try:
        diametro = nx.diameter(subgrafo)
        print(f"  Diâmetro              : {diametro}")
    except Exception:
        diametro = None
        print(f"  Diâmetro              : N/A")

    # centralidades
    centralidade_grau       = nx.degree_centrality(G)
    centralidade_entre      = nx.betweenness_centrality(G, weight="weight")
    centralidade_prox       = nx.closeness_centrality(G)

    print(f"\n  Top {top_n} por grau (nós mais conectados):")
    top_grau = sorted(centralidade_grau.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for no, val in top_grau:
        tipo = G.nodes[no].get("tipo", "?")
        freq = G.nodes[no].get("count", 0)
        print(f"    {no:<30} [{tipo}]  grau={val:.4f}  freq={freq}")

    print(f"\n  Top {top_n} por betweenness (pontes entre comunidades):")
    top_entre = sorted(centralidade_entre.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for no, val in top_entre:
        tipo = G.nodes[no].get("tipo", "?")
        print(f"    {no:<30} [{tipo}]  between={val:.4f}")

    # distribuição por tipo de entidade
    print(f"\n  Distribuição por tipo NER:")
    tipos = {}
    for _, dados in G.nodes(data=True):
        t = dados.get("tipo", "?")
        tipos[t] = tipos.get(t, 0) + 1
    for tipo, qtd in sorted(tipos.items(), key=lambda x: x[1], reverse=True):
        print(f"    {tipo:<10} → {qtd} nós")

    return {
        "estrategia"         : estrategia,
        "nos"                : n_nos,
        "arestas"            : n_arestas,
        "densidade"          : densidade,
        "componentes"        : len(componentes),
        "maior_componente"   : len(maior),
        "clustering_medio"   : clustering_medio,
        "diametro"           : diametro,
        "top_grau"           : top_grau,
        "top_betweenness"    : top_entre,
        "distribuicao_tipos" : tipos,
    }


# ─────────────────────────────────────────────
# COMPARAÇÃO DE DESEMPENHO ENTRE ESTRATÉGIAS
# ─────────────────────────────────────────────

def comparar_estrategias(texto_limpo: str, doc, k: int = 500) -> dict:
    """
    Executa as 3 estratégias, mede tempo, constrói grafos e compara métricas.
    Retorna dicionário com os 3 grafos e suas métricas.
    """
    estrategias = {
        "sentenca"     : lambda: janela_sentenca(doc),
        "paragrafo"    : lambda: janela_paragrafo(texto_limpo, doc),
        f"k{k}chars"   : lambda: janela_k_caracteres(texto_limpo, doc, k=k),
    }

    resultados = {}

    print("\n" + "═"*50)
    print("  COMPARAÇÃO DE ESTRATÉGIAS DE JANELA")
    print("═"*50)

    for nome, fn_janela in estrategias.items():
        t0      = time.perf_counter()
        janelas = fn_janela()
        G       = criar_grafo(janelas, estrategia=nome)
        t1      = time.perf_counter()

        metricas          = analisar_grafo(G)
        metricas["tempo"] = round(t1 - t0, 4)

        print(f"  ⏱ Tempo de construção: {metricas['tempo']}s")
        resultados[nome] = {"grafo": G, "metricas": metricas}

    # resumo comparativo
    print("\n" + "═"*50)
    print("  RESUMO COMPARATIVO")
    print(f"  {'Estratégia':<15} {'Nós':>6} {'Arestas':>8} "
          f"{'Densidade':>10} {'Clustering':>11} {'Tempo(s)':>9}")
    print("  " + "-"*65)
    for nome, r in resultados.items():
        m = r["metricas"]
        print(
            f"  {nome:<15} {m.get('nos',0):>6} {m.get('arestas',0):>8} "
            f"{m.get('densidade',0):>10.4f} {m.get('clustering_medio',0):>11.4f} "
            f"{m.get('tempo',0):>9.4f}"
        )

    return resultados


# ─────────────────────────────────────────────
# BUSCA INTERATIVA (EGO GRAPH)
# ─────────────────────────────────────────────

def buscar_termo(G: nx.Graph, visualizar_ego_fn=None):
    """
    Loop de busca interativa por termo no grafo.
    Se visualizar_ego_fn for passada, chama a visualização.
    """
    nos_disponiveis = sorted(G.nodes())
    print(f"\n  Grafo com {len(nos_disponiveis)} entidades disponíveis.")
    print("  Digite parte do nome para buscar, ou 'sair' para encerrar.\n")

    while True:
        termo = input("  Termo: ").strip().lower()

        if termo == "sair":
            break

        # busca parcial — útil quando não se lembra do nome exato
        matches = [n for n in nos_disponiveis if termo in n]

        if not matches:
            print(f"  Nenhuma entidade contém '{termo}'.")
            continue

        if len(matches) > 1:
            print(f"  Encontrados {len(matches)}: {', '.join(matches[:10])}")
            termo = input("  Digite o nome exato: ").strip().lower()
            if termo not in G:
                print("  Não encontrado.")
                continue

        else:
            termo = matches[0]

        # informações do nó
        dados = G.nodes[termo]
        vizinhos = sorted(
            G[termo].items(), key=lambda x: x[1].get("weight", 0), reverse=True
        )
        print(f"\n  [{dados.get('tipo','?')}] {termo}  (freq={dados.get('count',0)})")
        print(f"  Vizinhos ({len(vizinhos)}):")
        for viz, attrs in vizinhos[:10]:
            print(f"    → {viz:<30} peso={attrs.get('weight',1)}")

        if visualizar_ego_fn:
            visualizar_ego_fn(G, termo, raio=2)