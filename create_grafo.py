"""
create_grafo.py
Constrói dois tipos de grafo a partir de TCCs:

  1. Grafo de Co-ocorrência NER  (original, mantido)
     nó = entidade nomeada | aresta = janela de co-ocorrência

  2. Grafo Relacional Acadêmico  ← NOVO
     nó = Universidade / Departamento / Orientador / Autor / Trabalho
     aresta tipada:
       UNIV ──[tem_dept]──▶ DEPT
       DEPT ──[tem_prof]──▶ ORIENTADOR
       ORIENTADOR ──[orientou]──▶ AUTOR
       AUTOR ──[produziu]──▶ TRABALHO
       (opcionalmente: ORIENTADOR ──[tema]──▶ AREA)
"""

import time
import networkx as nx
from itertools import combinations


# ═════════════════════════════════════════════
# PARTE 1 — GRAFO DE CO-OCORRÊNCIA NER
# ═════════════════════════════════════════════

def janela_sentenca(doc):
    """Cada sentença detectada pelo spaCy é uma janela de co-ocorrência."""
    return [
        [(ent.text.strip(), ent.label_) for ent in sent.ents]
        for sent in doc.sents
        if len(list(sent.ents)) >= 2
    ]


def janela_paragrafo(texto_limpo: str, doc):
    """Cada parágrafo (\n\n) é uma janela de co-ocorrência."""
    paragrafos = texto_limpo.split("\n\n")
    janelas, offset = [], 0
    for par in paragrafos:
        inicio, fim = offset, offset + len(par)
        ents = [
            (ent.text.strip(), ent.label_)
            for ent in doc.ents
            if ent.start_char >= inicio and ent.end_char <= fim
        ]
        if len(ents) >= 2:
            janelas.append(ents)
        offset = fim + 2
    return janelas


def janela_k_caracteres(texto_limpo: str, doc, k: int = 500):
    """Janelas deslizantes de k caracteres (sem sobreposição)."""
    janelas, total = [], len(texto_limpo)
    for inicio in range(0, total, k):
        fim = min(inicio + k, total)
        ents = [
            (ent.text.strip(), ent.label_)
            for ent in doc.ents
            if ent.start_char >= inicio and ent.end_char <= fim
        ]
        if len(ents) >= 2:
            janelas.append(ents)
    return janelas


def criar_grafo(lista_entidades: list, estrategia: str = "") -> nx.Graph:
    """
    Constrói grafo ponderado de co-ocorrência NER.
    nó   = entidade (texto normalizado)
    aresta = co-ocorrência dentro da mesma janela
    peso = frequência de co-ocorrência
    """
    G = nx.Graph()
    G.graph["estrategia"] = estrategia
    G.graph["tipo"] = "coocorrencia"

    for janela in lista_entidades:
        ents_unicas = list({
            (nome.lower().strip(), tipo)
            for nome, tipo in janela
            if nome.strip()
        })
        for nome, tipo in ents_unicas:
            if not G.has_node(nome):
                G.add_node(nome, tipo=tipo, count=0)
            G.nodes[nome]["count"] += 1

        for (n1, _), (n2, _) in combinations(ents_unicas, 2):
            if G.has_edge(n1, n2):
                G[n1][n2]["weight"] += 1
            else:
                G.add_edge(n1, n2, weight=1)

    return G


# ═════════════════════════════════════════════
# PARTE 2 — GRAFO RELACIONAL ACADÊMICO  ← NOVO
# ═════════════════════════════════════════════

def _id_no(tipo: str, nome: str) -> str:
    """Gera ID único de nó: 'UNIV::UFRN', 'AUTOR::João Silva', etc."""
    return f"{tipo}::{nome.strip().lower()}"


def adicionar_trabalho_ao_grafo(
    G_rel: nx.DiGraph,
    meta: dict,
    doc=None,
) -> nx.DiGraph:
    """
    Recebe o dicionário de metadados de UM arquivo e insere/atualiza
    os nós e arestas no grafo relacional dirigido G_rel.

    Hierarquia inserida:
      UNIV ──[tem_dept]──▶ DEPT  (se dept disponível)
      UNIV ──[tem_prof]──▶ ORIENTADOR  (fallback se não há dept)
      DEPT ──[tem_prof]──▶ ORIENTADOR
      ORIENTADOR ──[orientou]──▶ AUTOR
      AUTOR ──[produziu]──▶ TRABALHO
      ORIENTADOR ──[tema]──▶ AREA  (se doc fornecido, para áreas detectadas)

    Cada nó recebe atributo 'tipo' e 'label' (nome legível).
    Cada aresta recebe 'relacao' (nome da relação) e 'peso' (acumulado).
    """

    univ      = meta.get("sigla_univ") or meta.get("universidade")
    dept      = meta.get("departamento")
    orientador = meta.get("orientador")
    coorientador = meta.get("coorientador")
    autor     = meta.get("autor")
    titulo    = meta.get("titulo") or meta.get("arquivo", "trabalho_sem_titulo")
    ano       = meta.get("ano") or ""

    def _add_no(nid: str, tipo: str, label: str, **kwargs):
        if not G_rel.has_node(nid):
            G_rel.add_node(nid, tipo=tipo, label=label, count=0, **kwargs)
        G_rel.nodes[nid]["count"] += 1

    def _add_aresta(u: str, v: str, relacao: str):
        if G_rel.has_edge(u, v):
            G_rel[u][v]["peso"] += 1
        else:
            G_rel.add_edge(u, v, relacao=relacao, peso=1)

    # — Universidade —
    if univ:
        id_univ = _id_no("UNIV", univ)
        _add_no(id_univ, "UNIV", univ)
    else:
        id_univ = None

    # — Departamento —
    if dept:
        id_dept = _id_no("DEPT", dept)
        _add_no(id_dept, "DEPT", dept)
        if id_univ:
            _add_aresta(id_univ, id_dept, "tem_dept")
    else:
        id_dept = None

    # — Orientador —
    id_orient = None
    if orientador:
        id_orient = _id_no("ORIENTADOR", orientador)
        _add_no(id_orient, "ORIENTADOR", orientador)
        if id_dept:
            _add_aresta(id_dept, id_orient, "tem_prof")
        elif id_univ:
            _add_aresta(id_univ, id_orient, "tem_prof")

    # — Coorientador (tratado como orientador adicional) —
    id_coorient = None
    if coorientador:
        id_coorient = _id_no("ORIENTADOR", coorientador)
        _add_no(id_coorient, "ORIENTADOR", coorientador)
        if id_dept:
            _add_aresta(id_dept, id_coorient, "tem_prof")
        elif id_univ:
            _add_aresta(id_univ, id_coorient, "tem_prof")

    # — Autor —
    id_autor = None
    if autor:
        id_autor = _id_no("AUTOR", autor)
        _add_no(id_autor, "AUTOR", autor)
        if id_orient:
            _add_aresta(id_orient, id_autor, "orientou")
        if id_coorient:
            _add_aresta(id_coorient, id_autor, "coorientou")

    # — Trabalho —
    id_trab = _id_no("TRABALHO", titulo)
    _add_no(id_trab, "TRABALHO", titulo, ano=ano)
    if id_autor:
        _add_aresta(id_autor, id_trab, "produziu")

    # — Áreas temáticas (extraídas do NER do doc) ─────────────────────────────
    if doc is not None and id_orient:
        areas_vistas = set()
        for ent in doc.ents:
            if ent.label_ == "AREA" and ent.text.strip().lower() not in areas_vistas:
                areas_vistas.add(ent.text.strip().lower())
                id_area = _id_no("AREA", ent.text.strip())
                _add_no(id_area, "AREA", ent.text.strip())
                _add_aresta(id_orient, id_area, "tema")
                if id_trab:
                    _add_aresta(id_trab, id_area, "aborda")

    return G_rel


def criar_grafo_relacional(lista_metadados: list, lista_docs=None) -> nx.DiGraph:
    """
    Constrói o grafo relacional a partir de uma lista de dicionários
    de metadados (um por arquivo).

    lista_docs: lista de docs spaCy na mesma ordem (opcional, para áreas).
    """
    G_rel = nx.DiGraph()
    G_rel.graph["tipo"] = "relacional_academico"

    for i, meta in enumerate(lista_metadados):
        doc = lista_docs[i] if lista_docs and i < len(lista_docs) else None
        adicionar_trabalho_ao_grafo(G_rel, meta, doc=doc)

    n_nos     = G_rel.number_of_nodes()
    n_arestas = G_rel.number_of_edges()
    print(f"\n  Grafo relacional construído:")
    print(f"    Nós    : {n_nos}")
    print(f"    Arestas: {n_arestas}")

    # Resumo por tipo de nó
    tipos = {}
    for _, d in G_rel.nodes(data=True):
        t = d.get("tipo", "?")
        tipos[t] = tipos.get(t, 0) + 1
    for t, q in sorted(tipos.items(), key=lambda x: x[1], reverse=True):
        print(f"      {t:<12} → {q} nó(s)")

    return G_rel


# ═════════════════════════════════════════════
# ANÁLISE DO GRAFO DE CO-OCORRÊNCIA
# ═════════════════════════════════════════════

def analisar_grafo(G: nx.Graph, top_n: int = 10) -> dict:
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

    densidade        = nx.density(G)
    componentes      = list(nx.connected_components(G))
    maior            = max(componentes, key=len)
    clustering_medio = nx.average_clustering(G, weight="weight")

    print(f"  Densidade             : {densidade:.4f}")
    print(f"  Componentes conectados: {len(componentes)}")
    print(f"  Maior componente      : {len(maior)} nós")
    print(f"  Clustering médio      : {clustering_medio:.4f}")

    subgrafo = G.subgraph(maior)
    try:
        diametro = nx.diameter(subgrafo)
        print(f"  Diâmetro              : {diametro}")
    except Exception:
        diametro = None
        print(f"  Diâmetro              : N/A")

    centralidade_grau  = nx.degree_centrality(G)
    centralidade_entre = nx.betweenness_centrality(G, weight="weight")

    print(f"\n  Top {top_n} por grau:")
    top_grau = sorted(centralidade_grau.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for no, val in top_grau:
        tipo = G.nodes[no].get("tipo", "?")
        freq = G.nodes[no].get("count", 0)
        print(f"    {no:<35} [{tipo}]  grau={val:.4f}  freq={freq}")

    print(f"\n  Top {top_n} por betweenness:")
    top_entre = sorted(centralidade_entre.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for no, val in top_entre:
        tipo = G.nodes[no].get("tipo", "?")
        print(f"    {no:<35} [{tipo}]  between={val:.4f}")

    print(f"\n  Distribuição por tipo NER:")
    tipos = {}
    for _, dados in G.nodes(data=True):
        t = dados.get("tipo", "?")
        tipos[t] = tipos.get(t, 0) + 1
    for tipo, qtd in sorted(tipos.items(), key=lambda x: x[1], reverse=True):
        print(f"    {tipo:<12} → {qtd} nós")

    return {
        "estrategia": estrategia, "nos": n_nos, "arestas": n_arestas,
        "densidade": densidade, "componentes": len(componentes),
        "maior_componente": len(maior), "clustering_medio": clustering_medio,
        "diametro": diametro, "top_grau": top_grau, "top_betweenness": top_entre,
        "distribuicao_tipos": tipos,
    }


# ═════════════════════════════════════════════
# ANÁLISE DO GRAFO RELACIONAL
# ═════════════════════════════════════════════

def analisar_grafo_relacional(G_rel: nx.DiGraph, top_n: int = 10) -> dict:
    """
    Análise específica para o grafo relacional acadêmico.
    Destaca orientadores mais produtivos, universidades mais presentes, etc.
    """
    print(f"\n{'═'*50}")
    print(f"  ANÁLISE — GRAFO RELACIONAL ACADÊMICO")
    print(f"{'═'*50}")

    n_nos     = G_rel.number_of_nodes()
    n_arestas = G_rel.number_of_edges()
    print(f"  Nós    : {n_nos}  |  Arestas: {n_arestas}")

    if n_nos == 0:
        return {}

    # — orientadores mais produtivos (out-degree em "orientou") —
    orientadores = {
        n: d for n, d in G_rel.out_degree()
        if G_rel.nodes[n].get("tipo") == "ORIENTADOR"
    }
    print(f"\n  Top {top_n} orientadores (por alunos orientados):")
    for no, grau in sorted(orientadores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        label = G_rel.nodes[no].get("label", no)
        print(f"    {label:<35}  out-degree={grau}")

    # — universidades mais presentes —
    universidades = {
        n: G_rel.nodes[n].get("count", 1)
        for n in G_rel.nodes()
        if G_rel.nodes[n].get("tipo") == "UNIV"
    }
    print(f"\n  Universidades presentes:")
    for no, cnt in sorted(universidades.items(), key=lambda x: x[1], reverse=True):
        label = G_rel.nodes[no].get("label", no)
        print(f"    {label:<40}  freq={cnt}")

    # — áreas temáticas mais recorrentes —
    areas = {
        n: G_rel.nodes[n].get("count", 1)
        for n in G_rel.nodes()
        if G_rel.nodes[n].get("tipo") == "AREA"
    }
    print(f"\n  Top {top_n} áreas temáticas:")
    for no, cnt in sorted(areas.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        label = G_rel.nodes[no].get("label", no)
        print(f"    {label:<35}  freq={cnt}")

    return {
        "nos": n_nos, "arestas": n_arestas,
        "orientadores": orientadores,
        "universidades": universidades,
        "areas": areas,
    }


# ═════════════════════════════════════════════
# COMPARAÇÃO DE ESTRATÉGIAS (co-ocorrência)
# ═════════════════════════════════════════════

def comparar_estrategias(texto_limpo: str, doc, k: int = 500) -> dict:
    estrategias = {
        "sentenca":   lambda: janela_sentenca(doc),
        "paragrafo":  lambda: janela_paragrafo(texto_limpo, doc),
        f"k{k}chars": lambda: janela_k_caracteres(texto_limpo, doc, k=k),
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
        metricas = analisar_grafo(G)
        metricas["tempo"] = round(t1 - t0, 4)
        print(f"  ⏱ Tempo de construção: {metricas['tempo']}s")
        resultados[nome] = {"grafo": G, "metricas": metricas}

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


# ═════════════════════════════════════════════
# BUSCA INTERATIVA (EGO GRAPH)
# ═════════════════════════════════════════════

def buscar_termo(G: nx.Graph, visualizar_ego_fn=None):
    nos_disponiveis = sorted(G.nodes())
    print(f"\n  Grafo com {len(nos_disponiveis)} entidades disponíveis.")
    print("  Digite parte do nome para buscar, ou 'sair' para encerrar.\n")

    while True:
        termo = input("  Termo: ").strip().lower()
        if termo == "sair":
            break

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

        dados = G.nodes[termo]
        vizinhos = sorted(
            G[termo].items(), key=lambda x: x[1].get("weight", 0), reverse=True
        )
        print(f"\n  [{dados.get('tipo','?')}] {termo}  (freq={dados.get('count',0)})")
        print(f"  Vizinhos ({len(vizinhos)}):")
        for viz, attrs in vizinhos[:10]:
            print(f"    → {viz:<35} peso={attrs.get('weight',1)}")

        if visualizar_ego_fn:
            visualizar_ego_fn(G, termo, raio=2)