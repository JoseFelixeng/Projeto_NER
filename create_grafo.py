"""
create_grafo.py — v4
Dois grafos consolidados:

  1. Co-ocorrência NER (sentença / parágrafo / k-chars)
  2. Grafo Relacional Acadêmico (DiGraph hierárquico)
     UNIV → DEPT → ORIENTADOR → AUTOR → TRABALHO → AREA → FERRAMENTA

v4 — fixes:
  - _limpar_janela usa filtrar_entidade() do preprocessamento
  - _nid() produz IDs estáveis (sem lowercase nos nomes de display)
  - adicionar_trabalho() aceita palavras_chave do TCC como nós AREA
  - analisar_relacional() produz os 3 rankings pedidos:
      áreas com mais projetos, professores mais envolvidos, ferramentas
"""

import time
import networkx as nx
from itertools import combinations
from preprocessamento import filtrar_entidade, _nfd

# Labels que participam da co-ocorrência
LABELS_COOC = {
    "PER","ORG","LOC","MISC",
    "TEC","AREA","FERRAMENTA","HARDWARE","CONCEITO",
    "INST","DEPT","CURSO","ORIENTADOR",
}


# ══════════════════════════════════════════════════════════════════
# PARTE 1 — CO-OCORRÊNCIA NER
# ══════════════════════════════════════════════════════════════════

def _limpar_janela(ents: list) -> list:
    vistos = set()
    resultado = []
    for nome, label in ents:
        nome_norm = nome.lower().strip()
        if not nome_norm or len(nome_norm) <= 2:
            continue
        if label not in LABELS_COOC:
            continue
        if not filtrar_entidade(nome_norm, label):
            continue
        chave = (nome_norm, label)
        if chave not in vistos:
            vistos.add(chave)
            resultado.append(chave)
    return resultado


def janela_sentenca(doc) -> list:
    janelas = []
    for sent in doc.sents:
        ents = [(e.text.strip(), e.label_) for e in sent.ents]
        ents = _limpar_janela(ents)
        if len(ents) >= 2:
            janelas.append(ents)
    return janelas


def janela_paragrafo(texto_limpo: str, doc) -> list:
    paragrafos = texto_limpo.split("\n\n")
    janelas, offset = [], 0
    for par in paragrafos:
        inicio, fim = offset, offset + len(par)
        ents = [
            (e.text.strip(), e.label_)
            for e in doc.ents
            if e.start_char >= inicio and e.end_char <= fim
        ]
        ents = _limpar_janela(ents)
        if len(ents) >= 2:
            janelas.append(ents)
        offset = fim + 2
    return janelas


def janela_k_caracteres(texto_limpo: str, doc, k: int = 500) -> list:
    janelas, total = [], len(texto_limpo)
    for inicio in range(0, total, k):
        fim = min(inicio + k, total)
        ents = [
            (e.text.strip(), e.label_)
            for e in doc.ents
            if e.start_char >= inicio and e.end_char <= fim
        ]
        ents = _limpar_janela(ents)
        if len(ents) >= 2:
            janelas.append(ents)
    return janelas


def criar_grafo(lista_entidades: list, estrategia: str = "") -> nx.Graph:
    G = nx.Graph()
    G.graph["estrategia"] = estrategia
    G.graph["tipo"]       = "coocorrencia"
    for janela in lista_entidades:
        for nome, label in janela:
            if not G.has_node(nome):
                G.add_node(nome, tipo=label, count=0)
            G.nodes[nome]["count"] += 1
        for (n1, _), (n2, _) in combinations(janela, 2):
            if G.has_edge(n1, n2):
                G[n1][n2]["weight"] += 1
            else:
                G.add_edge(n1, n2, weight=1)
    return G


# ══════════════════════════════════════════════════════════════════
# ANÁLISE CO-OCORRÊNCIA
# ══════════════════════════════════════════════════════════════════

def analisar_grafo(G: nx.Graph, top_n: int = 10) -> dict:
    estrategia = G.graph.get("estrategia","?")
    print(f"\n{'═'*55}")
    print(f"  ESTRATÉGIA: {estrategia.upper()}")
    print(f"{'═'*55}")

    n_nos, n_arestas = G.number_of_nodes(), G.number_of_edges()
    print(f"  Nós: {n_nos}  |  Arestas: {n_arestas}")
    if n_nos == 0:
        print("  [Grafo vazio]"); return {}

    densidade = nx.density(G)
    flag = ("⚠ ALTA" if densidade > 0.15
            else "⚠ BAIXA" if densidade < 0.005 else "✓ ok")
    print(f"  Densidade: {densidade:.4f}  {flag}")

    componentes = list(nx.connected_components(G))
    maior       = max(componentes, key=len)
    clust       = nx.average_clustering(G, weight="weight")
    print(f"  Componentes: {len(componentes)}  |  Maior: {len(maior)}  "
          f"|  Clustering: {clust:.4f}")
    try:
        diam = nx.diameter(G.subgraph(maior))
    except Exception:
        diam = None
    print(f"  Diâmetro: {diam or 'N/A'}")

    deg  = nx.degree_centrality(G)
    bet  = nx.betweenness_centrality(G, weight="weight")

    print(f"\n  Top {top_n} por grau:")
    top_grau = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for no, v in top_grau:
        tipo = G.nodes[no].get("tipo","?")
        freq = G.nodes[no].get("count",0)
        print(f"    {no:<35} [{tipo}]  grau={v:.4f}  freq={freq}")

    print(f"\n  Top {top_n} por betweenness:")
    top_bet = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for no, v in top_bet:
        print(f"    {no:<35} [{G.nodes[no].get('tipo','?')}]  {v:.4f}")

    print(f"\n  Top 10 arestas mais frequentes:")
    top_edges = sorted(G.edges(data=True),
                       key=lambda x: x[2].get("weight",0), reverse=True)[:10]
    for u, v, d in top_edges:
        print(f"    {u:<25} ↔ {v:<25}  peso={d.get('weight',1)}")

    print(f"\n  Distribuição por label NER:")
    tipos: dict = {}
    for _, d in G.nodes(data=True):
        t = d.get("tipo","?")
        tipos[t] = tipos.get(t,0) + 1
    for t, q in sorted(tipos.items(), key=lambda x: x[1], reverse=True):
        print(f"    {t:<12} → {q:4d} nós  ({100*q/n_nos:.1f}%)")

    return {
        "estrategia": estrategia, "nos": n_nos, "arestas": n_arestas,
        "densidade": densidade, "componentes": len(componentes),
        "maior_componente": len(maior), "clustering_medio": clust,
        "diametro": diam, "top_grau": top_grau, "top_betweenness": top_bet,
        "distribuicao_tipos": tipos,
        "top_arestas": [(u,v,d["weight"]) for u,v,d in top_edges],
    }


def resumo_comparativo(resultados: dict) -> None:
    print("\n" + "═"*70)
    print("  RESUMO COMPARATIVO")
    print(f"  {'Estratégia':<15} {'Nós':>6} {'Arestas':>8} "
          f"{'Densidade':>10} {'Clustering':>11} {'Componentes':>13}")
    print("  " + "-"*68)
    melhor, melhor_score = None, -1
    for nome, r in resultados.items():
        m = r.get("metricas", {})
        nos  = m.get("nos",0)
        dens = m.get("densidade",0)
        clust= m.get("clustering_medio",0)
        comp = m.get("componentes",0)
        if nos > 0:
            score = clust - abs(dens - 0.05)*5 - (comp/nos)
            if score > melhor_score:
                melhor_score, melhor = score, nome
        flag = "  ←" if nome == melhor else ""
        print(f"  {nome:<15} {nos:>6} {m.get('arestas',0):>8} "
              f"{dens:>10.4f} {clust:>11.4f} {comp:>13}{flag}")
    print(f"\n  Melhor equilíbrio: '{melhor}'")
    print("═"*70)


# ══════════════════════════════════════════════════════════════════
# PARTE 2 — GRAFO RELACIONAL ACADÊMICO
# Hierarquia: UNIV → DEPT → ORIENTADOR → AUTOR → TRABALHO → AREA → FERRAMENTA
# ══════════════════════════════════════════════════════════════════

def _nid(tipo: str, nome: str) -> str:
    """ID estável: tipo::nome_nfd. Usa _nfd para evitar duplicatas por acento."""
    return f"{tipo}::{_nfd(nome)}"


def _add_no(G: nx.DiGraph, nid: str, tipo: str, label: str, **attrs):
    if not G.has_node(nid):
        G.add_node(nid, tipo=tipo, label=label, count=0, **attrs)
    G.nodes[nid]["count"] += 1


def _add_aresta(G: nx.DiGraph, u: str, v: str, relacao: str):
    if G.has_edge(u, v):
        G[u][v]["peso"] = G[u][v].get("peso", 0) + 1
    else:
        G.add_edge(u, v, relacao=relacao, peso=1)


def adicionar_trabalho(G: nx.DiGraph, meta: dict, doc=None) -> None:
    """
    Insere um TCC no grafo relacional com hierarquia completa:
      UNIV → DEPT → ORIENTADOR → AUTOR → TRABALHO → AREA → FERRAMENTA

    Se doc for fornecido, extrai AREA e FERRAMENTA do NER do texto.
    palavras_chave do meta (lista) também viram nós AREA.
    """
    univ         = meta.get("sigla_univ") or meta.get("universidade")
    dept         = meta.get("departamento")
    orientador   = meta.get("orientador")
    coorientador = meta.get("coorientador")
    autor        = meta.get("autor")
    titulo       = meta.get("titulo") or meta.get("arquivo","sem_titulo")
    ano          = meta.get("ano","")
    kws          = meta.get("palavras_chave", [])   # lista vinda do doc

    # UNIV
    id_univ = None
    if univ:
        id_univ = _nid("UNIV", univ)
        _add_no(G, id_univ, "UNIV", univ)

    # DEPT
    id_dept = None
    if dept:
        id_dept = _nid("DEPT", dept)
        _add_no(G, id_dept, "DEPT", dept)
        if id_univ: _add_aresta(G, id_univ, id_dept, "tem_dept")

    # ORIENTADOR
    id_orient = None
    if orientador:
        id_orient = _nid("ORIENTADOR", orientador)
        _add_no(G, id_orient, "ORIENTADOR", orientador)
        if id_dept:  _add_aresta(G, id_dept,  id_orient, "tem_prof")
        elif id_univ: _add_aresta(G, id_univ, id_orient, "tem_prof")

    # COORIENTADOR
    id_coori = None
    if coorientador:
        id_coori = _nid("ORIENTADOR", coorientador)
        _add_no(G, id_coori, "ORIENTADOR", coorientador)
        if id_dept:  _add_aresta(G, id_dept, id_coori, "tem_prof")
        if id_orient: _add_aresta(G, id_orient, id_coori, "co_orientou")

    # AUTOR
    id_autor = None
    if autor:
        id_autor = _nid("AUTOR", autor)
        _add_no(G, id_autor, "AUTOR", autor)
        if id_orient: _add_aresta(G, id_orient, id_autor, "orientou")
        if id_coori:  _add_aresta(G, id_coori,  id_autor, "co_orientou_aluno")

    # TRABALHO
    id_trab = _nid("TRABALHO", titulo)
    _add_no(G, id_trab, "TRABALHO", titulo, ano=ano)
    if id_autor:  _add_aresta(G, id_autor,  id_trab, "produziu")
    if id_orient: _add_aresta(G, id_orient, id_trab, "orientou_trab")

    # AREA e FERRAMENTA — do NER do texto
    if doc is not None:
        areas_vistas: set = set()
        ferr_vistas:  set = set()
        for ent in doc.ents:
            nome_nfd = _nfd(ent.text.strip())
            if ent.label_ == "AREA" and nome_nfd not in areas_vistas:
                areas_vistas.add(nome_nfd)
                id_area = _nid("AREA", ent.text.strip())
                _add_no(G, id_area, "AREA", ent.text.strip())
                _add_aresta(G, id_trab, id_area, "aborda")
                if id_orient: _add_aresta(G, id_orient, id_area, "pesquisa")
            elif ent.label_ == "FERRAMENTA" and nome_nfd not in ferr_vistas:
                ferr_vistas.add(nome_nfd)
                id_ferr = _nid("FERRAMENTA", ent.text.strip())
                _add_no(G, id_ferr, "FERRAMENTA", ent.text.strip())
                _add_aresta(G, id_trab, id_ferr, "usa")

    # AREA — das palavras-chave extraídas
    for kw in kws:
        kw_nfd = _nfd(kw)
        id_kw  = _nid("AREA", kw)
        _add_no(G, id_kw, "AREA", kw)
        _add_aresta(G, id_trab, id_kw, "keyword")


def criar_grafo_relacional(
    lista_metadados: list,
    lista_docs: list = None,
) -> nx.DiGraph:
    """
    Constrói o grafo relacional a partir de N TCCs.
    lista_docs deve estar na mesma ordem que lista_metadados.
    """
    G = nx.DiGraph()
    G.graph["tipo"] = "relacional_academico"
    for i, meta in enumerate(lista_metadados):
        doc = lista_docs[i] if lista_docs and i < len(lista_docs) else None
        adicionar_trabalho(G, meta, doc=doc)

    tipos: dict = {}
    for _, d in G.nodes(data=True):
        t = d.get("tipo","?"); tipos[t] = tipos.get(t,0) + 1
    print(f"\n  Grafo relacional: {G.number_of_nodes()} nós, "
          f"{G.number_of_edges()} arestas")
    for t, q in sorted(tipos.items(), key=lambda x: x[1], reverse=True):
        print(f"    {t:<12} → {q}")
    return G


# ══════════════════════════════════════════════════════════════════
# ANÁLISE DO GRAFO RELACIONAL
# Responde: áreas com mais projetos | professores mais envolvidos | ferramentas
# ══════════════════════════════════════════════════════════════════

def analisar_relacional(G: nx.DiGraph, top_n: int = 10) -> dict:
    print(f"\n{'═'*60}")
    print("  ANÁLISE — GRAFO RELACIONAL ACADÊMICO")
    print(f"{'═'*60}")
    print(f"  Nós: {G.number_of_nodes()}  |  Arestas: {G.number_of_edges()}")

    # ── 1. Áreas com mais projetos ──────────────────────────────
    areas: dict = {}
    for n, d in G.nodes(data=True):
        if d.get("tipo") == "AREA":
            label = d.get("label", n.split("::")[-1])
            # conta TCCs que apontam para esta área
            n_trab = sum(
                1 for pred in G.predecessors(n)
                if G.nodes[pred].get("tipo") == "TRABALHO"
            )
            areas[label] = max(n_trab, d.get("count",1))
    print(f"\n  ── Áreas com mais projetos (top {top_n}) ──")
    for label, cnt in sorted(areas.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        barra = "█" * min(cnt, 30)
        print(f"    {label:<30}  {cnt:3d}  {barra}")

    # ── 2. Professores mais envolvidos ──────────────────────────
    profs: dict = {}
    for n, d in G.nodes(data=True):
        if d.get("tipo") == "ORIENTADOR":
            label  = d.get("label", n.split("::")[-1])
            alunos = sum(
                1 for _, v, ed in G.out_edges(n, data=True)
                if ed.get("relacao") == "orientou"
            )
            trab   = sum(
                1 for _, v, ed in G.out_edges(n, data=True)
                if ed.get("relacao") == "orientou_trab"
            )
            areas_p = sum(
                1 for _, v, ed in G.out_edges(n, data=True)
                if ed.get("relacao") == "pesquisa"
            )
            profs[label] = {"alunos": alunos, "trabalhos": trab, "areas": areas_p}
    print(f"\n  ── Professores mais envolvidos (top {top_n}) ──")
    print(f"  {'Professor':<38} {'Alunos':>7} {'Trab.':>6} {'Áreas':>6}")
    print("  " + "-"*60)
    for label, p in sorted(profs.items(),
                            key=lambda x: x[1]["trabalhos"], reverse=True)[:top_n]:
        print(f"  {label:<38} {p['alunos']:>7} {p['trabalhos']:>6} {p['areas']:>6}")

    # ── 3. Ferramentas mais usadas ──────────────────────────────
    ferrs: dict = {}
    for n, d in G.nodes(data=True):
        if d.get("tipo") == "FERRAMENTA":
            label = d.get("label", n.split("::")[-1])
            n_trab = sum(
                1 for pred in G.predecessors(n)
                if G.nodes[pred].get("tipo") == "TRABALHO"
            )
            ferrs[label] = max(n_trab, d.get("count",1))
    print(f"\n  ── Ferramentas mais usadas (top {top_n}) ──")
    for label, cnt in sorted(ferrs.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        barra = "█" * min(cnt, 30)
        print(f"    {label:<25}  {cnt:3d}  {barra}")

    # ── 4. Betweenness (nós ponte) ──────────────────────────────
    G_und = G.to_undirected()
    bet   = nx.betweenness_centrality(G_und, normalized=True)
    print(f"\n  ── Top {top_n} nós ponte (betweenness) ──")
    for no, v in sorted(bet.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        label = G.nodes[no].get("label", no.split("::")[-1])
        tipo  = G.nodes[no].get("tipo","?")
        print(f"    {label:<38} [{tipo}]  {v:.4f}")

    # ── 5. Comunidades ──────────────────────────────────────────
    coms = []
    try:
        from networkx.algorithms.community import louvain_communities
        coms = louvain_communities(G_und, seed=42)
        print(f"\n  ── Comunidades Louvain: {len(coms)} ──")
        for i, com in enumerate(sorted(coms, key=len, reverse=True)[:5]):
            oris = [G.nodes[n].get("label",n) for n in com
                    if G.nodes[n].get("tipo") == "ORIENTADOR"]
            print(f"    Com.{i+1}: {len(com)} nós | orientadores: {', '.join(oris[:3])}")
    except Exception as e:
        print(f"\n  Louvain: {e}")

    return {
        "nos": G.number_of_nodes(), "arestas": G.number_of_edges(),
        "areas": areas, "professores": profs, "ferramentas": ferrs,
        "comunidades": len(coms),
    }


# ══════════════════════════════════════════════════════════════════
# BUSCA INTERATIVA
# ══════════════════════════════════════════════════════════════════

def buscar_termo(G: nx.Graph, visualizar_ego_fn=None) -> None:
    nos = sorted(G.nodes())
    print(f"\n  {len(nos)} entidades. Digite 'sair' para encerrar.\n")
    while True:
        termo = input("  Termo: ").strip().lower()
        if termo == "sair": break
        matches = [n for n in nos if termo in n]
        if not matches:
            print(f"  ✗ '{termo}' não encontrado."); continue
        if len(matches) > 1:
            print(f"  Encontrados {len(matches)}:")
            for i, m in enumerate(matches[:15]):
                print(f"    [{i}] {m}")
            escolha = input("  Número ou nome exato: ").strip()
            if escolha.isdigit() and int(escolha) < len(matches):
                termo = matches[int(escolha)]
            elif escolha in G:
                termo = escolha
            else:
                print("  Não encontrado."); continue
        else:
            termo = matches[0]
        dados    = G.nodes[termo]
        vizinhos = sorted(G[termo].items(),
                          key=lambda x: x[1].get("weight",0), reverse=True)
        print(f"\n  [{dados.get('tipo','?')}] {termo}  "
              f"(freq={dados.get('count',0)}, grau={G.degree(termo)})")
        for viz, attrs in vizinhos[:10]:
            print(f"    → [{G.nodes[viz].get('tipo','?')}] {viz:<35} "
                  f"peso={attrs.get('weight',1)}")
        if visualizar_ego_fn:
            visualizar_ego_fn(G, termo, raio=2)