"""
Pipeline de pré-processamento com Regex para uso com spaCy
Arquivo-alvo: AnalisedeModelos_Oliveira_2025.txt (TCC extraído de PDF)
"""

import re # Regex 
import spacy # Biblioteca para a analise de linguagem natural 
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from collections import Counter
# MAPA DE SUBSTITUIÇÕES (ligaduras e artefatos de PDF)

SUBSTITUICOES = [
    # Artefato específico deste arquivo (ê corrompido)
    (r"Ÿ",          "ê"),
    # Ligaduras tipográficas comuns em PDFs
    (r"ﬁ",          "fi"),
    (r"ﬂ",          "fl"),
    (r"ﬀ",          "ff"),
    (r"ﬃ",         "ffi"),
    (r"ﬄ",         "ffl"),
    # Espaços especiais
    (r"\u00A0",     " "),   # non-breaking space
    (r"\u2009",     " "),   # thin space
    # Traços tipográficos → hífen simples
    (r"[\u2013\u2014]", "-"),
    # Aspas tipográficas → aspas simples
    (r"[\u201C\u201D\u201E]", '"'),
    (r"[\u2018\u2019]",       "'"),
]

nlp = spacy.load("pt_core_news_sm")

# ETAPA 1 — Leitura
def ler_arquivo(caminho: str) -> str:
    """Lê o arquivo garantindo UTF-8; substitui bytes inválidos."""
    with open(caminho, encoding="utf-8", errors="replace") as f:
        return f.read()



# ETAPA 2 — Normalização de caracteres com Regex

def normalizar_caracteres(texto: str) -> str:
    """
    Aplica todas as substituições de ligaduras e artefatos de PDF
    usando re.sub, que é mais rápido e legível que múltiplos str.replace.
    """
    for padrao, substituto in SUBSTITUICOES:
        texto = re.sub(padrao, substituto, texto)
    return texto



# ETAPA 3 — Remoção de seções irrelevantes

def recortar_corpo(texto: str) -> str:
    """
    Extrai o corpo técnico do TCC:
    começa no RESUMO, termina antes das REFERÊNCIAS.
    re.search localiza os marcadores sem precisar iterar linha a linha.
    """
    inicio = re.search(r"\bRESUMO\b", texto)
    fim    = re.search(r"\bREFER[ÊE]NCIAS?\b|\bREFERĹNCIAS\b", texto)

    if inicio and fim:
        return texto[inicio.start() : fim.start()]
    return texto  # fallback: retorna tudo se não encontrar



# ETAPA 4 — Remoção de ruídos estruturais com Regex
def remover_ruidos(texto: str) -> str:
    """
    Remove com re.sub (flags MULTILINE para ^ e $ por linha):
      • Números de página isolados
      • Legendas de figura/tabela
      • Cabeçalhos repetitivos (nome do autor, título da obra)
      • Linhas muito curtas (sobras de layout)
    """
    # Números de página isolados (linha só com dígitos)
    texto = re.sub(r"^\d{1,4}\s*$", "", texto, flags=re.MULTILINE)

    # Legendas: "Figura 3 –", "Tabela 1 –", "Gráfico 2 –" etc.
    texto = re.sub(
        r"^(Figura|Tabela|Gráfico|Quadro|Esquema|Imagem)\s+\d+[\s\-–—].*$",
        "",
        texto,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Fonte: linhas de rodapé de tabela
    texto = re.sub(r"^Fonte:.*$", "", texto, flags=re.MULTILINE | re.IGNORECASE)

    # Cabeçalhos de seção em caixa alta (ex: "CAPÍTULO 1", "INTRODUÇÃO")
    # Mantemos apenas se forem curtos demais para ter conteúdo real
    texto = re.sub(r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ\s]{2,40}$\n", "", texto, flags=re.MULTILINE)

    # Linhas com menos de 3 caracteres não-espaço (sobras de layout)
    texto = re.sub(r"^\s{0,5}\S{1,2}\s*$", "", texto, flags=re.MULTILINE)
    return texto


# ETAPA 5 — Restauração de parágrafos
def restaurar_paragrafos(texto: str) -> str:
    """
    O texto de PDF vem colado numa linha só.
    Estratégia com Regex:
      1. Insere quebra dupla após ponto/exclamação/interrogação
         seguido de espaço e letra maiúscula (início de nova frase).
      2. Normaliza espaços internos.
      3. Colapsa linhas em branco excessivas.
    """
    # Quebra de parágrafo após pontuação final + letra maiúscula
    texto = re.sub(
        r"([.!?])\s+([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ])",
        r"\1\n\n\2",
        texto,
    )

    # Remove espaços múltiplos dentro de uma linha
    texto = re.sub(r"[ \t]{2,}", " ", texto)

    # Colapsa mais de 2 quebras de linha consecutivas
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()



# ETAPA 6 — Limpeza final
def limpeza_final(texto: str) -> str:
    """
    Polimento final:
      • Remove caracteres de controle (exceto \n e \t)
      • Trim em cada parágrafo
    """
    # Caracteres de controle indesejados
    texto = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", texto)

    # Trim por parágrafo
    paragrafos = texto.split("\n\n")
    paragrafos = [p.strip() for p in paragrafos if p.strip()]
    return "\n\n".join(paragrafos)



# PIPELINE COMPLETO
def preprocessar(caminho: str) -> str:
    texto = ler_arquivo(caminho)
    texto = normalizar_caracteres(texto)
    texto = recortar_corpo(texto)
    texto = remover_ruidos(texto)
    texto = restaurar_paragrafos(texto)
    texto = limpeza_final(texto)
    return texto



# DIAGNÓSTICO — chars suspeitos que restaram
def diagnosticar(texto: str, n: int = 20) -> None:
    """
    Lista os N caracteres não-ASCII mais frequentes que sobraram.
    Útil para descobrir artefatos que o pipeline ainda não cobre.
    """
    from collections import Counter

    letras_pt = set("áéíóúàâêôãõçüÁÉÍÓÚÀÂÊÔÃÕÇÜ")
    suspeitos  = [c for c in texto if ord(c) > 127 and c not in letras_pt]
    contagem   = Counter(suspeitos).most_common(n)

    print("─── Chars suspeitos restantes ───")
    for char, freq in contagem:
        print(f"  U+{ord(char):04X}  '{char}'  →  {freq}×")


# ─────────────────────────────────────────────
# SPACY
# ─────────────────────────────────────────────
def processar_spacy(texto: str):
    """
    Carrega o modelo português e processa o texto limpo.
    Instale antes:  python -m spacy download pt_core_news_lg
    """
    nlp = spacy.load("pt_core_news_lg")
    nlp.max_length = 2_000_000  # TCC pode ser longo

    doc = nlp(texto)

    print(f"\n─── Sentenças detectadas: {len(list(doc.sents))} ───")
    for i, sent in enumerate(doc.sents):
        if i >= 5:
            break
        print(f"  [{i+1}] {sent.text[:90]}")

    print(f"\n─── Entidades nomeadas (primeiras 10) ───")
    for ent in list(doc.ents)[:10]:
        print(f"  {ent.text:<30} → {ent.label_}")

    print(f"\n─── Tokens relevantes (sem stopwords, primeiros 15) ───")
    tokens = [
        t.lemma_
        for t in doc
        if not t.is_stop and not t.is_punct and t.is_alpha and len(t) > 2
    ]
    print("  ", tokens[:15])

    return doc

def extrair_entidades(doc):
    lista = []

    for sent in doc.sents:
        ents = list(set([ent.text.strip() for ent in sent.ents if len(ent.text) > 2]))

        if len(ents) > 1:
            lista.append(ents)

    return lista



def visualizar_ego_galaxia(G, termo_central):
    import matplotlib.pyplot as plt
    import networkx as nx

    if termo_central not in G:
        print(f"Termo '{termo_central}' não encontrado no grafo.")
        return

    # Criar ego graph
    ego = nx.ego_graph(G, termo_central, radius=2)

    centralidade = nx.degree_centrality(ego)

    # Layout tipo galáxia
    pos = nx.spring_layout(ego, center=(0, 0), k=0.5)

    node_sizes = []
    node_colors = []

    for node in ego.nodes():
        if node == termo_central:
            node_sizes.append(8000)  # centro gigante
            node_colors.append(1.0)
        else:
            valor = centralidade[node]
            node_sizes.append(3000 * valor)  # menores orbitam
            node_colors.append(valor)

    plt.figure(figsize=(12, 10))

    nx.draw_networkx_nodes(
        ego,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.plasma,
        alpha=0.9
    )

    nx.draw_networkx_edges(
        ego,
        pos,
        alpha=0.2
    )

    # Mostrar labels só relevantes
    labels = {
        n: n for n in ego.nodes()
        if centralidade[n] > 0.05 or n == termo_central
    }

    nx.draw_networkx_labels(
        ego,
        pos,
        labels,
        font_size=9
    )

    plt.title(f"🌌 Ego Graph - {termo_central}")
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
        print(f"{no} → {valor:.4f}")


def buscar_termo(G):
    while True:
        termo = input("\n🔍 Digite um termo (ou 'sair'): ")

        if termo.lower() == "sair":
            break

        visualizar_ego_galaxia(G, termo)

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




# EXECUÇÃO
if __name__ == "__main__":
    CAMINHO = "AnalisedeModelos_Oliveira_2025.txt"

    print("Processando...")
    texto_limpo = preprocessar(CAMINHO)

    print(f"Caracteres após limpeza: {len(texto_limpo)}")
    print(f"Parágrafos: {texto_limpo.count(chr(10)*2) + 1}")
    print("\n─── Amostra ───")
    print(texto_limpo[:500])

    # Diagnóstico de chars que sobraram
    diagnosticar(texto_limpo)

    # Processar com spaCy
    doc = processar_spacy(texto_limpo)


    print("Extraindo entidades...")
    lista_entidades = extrair_entidades(doc)

    print("Criando grafo...")
    G = criar_grafo(lista_entidades)

    print("Analisando grafo...")
    analisar_grafo(G)

    print("Visualizando...")
    nx.ego_graph(G, "UFRN", radius=2)

    print("Entrando no modo de busca...")
    buscar_termo(G)
    
    # tokens relevantes
    tokens = [
        t.lemma_
        for t in doc
        if not t.is_stop and not t.is_punct and t.is_alpha and len(t) > 2
    ]

    # frequência
    contagem = Counter(tokens)
    palavras, freq = zip(*contagem.most_common(30))

    # gerar posições tipo galáxia
    x = np.random.normal(0, 1, len(palavras))
    y = np.random.normal(0, 1, len(palavras))

    df = pd.DataFrame({
        "palavra": palavras,
        "freq": freq,
        "x": x,
        "y": y
    })

    # gráfico
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers+text",
        text=df["palavra"],          # 🔥 nomes como na imagem
        textposition="top center",
        marker=dict(
            size=df["freq"],        # 🔥 tamanho = relevância
            color="white",
            opacity=0.8
        )
    ))

    # estilo espaço
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        title="Mapa Semântico (Estilo Galáxia)"
    )

    fig.show()