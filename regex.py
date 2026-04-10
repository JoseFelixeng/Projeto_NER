import re # Regex 
import spacy # Biblioteca para a analise de linguagem natural 


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
def ler_arquivo(caminhos: str) -> str:
    """Lê o arquivo garantindo UTF-8; substitui bytes inválidos."""
    with open(caminhos, encoding="utf-8", errors="replace") as f:
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
      • Remove caracteres da área de uso privado (U+E000–U+F8FF)
      • Remove caracteres não imprimíveis diversos
      • Trim em cada parágrafo
    """
    # 1. Caracteres de controle indesejados (já existente)
    texto = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", texto)

    # 2. Remove toda a faixa de uso privado (Private Use Area)
    texto = re.sub(r"[\uE000-\uF8FF]+", "", texto)

    # 3. Remove outros caracteres Unicode não comuns em texto legível
    #    (categoria "Other, Control", "Other, Format", "Other, Surrogate")
    import unicodedata
    texto = ''.join(
        c for c in texto
        if unicodedata.category(c)[0] not in ('C', 'Z') or c in ('\n', '\t', ' ')
    )

    # 4. Trim por parágrafo
    paragrafos = texto.split("\n\n")
    paragrafos = [p.strip() for p in paragrafos if p.strip()]
    return "\n\n".join(paragrafos)

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


# PIPELINE COMPLETO
def preprocessar(caminhos: str) -> str:
    texto = ler_arquivo(caminhos)
    texto = normalizar_caracteres(texto)
    texto = recortar_corpo(texto)
    texto = remover_ruidos(texto)
    texto = restaurar_paragrafos(texto)
    texto = limpeza_final(texto)
    return texto

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