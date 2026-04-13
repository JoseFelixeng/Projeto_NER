import re
import unicodedata
from collections import Counter
import spacy

# ─────────────────────────────────────────────
# MAPA DE SUBSTITUIÇÕES (ligaduras e artefatos de PDF)
# ─────────────────────────────────────────────
SUBSTITUICOES = [
    (r"Ÿ",               "ê"),
    (r"ﬁ",               "fi"),
    (r"ﬂ",               "fl"),
    (r"ﬀ",               "ff"),
    (r"ﬃ",              "ffi"),
    (r"ﬄ",              "ffl"),
    (r"\u00A0",          " "),
    (r"\u2009",          " "),
    (r"[\u2013\u2014]",  "-"),
    (r"[\u201C\u201D\u201E]", '"'),
    (r"[\u2018\u2019]",  "'"),
]

# Abreviações que NÃO encerram parágrafo
ABREVIACOES = re.compile(
    r"\b("
    r"[Pp]rof|[Dd]r|[Ss]r|[Ss]ra|[Ee]ng|[Aa]dv|"
    r"[Ee]t al|[Aa]pud|[Oo]p\.?\s*cit|[Ii]bid|"
    r"fig|tab|eq|cap|vol|num|n[uú]m|p[aá]g|pp|"
    r"jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez|"
    r"[A-Z]"
    r")\.",
    re.IGNORECASE,
)

# Siglas técnicas preservadas pelo filtro de caixa alta
SIGLAS_TECNICAS = re.compile(
    r"^("
    r"LSTM|CNN|RNN|GAN|NLP|NER|API|GPU|CPU|RAM|"
    r"UFRN|UFPB|UFC|USP|UFMG|UFSC|"
    r"HTTP|REST|SQL|NoSQL|JSON|XML|CSV|PDF|"
    r"BFS|DFS|KNN|SVM|MLP|DBN|AE|DAE|SAE|"
    r"IoT|ML|DL|AI|LLM|RAG|PLN|"
    r"IEEE|ACM|ABNT|NBR"
    r")$"
)

# ─────────────────────────────────────────────
# PADRÕES DE RUÍDO ESTRUTURAL DE TCC
# Identificados pela análise dos grafos gerados
# ─────────────────────────────────────────────

# Fragmentos de sumário/lista que escaparam da limpeza e viraram nós no grafo
# Ex: "lista de ilustrações figura", "fundamentação teórica capítulo"
RUIDO_ESTRUTURAL = re.compile(
    r"\b("
    r"lista de (ilustra[çc][õo]es?|figuras?|tabelas?|abreviaturas?|siglas?)|"
    r"(fundamenta[çc][aã]o|referencial)\s+te[oó]rica?|"
    r"cap[íi]tulo\s+(conclus[aã]o|experimentos?|resultados?|introdu[çc][aã]o)|"
    r"sumário\s+sumário|"
    r"resultados?\s+figura|"
    r"implementa[çc][aã]o\s+figura|"
    r"p[aá]gina\s+na\s+internet|"       # nó falso recorrente nos grafos
    r"in\s+this\s+context|"             # fragmento inglês classificado como MISC
    r"this\s+work\s+presents?|"
    r"moreover|furthermore|nevertheless" # conectivos em inglês viram nós
    r")\b",
    re.IGNORECASE,
)

# Frases longas em inglês dentro de TCCs (texto técnico misturado)
# Capturadas pelo NER como MISC — filtro por comprimento de tokens aplicado no NER
MAX_TOKENS_ENTIDADE = 5   # entidades com mais palavras que isso são descartadas

# Padrões de entidades que são claramente artefatos de extração PDF
# Ex: "all experiments were conducted using an nvidia tesla vgpu"
ENTIDADE_ARTEFATO = re.compile(
    r"^("
    r"all\s+experiments?\s+were|"
    r"american\s+standard\s+code|"
    r"this\s+(work|study|paper)\s+(presents?|proposes?|describes?)|"
    r"including\s+data\s+load|"
    r"enables?\s+the\s+integrat|"
    r"to\s+address\s+the\s+(e+cts?|effects?)|"
    r"rich\s+framework\s+of|"
    r"table\s+contracting\s+path|"
    r"furthermore|moreover|nevertheless|"
    r"license\s+c[bc]|"                  # "license cb observabilidade..."
    r"abstract\s+this\s+study"
    r")",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────
# ETAPA 1 — Leitura
# ─────────────────────────────────────────────
def ler_arquivo(caminho: str) -> str:
    with open(caminho, encoding="utf-8", errors="replace") as f:
        return f.read()


# ─────────────────────────────────────────────
# ETAPA 2 — Normalização de caracteres
# ─────────────────────────────────────────────
def normalizar_caracteres(texto: str) -> str:
    for padrao, substituto in SUBSTITUICOES:
        texto = re.sub(padrao, substituto, texto)
    return texto


# ─────────────────────────────────────────────
# ETAPA 3 — Recorte do corpo técnico
# ─────────────────────────────────────────────
def recortar_corpo(texto: str) -> str:
    inicio = re.search(r"\b(RESUMO|ABSTRACT)\b", texto)
    fim    = re.search(
        r"\b(REFER[ÊEĹ]NCIAS?|BIBLIOGRAPHY|BIBLIOGRAF[ÍI]A)\b", texto
    )
    if inicio and fim and inicio.start() < fim.start():
        return texto[inicio.start(): fim.start()]
    return texto


# ─────────────────────────────────────────────
# ETAPA 4 — Remoção de ruídos estruturais
# ─────────────────────────────────────────────
def remover_ruidos(texto: str) -> str:
    # Números de página isolados
    texto = re.sub(r"^\d{1,4}\s*$", "", texto, flags=re.MULTILINE)

    # Legendas de figuras/tabelas
    texto = re.sub(
        r"^(Figura|Tabela|Gráfico|Quadro|Esquema|Imagem)\s+\d+[\s\-–—].*$",
        "",
        texto,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Linhas de fonte de tabela
    texto = re.sub(r"^Fonte:.*$", "", texto, flags=re.MULTILINE | re.IGNORECASE)

    # NOVO — Remove padrões estruturais de TCC identificados nos grafos
    # Ex: "Lista de Ilustrações Figura", "Fundamentação Teórica Capítulo"
    texto = RUIDO_ESTRUTURAL.sub("", texto)

    # NOVO — Remove linhas que são URLs ou caminhos técnicos
    # Ex: "https://www.datacamp.com/tutorial/set-up-and-configure-mysql-in-docker"
    texto = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        texto,
        flags=re.MULTILINE,
    )

    # NOVO — Remove linhas de cabeçalho de monografia repetitivas
    # Ex: "universidade federal do rio grande do norte centro de tecnologia graduação..."
    texto = re.sub(
        r"^universidade\s+(federal|do)\s+.{10,120}graduação.*$",
        "",
        texto,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Cabeçalhos em CAIXA ALTA (preserva siglas técnicas)
    def _remover_se_nao_sigla(m):
        linha = m.group(0).strip()
        if SIGLAS_TECNICAS.match(linha):
            return m.group(0)
        return ""

    texto = re.sub(
        r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ\s]{2,60}$",
        _remover_se_nao_sigla,
        texto,
        flags=re.MULTILINE,
    )

    # Linhas com 1 único caractere não-espaço
    texto = re.sub(r"^\s*\S\s*$", "", texto, flags=re.MULTILINE)

    return texto


# ─────────────────────────────────────────────
# ETAPA 5 — Restauração de parágrafos
# ─────────────────────────────────────────────
def restaurar_paragrafos(texto: str) -> str:
    MARCA = "\x00"
    texto = ABREVIACOES.sub(lambda m: m.group(0).replace(".", MARCA), texto)
    texto = re.sub(
        r"([.!?])\s{2,}([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ])",
        r"\1\n\n\2",
        texto,
    )
    texto = texto.replace(MARCA, ".")
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# ─────────────────────────────────────────────
# ETAPA 6 — Limpeza final
# ─────────────────────────────────────────────
def limpeza_final(texto: str) -> str:
    texto = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", texto)
    texto = re.sub(r"[\uE000-\uF8FF]+", "", texto)
    texto = "".join(
        c for c in texto
        if unicodedata.category(c)[0] not in ("C", "Z") or c in ("\n", "\t", " ")
    )
    paragrafos = texto.split("\n\n")
    paragrafos = [p.strip() for p in paragrafos if p.strip()]
    return "\n\n".join(paragrafos)


# ─────────────────────────────────────────────
# DIAGNÓSTICO
# ─────────────────────────────────────────────
def diagnosticar(texto: str, n: int = 20) -> None:
    letras_pt = set("áéíóúàâêôãõçüÁÉÍÓÚÀÂÊÔÃÕÇÜ")
    suspeitos  = [c for c in texto if ord(c) > 127 and c not in letras_pt]
    contagem   = Counter(suspeitos).most_common(n)
    print("─── Chars suspeitos restantes ───")
    for char, freq in contagem:
        print(f"  U+{ord(char):04X}  '{char}'  →  {freq}×")


# ─────────────────────────────────────────────
# FILTRO DE ENTIDADES NER
# Aplicado APÓS o spaCy — remove ruídos que o modelo não consegue evitar
# ─────────────────────────────────────────────
def filtrar_entidade(texto_ent: str, label: str) -> bool:
    """
    Retorna True se a entidade deve ser MANTIDA, False se deve ser descartada.

    Regras baseadas nos problemas observados nos grafos:
    1. Entidades muito longas (frases inteiras classificadas como MISC)
    2. Padrões de artefatos de PDF identificados nos grafos
    3. Entidades com menos de 2 caracteres (letras isoladas)
    4. Conectivos e fragmentos em inglês capturados como MISC
    """
    texto_limpo = texto_ent.strip()

    # Regra 1 — comprimento máximo em tokens
    if len(texto_limpo.split()) > MAX_TOKENS_ENTIDADE:
        return False

    # Regra 2 — padrões de artefato identificados nas figuras
    if ENTIDADE_ARTEFATO.match(texto_limpo):
        return False

    # Regra 3 — entidades muito curtas (1 char) ou só números
    if len(texto_limpo) < 2 or texto_limpo.isdigit():
        return False

    # Regra 4 — MISC com texto puramente em inglês (heurística por stopwords)
    # Se for MISC e tiver palavras como "the", "this", "using", "were" → artefato
    STOPWORDS_EN = {"the", "this", "using", "were", "with", "that",
                    "from", "into", "and", "for", "are", "has", "have",
                    "been", "its", "their", "these", "those", "which"}
    if label == "MISC":
        palavras = set(texto_limpo.lower().split())
        if palavras & STOPWORDS_EN:
            return False

    # Regra 5 — fragmentos de estrutura documental que ainda escapam
    if RUIDO_ESTRUTURAL.search(texto_limpo):
        return False

    return True


# ─────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────
def preprocessar(caminho: str, diagnostico: bool = False) -> str:
    texto = ler_arquivo(caminho)
    texto = normalizar_caracteres(texto)
    texto = recortar_corpo(texto)
    texto = remover_ruidos(texto)
    texto = restaurar_paragrafos(texto)
    texto = limpeza_final(texto)
    if diagnostico:
        diagnosticar(texto)
    return texto


# ─────────────────────────────────────────────
# PROCESSAMENTO COM SPACY
# ─────────────────────────────────────────────
def processar_spacy(texto: str):
    """
    Carrega o modelo português, processa o texto e imprime diagnóstico.
    O filtro filtrar_entidade() deve ser aplicado nas funções de janela
    do grafo_ner.py ao consumir doc.ents.
    """
    MODELO = "pt_core_news_lg"

    nlp = spacy.load(MODELO)
    nlp.max_length = 2_000_000
    doc = nlp(texto)

    # diagnóstico — mostra o que o NER capturou antes da filtragem
    total_ents  = len(doc.ents)
    apos_filtro = [e for e in doc.ents if filtrar_entidade(e.text, e.label_)]

    print(f"\n─── Sentenças detectadas: {len(list(doc.sents))} ───")
    for i, sent in enumerate(doc.sents):
        if i >= 5:
            break
        print(f"  [{i+1}] {sent.text[:90]}")

    print(f"\n─── Entidades brutas: {total_ents}  →  após filtro: {len(apos_filtro)} ───")
    print("  Primeiras 20 entidades após filtro:")
    for ent in apos_filtro[:20]:
        print(f"  {ent.text:<35} → {ent.label_}")

    print(f"\n─── Tokens relevantes (sem stopwords, primeiros 15) ───")
    tokens = [
        t.lemma_
        for t in doc
        if not t.is_stop and not t.is_punct and t.is_alpha and len(t) > 2
    ]
    print("  ", tokens[:15])

    return doc