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
    (r"\u00A0",          " "),    # non-breaking space
    (r"\u2009",          " "),    # thin space
    (r"[\u2013\u2014]",  "-"),    # traços tipográficos
    (r"[\u201C\u201D\u201E]", '"'),
    (r"[\u2018\u2019]",  "'"),
]

# Abreviações comuns em TCCs de Eng. Comp. que NÃO encerram parágrafo
# Adicionado para evitar quebra errada na restaurar_paragrafos
ABREVIACOES = re.compile(
    r"\b("
    r"[Pp]rof|[Dd]r|[Ss]r|[Ss]ra|[Ee]ng|[Aa]dv|"   # títulos
    r"[Ee]t al|[Aa]pud|[Oo]p\.?\s*cit|[Ii]bid|"      # citações
    r"fig|tab|eq|cap|vol|num|n[uú]m|p[aá]g|pp|"       # referências
    r"jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez|" # meses
    r"[A-Z]"                                            # iniciais (ex: "J. Silva")
    r")\.",
    re.IGNORECASE,
)

# Siglas técnicas que NÃO devem ser removidas pelo filtro de caixa alta
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
# ETAPA 1 — Leitura
# ─────────────────────────────────────────────
def ler_arquivo(caminho: str) -> str:
    """Lê o arquivo garantindo UTF-8; substitui bytes inválidos."""
    with open(caminho, encoding="utf-8", errors="replace") as f:
        return f.read()


# ─────────────────────────────────────────────
# ETAPA 2 — Normalização de caracteres
# ─────────────────────────────────────────────
def normalizar_caracteres(texto: str) -> str:
    """Substitui ligaduras, artefatos de PDF e caracteres especiais."""
    for padrao, substituto in SUBSTITUICOES:
        texto = re.sub(padrao, substituto, texto)
    return texto


# ─────────────────────────────────────────────
# ETAPA 3 — Recorte do corpo técnico
# ─────────────────────────────────────────────
def recortar_corpo(texto: str) -> str:
    """
    Extrai o corpo técnico do TCC:
    começa no RESUMO (ou ABSTRACT), termina antes das REFERÊNCIAS.
    Aceita variações tipográficas comuns em PDFs mal extraídos.
    """
    inicio = re.search(
        r"\b(RESUMO|ABSTRACT)\b",
        texto
    )
    fim = re.search(
        r"\b(REFER[ÊEĹ]NCIAS?|BIBLIOGRAPHY|BIBLIOGRAF[ÍI]A)\b",
        texto
    )

    if inicio and fim and inicio.start() < fim.start():
        return texto[inicio.start(): fim.start()]
    return texto  # fallback: retorna tudo


# ─────────────────────────────────────────────
# ETAPA 4 — Remoção de ruídos estruturais
# ─────────────────────────────────────────────
def remover_ruidos(texto: str) -> str:
    """
    Remove ruídos de layout sem apagar entidades técnicas relevantes para o NER.

    MUDANÇAS em relação à versão anterior:
    - Filtro de caixa alta agora preserva siglas técnicas (LSTM, CNN, UFRN etc.)
    - Removido filtro de linhas curtas agressivo que apagava siglas de 2-3 chars
    """
    # Números de página isolados
    texto = re.sub(r"^\d{1,4}\s*$", "", texto, flags=re.MULTILINE)

    # Legendas: "Figura 3 –", "Tabela 1 –", etc.
    texto = re.sub(
        r"^(Figura|Tabela|Gráfico|Quadro|Esquema|Imagem)\s+\d+[\s\-–—].*$",
        "",
        texto,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Linhas de fonte de tabela
    texto = re.sub(r"^Fonte:.*$", "", texto, flags=re.MULTILINE | re.IGNORECASE)

    # Cabeçalhos em CAIXA ALTA — mas preserva siglas técnicas
    def _remover_se_nao_sigla(m):
        linha = m.group(0).strip()
        if SIGLAS_TECNICAS.match(linha):
            return m.group(0)   # preserva
        return ""               # remove

    texto = re.sub(
        r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ\s]{2,60}$",
        _remover_se_nao_sigla,
        texto,
        flags=re.MULTILINE,
    )

    # Linhas com 1 único caractere não-espaço (sobras de layout)
    # MUDANÇA: era \S{1,2}, agora só remove linhas de 1 char
    # para não apagar siglas como "ML", "AI", etc.
    texto = re.sub(r"^\s*\S\s*$", "", texto, flags=re.MULTILINE)

    return texto


# ─────────────────────────────────────────────
# ETAPA 5 — Restauração de parágrafos
# ─────────────────────────────────────────────
def restaurar_paragrafos(texto: str) -> str:
    """
    Reconstrói parágrafos a partir de texto linearizado de PDF.

    MUDANÇA PRINCIPAL: antes de quebrar em parágrafo, verifica se o ponto
    pertence a uma abreviação conhecida (Prof., Dr., et al., fig., etc.)
    para evitar fragmentação incorreta de sentenças.
    """
    # Marcador temporário para proteger abreviações
    MARCA = "\x00"

    # Protege abreviações substituindo o ponto final por marcador
    texto = ABREVIACOES.sub(lambda m: m.group(0).replace(".", MARCA), texto)

    # Agora quebra parágrafos somente em pontuação real
    texto = re.sub(
        r"([.!?])\s{2,}([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ])",
        r"\1\n\n\2",
        texto,
    )

    # Restaura os pontos protegidos
    texto = texto.replace(MARCA, ".")

    # Normaliza espaços e quebras excessivas
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()


# ─────────────────────────────────────────────
# ETAPA 6 — Limpeza final
# ─────────────────────────────────────────────
def limpeza_final(texto: str) -> str:
    """
    Polimento final: remove caracteres de controle, PUA e não-imprimíveis.
    Preserva \n, \t e espaço.
    """
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
    """Lista os N caracteres não-ASCII mais frequentes que sobraram."""
    letras_pt = set("áéíóúàâêôãõçüÁÉÍÓÚÀÂÊÔÃÕÇÜ")
    suspeitos  = [c for c in texto if ord(c) > 127 and c not in letras_pt]
    contagem   = Counter(suspeitos).most_common(n)

    print("─── Chars suspeitos restantes ───")
    for char, freq in contagem:
        print(f"  U+{ord(char):04X}  '{char}'  →  {freq}×")


# ─────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────
def preprocessar(caminho: str, diagnostico: bool = False) -> str:
    """
    Executa o pipeline completo de limpeza.
    Se diagnostico=True, imprime chars suspeitos ao final.
    """
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
    Carrega o modelo português e processa o texto limpo.

    MUDANÇA: modelo definido em um único lugar (evita inconsistência
    entre sm no topo do arquivo e lg aqui dentro).
    """
    MODELO = "pt_core_news_lg"  # troque por sm se não tiver o lg instalado

    nlp = spacy.load(MODELO)
    nlp.max_length = 2_000_000
    doc = nlp(texto)

    print(f"\n─── Sentenças detectadas: {len(list(doc.sents))} ───")
    for i, sent in enumerate(doc.sents):
        if i >= 5:
            break
        print(f"  [{i+1}] {sent.text[:90]}")

    print(f"\n─── Entidades nomeadas (primeiras 20) ───")
    for ent in list(doc.ents)[:20]:
        print(f"  {ent.text:<30} → {ent.label_}")

    print(f"\n─── Tokens relevantes (sem stopwords, primeiros 15) ───")
    tokens = [
        t.lemma_
        for t in doc
        if not t.is_stop and not t.is_punct and t.is_alpha and len(t) > 2
    ]
    print("  ", tokens[:15])

    return doc