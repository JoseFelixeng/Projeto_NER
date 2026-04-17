"""
preprocessamento.py — v4
Pipeline: limpeza → NER híbrido (spaCy + manual) → metadados.

Hierarquia do grafo:
  UNIV → DEPT → ORIENTADOR → AUTOR → TRABALHO → AREA → FERRAMENTA
"""

import re, unicodedata
from collections import Counter
import spacy
from spacy.tokens import Doc
from spacy.util import filter_spans

# ══════════════════════════════════════════════════════════════════
# 1. DICIONÁRIOS NER MANUAL
# ══════════════════════════════════════════════════════════════════
TERMOS_TEC = {
    "LSTM","CNN","RNN","GAN","Transformer","BERT","GPT","LLaMA",
    "YOLO","ResNet","VGG","AlexNet","U-Net","SVM","KNN",
    "Random Forest","XGBoost","LightGBM","CatBoost","K-Means",
    "DBSCAN","PCA","t-SNE","Autoencoder","VAE","DBN","MLP",
    "Seq2Seq","GCN","GAT","RAG","LoRA","QLoRA",
    "Graph Neural Network","Federated Learning","Transfer Learning",
}
TERMOS_AREA = {
    "Machine Learning","Deep Learning","Redes Neurais",
    "Visão Computacional","Processamento de Linguagem Natural",
    "PLN","NLP","Sistemas Embarcados","Internet das Coisas","IoT",
    "Robótica","Inteligência Artificial","IA","Big Data",
    "Data Science","Engenharia de Software","Computação em Nuvem",
    "Edge Computing","Segurança da Informação","Blockchain",
    "Teoria dos Grafos","Redes Complexas","Processamento de Sinais",
    "Sistemas Distribuídos","Banco de Dados","Computação Gráfica",
    "Redes de Computadores","Automação","Controle Automático",
    "Aprendizado por Reforço","Multimodal Learning",
}
TERMOS_FERRAMENTA = {
    "TensorFlow","PyTorch","Keras","Scikit-learn","Pandas","NumPy",
    "OpenCV","spaCy","Hugging Face","LangChain","LangGraph",
    "LlamaIndex","FastAPI","Docker","Streamlit","Git","GitHub",
    "Jupyter","Colab","Elasticsearch","Kibana","NetworkX","Gephi",
    "Pyvis","Matplotlib","Plotly","Seaborn","nxviz",
    "PostgreSQL","MongoDB","Redis","Kafka","Spark","Airflow",
    "Kubernetes","Terraform","Grafana","Prometheus",
    "Ollama","Chroma","FAISS","Pinecone","MLflow","DVC",
}
TERMOS_HARDWARE = {
    "GPU","CPU","TPU","FPGA","RAM","Arduino",
    "Raspberry Pi","ESP32","NVIDIA Jetson",
}
TERMOS_CONCEITO = {
    "Acurácia","Precisão","Recall","F1-score","Overfitting",
    "Underfitting","Dropout","Batch Normalization","Gradient Descent",
    "Adam","SGD","Embedding","Word2Vec","GloVe",
    "Coeficiente de Agrupamento","Centralidade","Betweenness",
    "Pagerank","Clustering","Densidade","Diâmetro",
    "Componentes Conectados","Named Entity Recognition","NER",
    "co-ocorrência","Co-Ocorrência",
}

TERMOS_ENG_COMP: dict = {}
for _t in TERMOS_TEC:        TERMOS_ENG_COMP[_t] = "TEC"
for _t in TERMOS_AREA:       TERMOS_ENG_COMP[_t] = "AREA"
for _t in TERMOS_FERRAMENTA: TERMOS_ENG_COMP[_t] = "FERRAMENTA"
for _t in TERMOS_HARDWARE:   TERMOS_ENG_COMP[_t] = "HARDWARE"
for _t in TERMOS_CONCEITO:   TERMOS_ENG_COMP[_t] = "CONCEITO"

# Professores DCA — armazenados em NFD+lowercase para busca robusta
PROFESSORES_DCA = {
    "adelardo adelino dantas de medeiros",
    "adriao duarte doria neto",
    "agostinho de medeiros brito junior",
    "anderson luiz de oliveira cavalcanti",
    "andre laurindo maitelli",
    "andres ortiz salazar",
    "carlos eduardo trabuco dorea",
    "carlos manuel dias viegas",
    "diogo pinheiro fernandes pedrosa",
    "eduardo de lucena falcao",
    "fabio meneghetti ugulino de araujo",
    "francisco das chagas mota",
    "ivanovitch medeiros dantas da silva",
    "jose ivonildo do rego",
    "luiz affonso henderson guedes de oliveira",
    "luiz felipe de queiroz silveira",
    "luiz marcos garcia goncalves",
    "manoel firmino de medeiros junior",
    "marcelo augusto costa fernandes",
    "pablo javier alsina",
    "paulo sergio da motta pires",
    "ricardo ferreira pinheiro",
    "samuel xavier de souza",
    "sebastian yuri cavalcanti catunda",
    "tiago tavares leite barros",
}

UNIVERSIDADES = {
    "UFRN":"Universidade Federal do Rio Grande do Norte",
    "UFPB":"Universidade Federal da Paraíba",
    "UFC":"Universidade Federal do Ceará",
    "UFPE":"Universidade Federal de Pernambuco",
    "UFAL":"Universidade Federal de Alagoas",
    "UFBA":"Universidade Federal da Bahia",
    "UFERSA":"Universidade Federal Rural do Semi-Árido",
    "IFRN":"Instituto Federal do Rio Grande do Norte",
    "USP":"Universidade de São Paulo",
    "UNICAMP":"Universidade Estadual de Campinas",
    "UFMG":"Universidade Federal de Minas Gerais",
    "UFSC":"Universidade Federal de Santa Catarina",
    "UFRJ":"Universidade Federal do Rio de Janeiro",
    "UFPR":"Universidade Federal do Paraná",
    "UnB":"Universidade de Brasília",
}

# chaves em NFD+lowercase para comparação robusta
UNIVERSIDADES_NOMES_LONGOS = {
    "universidade federal do rio grande do norte":"UFRN",
    "universidade federal da paraiba":"UFPB",
    "universidade federal do ceara":"UFC",
    "universidade federal de pernambuco":"UFPE",
    "universidade federal de alagoas":"UFAL",
    "universidade federal da bahia":"UFBA",
    "universidade federal rural do semi-arido":"UFERSA",
    "instituto federal do rio grande do norte":"IFRN",
    "universidade de sao paulo":"USP",
    "universidade estadual de campinas":"UNICAMP",
    "universidade federal de minas gerais":"UFMG",
    "universidade federal de santa catarina":"UFSC",
    "universidade federal do rio de janeiro":"UFRJ",
    "universidade federal do parana":"UFPR",
    "universidade de brasilia":"UnB",
}

DEPARTAMENTOS = {
    "DCA":"Departamento de Computação e Automação",
    "DIMAp":"Departamento de Informática e Matemática Aplicada",
    "CT":"Centro de Tecnologia",
    "CCET":"Centro de Ciências Exatas e da Terra",
    "IMD":"Instituto Metrópole Digital",
    "ECT":"Escola de Ciências e Tecnologia",
    "DEE":"Departamento de Engenharia Elétrica",
    "PPGEEC":"Prog. PG Eng. Elétrica e Computação",
    "PPgEEC":"Prog. PG Eng. Elétrica e Computação",
    "PPGSC":"Prog. PG Sistemas e Computação",
    "PPGCC":"Prog. PG Ciência da Computação",
}

CURSOS = {
    "Engenharia de Computação":"CURSO",
    "Ciência da Computação":"CURSO",
    "Sistemas de Informação":"CURSO",
    "Engenharia Elétrica":"CURSO",
    "Engenharia de Software":"CURSO",
    "Tecnologia da Informação":"CURSO",
    "Redes de Computadores":"CURSO",
    "Análise e Desenvolvimento de Sistemas":"CURSO",
    "Engenharia de Telecomunicações":"CURSO",
    "Engenharia Mecatrônica":"CURSO",
}

PREFIXOS_RX = (
    r"(?:Prof\.?\s*(?:Dr\.?|Dra?\.?|Me\.?|MSc\.?)?\s*"
    r"|Profa?\.?\s*(?:Dra?\.?|Me\.?|MSc?\.?)?\s*"
    r"|Dr\.?\s*|Dra?\.?\s*|Me\.?\s*|MSc\.?\s*|Esp\.?\s*)"
)

SUBSTITUICOES = [
    (r"Ÿ","ê"),(r"ﬁ","fi"),(r"ﬂ","fl"),(r"ﬀ","ff"),
    (r"ﬃ","ffi"),(r"ﬄ","ffl"),(r"\u00A0"," "),(r"\u2009"," "),
    (r"[\u2013\u2014]","-"),(r"[\u201C\u201D\u201E]",'"'),
    (r"[\u2018\u2019]","'"),
]

ABREVIACOES = re.compile(
    r"\b(Prof|Profa?|Dr|Dra?|Sr|Sra|Eng|Adv|et al|Apud|Op\.?\s*cit|Ibid"
    r"|fig|tab|eq|cap|vol|num|núm|pág|pp"
    r"|jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez|[A-Z])\.",
    re.IGNORECASE,
)

SIGLAS_TECNICAS = re.compile(
    r"^(LSTM|CNN|RNN|GAN|NLP|NER|API|GPU|CPU|RAM"
    r"|UFRN|UFPB|UFC|USP|UFMG|UFSC|UFAL|UFPE|UFERSA|IFRN"
    r"|DCA|DIMAp|CT|IMD|ECT|CCET|DEE|PPGEEC|PPgEEC|PPGSC|PPGCC"
    r"|HTTP|REST|SQL|NoSQL|JSON|XML|CSV|PDF"
    r"|BFS|DFS|KNN|SVM|MLP|DBN|AE|DAE|SAE"
    r"|IoT|ML|DL|AI|LLM|RAG|PLN|IEEE|ACM|ABNT|NBR)$"
)

RUIDO_ESTRUTURAL = re.compile(
    r"\b(lista de (ilustra[çc][õo]es?|figuras?|tabelas?|abreviaturas?|siglas?)"
    r"|(fundamenta[çc][aã]o|referencial)\s+te[oó]rica?"
    r"|cap[íi]tulo\s+(conclus[aã]o|experimentos?|resultados?|introdu[çc][aã]o)"
    r"|sumário\s+sumário|resultados?\s+figura|implementa[çc][aã]o\s+figura"
    r"|p[aá]gina\s+na\s+internet|this\s+work\s+presents?|moreover"
    r"|furthermore|nevertheless)\b",
    re.IGNORECASE,
)

ENTIDADE_ARTEFATO = re.compile(
    r"^(all\s+experiments?\s+were|american\s+standard\s+code"
    r"|this\s+(work|study|paper)\s+(presents?|proposes?|describes?)"
    r"|including\s+data\s+load|enables?\s+the\s+integrat"
    r"|to\s+address\s+the|rich\s+framework\s+of"
    r"|license\s+c[bc]|abstract\s+this\s+study|source\s*:\s*author)",
    re.IGNORECASE,
)

STOPWORDS_EN = {
    "the","this","using","were","with","that","from","into","and",
    "for","are","has","have","been","its","their","these","those",
    "which","while","although","despite","must","even","along","where",
}

MAX_TOKENS_ENTIDADE = 4

_LABELS_CONFIAVEIS = {
    "TEC","AREA","FERRAMENTA","HARDWARE","CONCEITO",
    "INST","DEPT","CURSO","ORIENTADOR",
}

# ══════════════════════════════════════════════════════════════════
# 2. UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════

def _nfd(texto: str) -> str:
    """NFD sem diacríticos, lowercase — para comparação robusta."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    ).lower()

# ══════════════════════════════════════════════════════════════════
# 3. SINGLETON SPACY
# ══════════════════════════════════════════════════════════════════
_nlp = None

def _carregar_modelo():
    global _nlp
    if _nlp is not None:
        return _nlp
    for modelo in ("pt_core_news_lg", "pt_core_news_sm"):
        try:
            _nlp = spacy.load(modelo)
            _nlp.max_length = 500_000
            print(f"  Modelo spaCy: {modelo}")
            return _nlp
        except OSError:
            continue
    raise RuntimeError("Execute: python -m spacy download pt_core_news_sm")

# ══════════════════════════════════════════════════════════════════
# 4. NER MANUAL
# ══════════════════════════════════════════════════════════════════

def aplicar_ner_manual(doc: Doc) -> Doc:
    """
    Adiciona entidades dos dicionários manuais ao doc spaCy.
    Labels: TEC | AREA | FERRAMENTA | HARDWARE | CONCEITO
            INST | DEPT | CURSO | ORIENTADOR
    """
    novas = []
    texto_lower = doc.text.lower()
    texto_nfd   = _nfd(doc.text)

    # termos técnicos
    for termo, label in TERMOS_ENG_COMP.items():
        for m in re.finditer(rf"\b{re.escape(termo.lower())}\b", texto_lower):
            sp = doc.char_span(m.start(), m.end(), label=label)
            if sp: novas.append(sp)

    # universidades — sigla
    for sigla in UNIVERSIDADES:
        for m in re.finditer(rf"\b{re.escape(sigla)}\b", doc.text):
            sp = doc.char_span(m.start(), m.end(), label="INST")
            if sp: novas.append(sp)

    # universidades — nome longo (NFD)
    for nome_nfd in UNIVERSIDADES_NOMES_LONGOS:
        for m in re.finditer(re.escape(nome_nfd), texto_nfd):
            sp = doc.char_span(m.start(), m.end(), label="INST")
            if sp: novas.append(sp)

    # departamentos
    for sigla in DEPARTAMENTOS:
        for m in re.finditer(rf"\b{re.escape(sigla)}\b", doc.text):
            sp = doc.char_span(m.start(), m.end(), label="DEPT")
            if sp: novas.append(sp)

    # cursos (NFD)
    for nome_curso in CURSOS:
        for m in re.finditer(re.escape(_nfd(nome_curso)), texto_nfd):
            sp = doc.char_span(m.start(), m.end(), label="CURSO")
            if sp: novas.append(sp)

    # professores DCA — busca NFD (case/acento insensitive)
    for prof_nfd in PROFESSORES_DCA:
        for m in re.finditer(re.escape(prof_nfd), texto_nfd):
            sp = doc.char_span(m.start(), m.end(), label="ORIENTADOR")
            if sp: novas.append(sp)

    doc.ents = filter_spans(list(doc.ents) + novas)
    return doc

# ══════════════════════════════════════════════════════════════════
# 5. FILTRO PÓS-NER
# ══════════════════════════════════════════════════════════════════

def filtrar_entidade(texto_ent: str, label: str) -> bool:
    """True → mantém | False → descarta."""
    t = texto_ent.strip()
    if label in _LABELS_CONFIAVEIS:
        return len(t) > 1
    if len(t) <= 2 or t.isdigit():
        return False
    if len(t.split()) > MAX_TOKENS_ENTIDADE:
        return False
    if ENTIDADE_ARTEFATO.match(t):
        return False
    if label == "MISC" and set(t.lower().split()) & STOPWORDS_EN:
        return False
    if RUIDO_ESTRUTURAL.search(t):
        return False
    return True

# ══════════════════════════════════════════════════════════════════
# 6. NORMALIZAÇÃO DE NOMES
# ══════════════════════════════════════════════════════════════════

def normalizar_nome_pessoa(nome: str) -> str:
    """
    Remove prefixos acadêmicos → Title Case.
    Ex: 'Prof. Dr. João da Silva' → 'João Da Silva'
    Retorna Title Case (não lowercase) para legibilidade nos grafos.
    """
    nome = re.sub(r"^" + PREFIXOS_RX, "", nome, flags=re.IGNORECASE).strip()
    nome = re.sub(r"^[.,;:\-]+", "", nome).strip()
    return " ".join(p.capitalize() for p in nome.split())

# ══════════════════════════════════════════════════════════════════
# 7. PALAVRAS-CHAVE DO TCC
# ══════════════════════════════════════════════════════════════════

def extrair_palavras_chave(doc) -> list:
    """
    Extrai palavras-chave do TCC a partir do NER.
    Prioridade: AREA > TEC > FERRAMENTA > CONCEITO.
    Deduplicado e ordenado por frequência decrescente.
    """
    contagem: dict = {}
    labels_kw = {"AREA", "TEC", "FERRAMENTA", "CONCEITO"}
    for ent in doc.ents:
        if ent.label_ in labels_kw and filtrar_entidade(ent.text, ent.label_):
            chave = _nfd(ent.text.strip())
            contagem[chave] = contagem.get(chave, 0) + 1
    return [
        k.title()
        for k, _ in sorted(contagem.items(), key=lambda x: x[1], reverse=True)
    ]

# ══════════════════════════════════════════════════════════════════
# 8. EXTRAÇÃO DE METADADOS
# ══════════════════════════════════════════════════════════════════

def extrair_metadados(texto_bruto: str, nome_arquivo: str = "") -> dict:
    """
    Extrai metadados da capa/preâmbulo (primeiros 6000 chars).
    Roda ANTES do preprocessar() para não perder a capa.
    """
    cab     = texto_bruto[:6000]
    cab_nfd = _nfd(cab)

    # Título
    titulo = None
    m = re.search(
        r"([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][^\n]{10,120})\n[^\n]*"
        r"(Trabalho de Conclus|Monografia|Disserta|Tese)",
        cab, re.IGNORECASE | re.DOTALL,
    )
    if m: titulo = m.group(1).strip()

    # Autor
    autor = None
    m = re.search(
        r"(?:Autor[a]?|Aluno[a]?)\s*:?\s*\n?\s*" + PREFIXOS_RX + r"?"
        r"([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+"
        r"(?:\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+){1,4})",
        cab, re.IGNORECASE,
    )
    if m: autor = normalizar_nome_pessoa(m.group(1).strip())
    if not autor:
        for linha in cab.split("\n")[:30]:
            linha = linha.strip()
            if re.match(
                r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+"
                r"(\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+){1,4}$", linha,
            ) and "UNIVERSIDADE" not in linha.upper():
                c = normalizar_nome_pessoa(linha)
                if len(c.split()) >= 2:
                    autor = c; break

    # Orientador
    orientador = None
    m = re.search(
        r"(?:Orientador[a]?|Advisor)\s*:?\s*\n?\s*" + PREFIXOS_RX + r"?\s*"
        r"([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+"
        r"(?:\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+){1,4})",
        cab, re.IGNORECASE,
    )
    if m: orientador = normalizar_nome_pessoa(m.group(1).strip())

    # Coorientador
    coorientador = None
    m = re.search(
        r"(?:Co-?[Oo]rientador[a]?)\s*:?\s*\n?\s*" + PREFIXOS_RX + r"?\s*"
        r"([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+"
        r"(?:\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+){1,4})",
        cab, re.IGNORECASE,
    )
    if m: coorientador = normalizar_nome_pessoa(m.group(1).strip())

    # Universidade
    universidade, sigla_univ = None, None
    for nome_nfd, sigla in UNIVERSIDADES_NOMES_LONGOS.items():
        if nome_nfd in cab_nfd:
            sigla_univ = sigla; universidade = UNIVERSIDADES[sigla]; break
    if not universidade:
        for sigla, nome in UNIVERSIDADES.items():
            if re.search(rf"\b{sigla}\b", cab):
                sigla_univ, universidade = sigla, nome; break

    # Departamento
    departamento = None
    for sigla, nome in DEPARTAMENTOS.items():
        if re.search(rf"\b{re.escape(sigla)}\b", cab):
            departamento = nome; break
    if not departamento:
        m = re.search(
            r"((?:Departamento|Centro|Instituto|Escola|Programa)\s+de\s+"
            r"[A-ZÁÉÍÓÚ][^\n]{5,80})", cab, re.IGNORECASE,
        )
        if m: departamento = m.group(1).strip()

    # Curso
    curso = None
    for nome_curso in CURSOS:
        if _nfd(nome_curso) in cab_nfd:
            curso = nome_curso; break

    # Ano
    anos = re.findall(r"\b(20\d{2}|19\d{2})\b", cab)
    ano  = max(set(anos), key=anos.count) if anos else None

    meta = {
        "arquivo":      nome_arquivo,
        "titulo":       titulo or nome_arquivo.replace(".txt",""),
        "autor":        autor,
        "orientador":   orientador,
        "coorientador": coorientador,
        "universidade": universidade,
        "sigla_univ":   sigla_univ or "UFRN",
        "departamento": departamento,
        "curso":        curso,
        "ano":          ano,
    }
    print("\n  ── Metadados extraídos ──")
    for k, v in meta.items():
        if v: print(f"    {k:<14}: {v}")
    return meta

# ══════════════════════════════════════════════════════════════════
# 9. DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════

def diagnosticar(texto: str, n: int = 20) -> None:
    letras_pt = set("áéíóúàâêôãõçüÁÉÍÓÚÀÂÊÔÃÕÇÜ")
    suspeitos = [c for c in texto if ord(c) > 127 and c not in letras_pt]
    contagem  = Counter(suspeitos).most_common(n)
    if contagem:
        print("  ── Chars suspeitos ──")
        for char, freq in contagem:
            print(f"    U+{ord(char):04X}  '{char}'  →  {freq}×")

# ══════════════════════════════════════════════════════════════════
# 10. PIPELINE DE LIMPEZA
# ══════════════════════════════════════════════════════════════════

def ler_arquivo(caminho: str) -> str:
    with open(caminho, encoding="utf-8", errors="replace") as f:
        return f.read()

def normalizar_caracteres(texto: str) -> str:
    for padrao, substituto in SUBSTITUICOES:
        texto = re.sub(padrao, substituto, texto)
    return texto

def recortar_corpo(texto: str) -> str:
    texto = re.sub(
        r"LISTA\s+DE\s+(ABREVIATURAS?|SIGLAS?|SÍMBOLOS?).+?(?=RESUMO|ABSTRACT)",
        "", texto, flags=re.DOTALL | re.IGNORECASE,
    )
    inicio = re.search(r"\b(RESUMO|ABSTRACT)\b", texto)
    fim    = re.search(r"\b(REFER[ÊEĹ]NCIAS?|BIBLIOGRAPHY|BIBLIOGRAF[ÍI]A)\b", texto)
    if inicio and fim and inicio.start() < fim.start():
        return texto[inicio.start(): fim.start()]
    return texto

def remover_ruidos(texto: str) -> str:
    texto = re.sub(r"^\d{1,4}\s*$", "", texto, flags=re.MULTILINE)
    texto = re.sub(
        r"^(Figura|Tabela|Gráfico|Quadro|Esquema|Imagem)\s+\d+[\s\-–—].*$",
        "", texto, flags=re.MULTILINE | re.IGNORECASE,
    )
    texto = re.sub(r"^Fonte:.*$", "", texto, flags=re.MULTILINE | re.IGNORECASE)
    texto = RUIDO_ESTRUTURAL.sub("", texto)
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto, flags=re.MULTILINE)
    def _remover_se_nao_sigla(m):
        linha = m.group(0).strip()
        return m.group(0) if SIGLAS_TECNICAS.match(linha) else ""
    texto = re.sub(
        r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ\s]{2,60}$",
        _remover_se_nao_sigla, texto, flags=re.MULTILINE,
    )
    texto = re.sub(r"^\s*\S\s*$", "", texto, flags=re.MULTILINE)
    return texto

def restaurar_paragrafos(texto: str) -> str:
    MARCA = "\x00"
    texto = ABREVIACOES.sub(lambda m: m.group(0).replace(".", MARCA), texto)
    texto = re.sub(r"([.!?])\s{2,}([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ])", r"\1\n\n\2", texto)
    texto = texto.replace(MARCA, ".")
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()

def limpeza_final(texto: str) -> str:
    texto = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", texto)
    texto = re.sub(r"[\uE000-\uF8FF]+", "", texto)
    texto = "".join(
        c for c in texto
        if unicodedata.category(c)[0] not in ("C","Z") or c in ("\n","\t"," ")
    )
    paragrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    return "\n\n".join(paragrafos)

def preprocessar(caminho: str, diagnostico: bool = False) -> str:
    texto = ler_arquivo(caminho)
    texto = normalizar_caracteres(texto)
    texto = recortar_corpo(texto)
    texto = remover_ruidos(texto)
    texto = restaurar_paragrafos(texto)
    texto = limpeza_final(texto)
    if diagnostico: diagnosticar(texto)
    return texto

# ══════════════════════════════════════════════════════════════════
# 11. PROCESSAMENTO SPACY
# ══════════════════════════════════════════════════════════════════

def processar_spacy(texto: str):
    nlp = _carregar_modelo()
    if len(texto) >= nlp.max_length:
        nlp.max_length = len(texto) + 10_000
    doc = next(nlp.pipe([texto], batch_size=1))
    doc = aplicar_ner_manual(doc)
    total       = len(doc.ents)
    apos_filtro = [e for e in doc.ents if filtrar_entidade(e.text, e.label_)]
    print(f"\n  Sentenças : {len(list(doc.sents))}")
    print(f"  Entidades : {total} brutas → {len(apos_filtro)} após filtro")
    por_label: dict = {}
    for e in apos_filtro: por_label.setdefault(e.label_, []).append(e.text)
    for label, ents in sorted(por_label.items()):
        print(f"    {label:<12} ({len(ents):>3}): {', '.join(ents[:5])}")
    return doc