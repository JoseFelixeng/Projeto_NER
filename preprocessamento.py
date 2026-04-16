"""
preprocessamento.py
Pipeline de pré-processamento de texto extraído de PDF.
Inclui limpeza, NER manual (termos técnicos + acadêmicos),
filtros de qualidade e extração de metadados estruturados.

MELHORIAS v2:
- NER manual expandido com entidades acadêmicas (PESSOA_ACAD, INST, DEPT, CURSO)
- extrair_metadados() reescrito: captura autor, orientador, coorientador,
  universidade, departamento, curso, título e ano com regex robustos
- normalizar_entidade_pessoa() para padronizar nomes (evita duplicatas no grafo)
- novo label ORIENTADOR e AUTOR para uso direto no grafo relacional
"""

import re
import unicodedata
import spacy
from collections import Counter
from spacy.tokens import Doc
from spacy.util import filter_spans


# ─────────────────────────────────────────────
# NER MANUAL — TECNOLOGIAS (mantido do original)
# ─────────────────────────────────────────────
TERMOS_ENG_COMP = {
    # TECNOLOGIAS / MODELOS / ALGORITMOS (TEC)
    "LSTM": "TEC", "CNN": "TEC", "RNN": "TEC", "GAN": "TEC",
    "Transformer": "TEC", "BERT": "TEC", "GPT": "TEC", "LLaMA": "TEC",
    "YOLO": "TEC", "ResNet": "TEC", "VGG": "TEC", "AlexNet": "TEC",
    "U-Net": "TEC", "SVM": "TEC", "KNN": "TEC", "Random Forest": "TEC",
    "XGBoost": "TEC", "LightGBM": "TEC", "CatBoost": "TEC", "K-Means": "TEC",
    "DBSCAN": "TEC", "PCA": "TEC", "t-SNE": "TEC", "Autoencoder": "TEC",
    "VAE": "TEC", "DBN": "TEC", "MLP": "TEC", "RBM": "TEC",
    "Seq2Seq": "TEC", "Attention": "TEC", "Self-Attention": "TEC",
    "Graph Neural Network": "TEC", "GCN": "TEC", "GAT": "TEC",
    "RAG": "TEC", "Reinforcement Learning": "TEC", "Federated Learning": "TEC",
    "Transfer Learning": "TEC", "Multimodal Learning": "TEC",

    # SUBÁREAS (AREA)
    "Machine Learning": "AREA", "Deep Learning": "AREA", "Redes Neurais": "AREA",
    "Visão Computacional": "AREA", "Processamento de Linguagem Natural": "AREA",
    "PLN": "AREA", "NLP": "AREA", "Sistemas Embarcados": "AREA",
    "Internet das Coisas": "AREA", "IoT": "AREA", "Robótica": "AREA",
    "Inteligência Artificial": "AREA", "IA": "AREA", "Big Data": "AREA",
    "Data Science": "AREA", "Engenharia de Software": "AREA",
    "Computação em Nuvem": "AREA", "Edge Computing": "AREA",
    "Segurança da Informação": "AREA", "Blockchain": "AREA",
    "Teoria dos Grafos": "AREA", "Redes Complexas": "AREA",
    "Processamento de Sinais": "AREA", "Sistemas Distribuídos": "AREA",
    "Banco de Dados": "AREA", "Computação Gráfica": "AREA",

    # FERRAMENTAS (FERRAMENTA)
    "TensorFlow": "FERRAMENTA", "PyTorch": "FERRAMENTA", "Keras": "FERRAMENTA",
    "Scikit-learn": "FERRAMENTA", "Pandas": "FERRAMENTA", "NumPy": "FERRAMENTA",
    "OpenCV": "FERRAMENTA", "spaCy": "FERRAMENTA", "Hugging Face": "FERRAMENTA",
    "Transformers": "FERRAMENTA", "LangChain": "FERRAMENTA", "FastAPI": "FERRAMENTA",
    "Docker": "FERRAMENTA", "Streamlit": "FERRAMENTA", "Git": "FERRAMENTA",
    "GitHub": "FERRAMENTA", "Jupyter": "FERRAMENTA", "Colab": "FERRAMENTA",
    "Elasticsearch": "FERRAMENTA", "Kibana": "FERRAMENTA",
    "NetworkX": "FERRAMENTA", "Gephi": "FERRAMENTA",
    "nxviz": "FERRAMENTA", "Pyvis": "FERRAMENTA", "Matplotlib": "FERRAMENTA",

    # HARDWARE
    "GPU": "HARDWARE", "CPU": "HARDWARE", "TPU": "HARDWARE", "FPGA": "HARDWARE",
    "RAM": "HARDWARE", "Arduino": "HARDWARE", "Raspberry Pi": "HARDWARE",
    "ESP32": "HARDWARE", "NVIDIA Jetson": "HARDWARE",

    # MÉTRICAS / CONCEITOS
    "Acurácia": "CONCEITO", "Precisão": "CONCEITO", "Recall": "CONCEITO",
    "F1-score": "CONCEITO", "Overfitting": "CONCEITO", "Underfitting": "CONCEITO",
    "Dropout": "CONCEITO", "Batch Normalization": "CONCEITO",
    "Gradient Descent": "CONCEITO", "Adam": "CONCEITO", "SGD": "CONCEITO",
    "Embedding": "CONCEITO", "Word2Vec": "CONCEITO", "GloVe": "CONCEITO",
    "Coeficiente de Agrupamento": "CONCEITO", "Centralidade": "CONCEITO",
    "Betweenness": "CONCEITO", "Pagerank": "CONCEITO", "Clustering": "CONCEITO",
    "Densidade": "CONCEITO", "Diâmetro": "CONCEITO", "Componentes Conectados": "CONCEITO",
    "Named Entity Recognition": "CONCEITO", "NER": "CONCEITO",
    "co-ocorrência": "CONCEITO", "Co-Ocorrência": "CONCEITO",
}

# ─────────────────────────────────────────────
# NER MANUAL — ENTIDADES ACADÊMICAS  ← NOVO
# ─────────────────────────────────────────────

# Universidades brasileiras (sigla → nome canônico)
UNIVERSIDADES = {
    # Nordeste
    "UFRN":    "Universidade Federal do Rio Grande do Norte",
    "UFPB":    "Universidade Federal da Paraíba",
    "UFC":     "Universidade Federal do Ceará",
    "UFPE":    "Universidade Federal de Pernambuco",
    "UFAL":    "Universidade Federal de Alagoas",
    "UFBA":    "Universidade Federal da Bahia",
    "UFERSA":  "Universidade Federal Rural do Semi-Árido",
    "IFRN":    "Instituto Federal do Rio Grande do Norte",
    # Demais regiões
    "USP":     "Universidade de São Paulo",
    "UNICAMP": "Universidade Estadual de Campinas",
    "UNESP":   "Universidade Estadual Paulista",
    "UFMG":    "Universidade Federal de Minas Gerais",
    "UFSC":    "Universidade Federal de Santa Catarina",
    "UFRJ":    "Universidade Federal do Rio de Janeiro",
    "UFPR":    "Universidade Federal do Paraná",
    "UnB":     "Universidade de Brasília",
    "UFAM":    "Universidade Federal do Amazonas",
    "UFPA":    "Universidade Federal do Pará",
    "UFES":    "Universidade Federal do Espírito Santo",
    "UFG":     "Universidade Federal de Goiás",
    "UFSCAR":  "Universidade Federal de São Carlos",
    "FIOCRUZ": "Fundação Oswaldo Cruz",
}

# Expansão de nomes longos das mesmas universidades
UNIVERSIDADES_NOMES_LONGOS = {
    "universidade federal do rio grande do norte": "UFRN",
    "universidade federal da paraíba":             "UFPB",
    "universidade federal do ceará":               "UFC",
    "universidade federal de pernambuco":          "UFPE",
    "universidade federal de alagoas":             "UFAL",
    "universidade federal da bahia":               "UFBA",
    "universidade federal rural do semi-árido":    "UFERSA",
    "instituto federal do rio grande do norte":    "IFRN",
    "universidade de são paulo":                   "USP",
    "universidade estadual de campinas":           "UNICAMP",
    "universidade federal de minas gerais":        "UFMG",
    "universidade federal de santa catarina":      "UFSC",
    "universidade federal do rio de janeiro":      "UFRJ",
    "universidade federal do paraná":              "UFPR",
    "universidade de brasília":                    "UnB",
}

# Departamentos e centros comuns
DEPARTAMENTOS = {
    "DCA":    "Departamento de Computação e Automação",
    "DIMAp":  "Departamento de Informática e Matemática Aplicada",
    "CT":     "Centro de Tecnologia",
    "CCET":   "Centro de Ciências Exatas e da Terra",
    "IMD":    "Instituto Metrópole Digital",
    "ECT":    "Escola de Ciências e Tecnologia",
    "DEE":    "Departamento de Engenharia Elétrica",
    "DEN":    "Departamento de Engenharia",
    "PPGEEC": "Programa de Pós-Graduação em Engenharia Elétrica e de Computação",
    "PPgEEC": "Programa de Pós-Graduação em Engenharia Elétrica e de Computação",
    "PPGSC":  "Programa de Pós-Graduação em Sistemas e Computação",
    "PPGCC":  "Programa de Pós-Graduação em Ciência da Computação",
}

# Cursos de graduação / pós
CURSOS = {
    "Engenharia de Computação":        "CURSO",
    "Ciência da Computação":           "CURSO",
    "Sistemas de Informação":          "CURSO",
    "Engenharia Elétrica":             "CURSO",
    "Engenharia de Software":          "CURSO",
    "Tecnologia da Informação":        "CURSO",
    "Redes de Computadores":           "CURSO",
    "Análise e Desenvolvimento de Sistemas": "CURSO",
    "Engenharia de Telecomunicações":  "CURSO",
    "Engenharia Mecatrônica":          "CURSO",
}

# Prefixos de tratamento acadêmico (usados na regex de orientador/banca)
PREFIXOS_ACADEMICOS = (
    r"Prof\.?\s*(?:Dr\.?|Me\.?|MSc\.?|Esp\.?)?"
    r"|Profa?\.?\s*(?:Dra?\.?|Me\.?|MSc\.?)?"
    r"|Dr\.?\s*|Dra?\.?\s*"
    r"|Me\.?\s*|MSc\.?\s*"
    r"|Esp\.?\s*"
)


# ─────────────────────────────────────────────
# CONSTANTES DE LIMPEZA (mantidas do original)
# ─────────────────────────────────────────────
SUBSTITUICOES = [
    (r"Ÿ", "ê"), (r"ﬁ", "fi"), (r"ﬂ", "fl"), (r"ﬀ", "ff"),
    (r"ﬃ", "ffi"), (r"ﬄ", "ffl"), (r"\u00A0", " "), (r"\u2009", " "),
    (r"[\u2013\u2014]", "-"), (r"[\u201C\u201D\u201E]", '"'),
    (r"[\u2018\u2019]", "'"),
]

ABREVIACOES = re.compile(
    r"\b(Prof|Profa?|Dr|Dra?|Sr|Sra|Eng|Adv|et al|Apud|Op\.?\s*cit|Ibid"
    r"|fig|tab|eq|cap|vol|num|núm|pág|pp"
    r"|jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez"
    r"|[A-Z])\.",
    re.IGNORECASE,
)

SIGLAS_TECNICAS = re.compile(
    r"^(LSTM|CNN|RNN|GAN|NLP|NER|API|GPU|CPU|RAM"
    r"|UFRN|UFPB|UFC|USP|UFMG|UFSC|UFAL|UFPE|UFERSA|IFRN"
    r"|DCA|DIMAp|CT|IMD|ECT|CCET"
    r"|HTTP|REST|SQL|NoSQL|JSON|XML|CSV|PDF"
    r"|BFS|DFS|KNN|SVM|MLP|DBN|AE|DAE|SAE"
    r"|IoT|ML|DL|AI|LLM|RAG|PLN|NLP"
    r"|IEEE|ACM|ABNT|NBR)$"
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
    texto = re.sub(r"^\d{1,4}\s*$", "", texto, flags=re.MULTILINE)
    texto = re.sub(
        r"^(Figura|Tabela|Gráfico|Quadro|Esquema|Imagem)\s+\d+[\s\-–—].*$",
        "", texto, flags=re.MULTILINE | re.IGNORECASE,
    )
    texto = re.sub(r"^Fonte:.*$", "", texto, flags=re.MULTILINE | re.IGNORECASE)

    def _remover_se_nao_sigla(m):
        linha = m.group(0).strip()
        return m.group(0) if SIGLAS_TECNICAS.match(linha) else ""

    texto = re.sub(
        r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ\s]{2,60}$",
        _remover_se_nao_sigla, texto, flags=re.MULTILINE,
    )
    texto = re.sub(r"^\s*\S\s*$", "", texto, flags=re.MULTILINE)
    return texto


# ─────────────────────────────────────────────
# ETAPA 5 — Restauração de parágrafos
# ─────────────────────────────────────────────
def restaurar_paragrafos(texto: str) -> str:
    MARCA = "\x00"
    texto = ABREVIACOES.sub(lambda m: m.group(0).replace(".", MARCA), texto)
    texto = re.sub(r"([.!?])\s{2,}([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ])", r"\1\n\n\2", texto)
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
    paragrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    return "\n\n".join(paragrafos)


# ─────────────────────────────────────────────
# DIAGNÓSTICO
# ─────────────────────────────────────────────
def diagnosticar(texto: str, n: int = 20) -> None:
    letras_pt = set("áéíóúàâêôãõçüÁÉÍÓÚÀÂÊÔÃÕÇÜ")
    suspeitos = [c for c in texto if ord(c) > 127 and c not in letras_pt]
    contagem  = Counter(suspeitos).most_common(n)
    print("─── Chars suspeitos restantes ───")
    for char, freq in contagem:
        print(f"  U+{ord(char):04X}  '{char}'  →  {freq}×")


# ─────────────────────────────────────────────
# UTILITÁRIO — normalizar nome de pessoa
# ─────────────────────────────────────────────
def normalizar_nome_pessoa(nome: str) -> str:
    """
    Remove prefixos acadêmicos e normaliza capitalização.
    Ex.: 'Prof. Dr. João Silva' → 'João Silva'
         'JOÃO SILVA' → 'João Silva'
    """
    # remove prefixos
    nome = re.sub(
        r"^(Prof\.?\s*(?:Dr\.?|Dra?\.?|Me\.?|MSc\.?)?\s*"
        r"|Profa?\.?\s*(?:Dra?\.?|Me\.?|MSc\.?)?\s*"
        r"|Dr\.?\s*|Dra?\.?\s*|Me\.?\s*|MSc\.?\s*|Esp\.?\s*)",
        "", nome, flags=re.IGNORECASE,
    ).strip()

    # Título Case
    partes = nome.split()
    conectivos = {"de", "da", "do", "dos", "das", "e", "ou"}
    nome = " ".join(
        p.capitalize() if p.lower() not in conectivos else p.lower()
        for p in partes
    )
    return nome


# ─────────────────────────────────────────────
# EXTRAÇÃO DE METADADOS — REESCRITO  ← NOVO
# ─────────────────────────────────────────────
def extrair_metadados(texto_bruto: str, nome_arquivo: str = "") -> dict:
    """
    Extrai metadados estruturados da capa/preâmbulo do TCC.

    Retorna dict com:
      arquivo, titulo, autor, orientador, coorientador,
      universidade, sigla_univ, departamento, curso, ano
    """
    cabecalho = texto_bruto[:6000]   # capa + folha de rosto cabem aqui

    # ── Título ──────────────────────────────────────────────────────────────
    titulo = None
    # Padrão: linha(s) ANTES da linha "Trabalho de Conclusão" ou "Monografia"
    m = re.search(
        r"([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][^\n]{10,120})\n[^\n]*"
        r"(Trabalho de Conclus|Monografia|Disserta|Tese)",
        cabecalho, re.IGNORECASE | re.DOTALL,
    )
    if m:
        titulo = m.group(1).strip()

    # ── Autor ────────────────────────────────────────────────────────────────
    autor = None
    # Busca linha após "Autor:" / "Aluno:"
    m = re.search(
        r"(?:Autor[a]?|Aluno[a]?)\s*:?\s*\n?\s*"
        r"(" + PREFIXOS_ACADEMICOS + r")?([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+){1,4})",
        cabecalho, re.IGNORECASE,
    )
    if m:
        autor = normalizar_nome_pessoa(m.group(0).split(":")[-1].strip())

    # Se não encontrou via label, tenta heurística: nome próprio na capa
    if not autor:
        linhas = cabecalho.split("\n")
        for linha in linhas[:30]:
            linha = linha.strip()
            # Nome próprio simples (2-5 palavras, sem números, sem caixa alta total)
            if re.match(
                r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+"
                r"(\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+){1,4}$",
                linha,
            ) and "UNIVERSIDADE" not in linha.upper():
                candidato = normalizar_nome_pessoa(linha)
                if len(candidato.split()) >= 2:
                    autor = candidato
                    break

    # ── Orientador ───────────────────────────────────────────────────────────
    orientador = None
    m = re.search(
        r"(?:Orientador[a]?|Advisor)\s*:?\s*\n?\s*"
        r"(?:" + PREFIXOS_ACADEMICOS + r")?\s*"
        r"([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+){1,4})",
        cabecalho, re.IGNORECASE,
    )
    if m:
        orientador = normalizar_nome_pessoa(m.group(1).strip())

    # ── Coorientador ─────────────────────────────────────────────────────────
    coorientador = None
    m = re.search(
        r"(?:Co-?[Oo]rientador[a]?|Co-?[Aa]dvisor)\s*:?\s*\n?\s*"
        r"(?:" + PREFIXOS_ACADEMICOS + r")?\s*"
        r"([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+){1,4})",
        cabecalho, re.IGNORECASE,
    )
    if m:
        coorientador = normalizar_nome_pessoa(m.group(1).strip())

    # ── Universidade ─────────────────────────────────────────────────────────
    universidade = None
    sigla_univ   = None

    # Tenta pelo nome longo primeiro
    for nome_longo, sigla in UNIVERSIDADES_NOMES_LONGOS.items():
        if nome_longo in cabecalho.lower():
            sigla_univ   = sigla
            universidade = UNIVERSIDADES[sigla]
            break

    # Tenta pela sigla se não achou pelo nome
    if not universidade:
        for sigla, nome in UNIVERSIDADES.items():
            if re.search(rf"\b{sigla}\b", cabecalho):
                sigla_univ   = sigla
                universidade = nome
                break

    # ── Departamento / Centro ────────────────────────────────────────────────
    departamento = None
    for sigla, nome in DEPARTAMENTOS.items():
        if re.search(rf"\b{re.escape(sigla)}\b", cabecalho):
            departamento = nome
            break
    # Também tenta pelo nome longo
    if not departamento:
        m = re.search(
            r"((?:Departamento|Centro|Instituto|Escola|Programa)\s+de\s+[A-ZÁÉÍÓÚ][^\n]{5,80})",
            cabecalho, re.IGNORECASE,
        )
        if m:
            departamento = m.group(1).strip()

    # ── Curso ────────────────────────────────────────────────────────────────
    curso = None
    for nome_curso in CURSOS:
        if nome_curso.lower() in cabecalho.lower():
            curso = nome_curso
            break
    if not curso:
        m = re.search(
            r"(?:Curso\s+de|Graduação\s+em|Bacharel(?:ado)?\s+em)\s+"
            r"([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç\s]{5,60})",
            cabecalho, re.IGNORECASE,
        )
        if m:
            curso = m.group(1).strip()

    # ── Ano ──────────────────────────────────────────────────────────────────
    ano = None
    anos = re.findall(r"\b(20\d{2}|19\d{2})\b", cabecalho)
    if anos:
        ano = max(set(anos), key=anos.count)

    meta = {
        "arquivo":      nome_arquivo,
        "titulo":       titulo,
        "autor":        autor,
        "orientador":   orientador,
        "coorientador": coorientador,
        "universidade": universidade,
        "sigla_univ":   sigla_univ,
        "departamento": departamento,
        "curso":        curso,
        "ano":          ano,
    }

    print("\n  ── Metadados extraídos ──")
    for k, v in meta.items():
        print(f"    {k:<14}: {v}")

    return meta


# ─────────────────────────────────────────────
# NER MANUAL — aplica dicionários ao doc spaCy
# ─────────────────────────────────────────────
def aplicar_ner_manual(doc: Doc) -> Doc:
    """
    Aplica TERMOS_ENG_COMP + entidades acadêmicas sem sobrescrever
    entidades existentes e sem sobreposição (usa filter_spans).
    """
    novas_ents = []
    texto_lower = doc.text.lower()

    # — termos técnicos —
    for termo, label in TERMOS_ENG_COMP.items():
        for match in re.finditer(
            rf"\b{re.escape(termo.lower())}\b", texto_lower
        ):
            sp = doc.char_span(match.start(), match.end(), label=label)
            if sp is not None:
                novas_ents.append(sp)

    # — siglas de universidades —
    for sigla in UNIVERSIDADES:
        for match in re.finditer(rf"\b{re.escape(sigla)}\b", doc.text):
            sp = doc.char_span(match.start(), match.end(), label="INST")
            if sp is not None:
                novas_ents.append(sp)

    # — nomes longos de universidades —
    for nome_longo, sigla in UNIVERSIDADES_NOMES_LONGOS.items():
        for match in re.finditer(
            re.escape(nome_longo), doc.text.lower()
        ):
            sp = doc.char_span(match.start(), match.end(), label="INST")
            if sp is not None:
                novas_ents.append(sp)

    # — siglas de departamentos —
    for sigla in DEPARTAMENTOS:
        for match in re.finditer(rf"\b{re.escape(sigla)}\b", doc.text):
            sp = doc.char_span(match.start(), match.end(), label="DEPT")
            if sp is not None:
                novas_ents.append(sp)

    # — nomes de cursos —
    for nome_curso in CURSOS:
        for match in re.finditer(
            re.escape(nome_curso.lower()), texto_lower
        ):
            sp = doc.char_span(match.start(), match.end(), label="CURSO")
            if sp is not None:
                novas_ents.append(sp)

    todas = list(doc.ents) + novas_ents
    doc.ents = filter_spans(todas)
    return doc


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


def processar_spacy(texto: str):
    """Carrega modelo português, processa texto e aplica NER manual."""
    MODELO = "pt_core_news_lg"
    nlp = spacy.load(MODELO)
    nlp.max_length = 2_000_000
    doc = nlp(texto)
    doc = aplicar_ner_manual(doc)

    print(f"\n─── Sentenças detectadas: {len(list(doc.sents))} ───")
    for i, sent in enumerate(doc.sents):
        if i >= 5:
            break
        print(f"  [{i+1}] {sent.text[:60]}")

    print(f"\n─── Entidades nomeadas (primeiras 15) ───")
    for ent in list(doc.ents)[:15]:
        print(f"  {ent.text:<35} → {ent.label_}")

    return doc