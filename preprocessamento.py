"""
preprocessamento.py — v5
Pipeline: limpeza → NER híbrido (spaCy + manual) → metadados.

Hierarquia do grafo:
  UNIV → DEPT → ORIENTADOR → AUTOR → TRABALHO → AREA → FERRAMENTA

v5 — fixes:
  - LOC manual expandido (Natal, RN, Nordeste, etc.)
  - Proteção de nomes de professores contra LOC adjacente
    (evita "Luiz Affonso Henderson Guedes De Oliveira Natal")
  - _separar_loc_de_nome() como etapa pós-NER
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

# ── LOC manual ────────────────────────────────────────────────────
# Cidades, estados e regiões que aparecem em TCCs da UFRN.
# Separados por granularidade para facilitar manutenção.
LOCS_CIDADE = {
    # RN
    "Natal", "Mossoró", "Caicó", "Parnamirim", "São Gonçalo do Amarante",
    "Currais Novos", "Pau dos Ferros", "Açu", "Macau",
    # Outras capitais nordestinas
    "Fortaleza", "Recife", "João Pessoa", "Maceió", "Salvador",
    "Teresina", "São Luís", "Aracaju",
    # Outras capitais brasileiras
    "Brasília", "São Paulo", "Rio de Janeiro", "Belo Horizonte",
    "Porto Alegre", "Curitiba", "Manaus", "Belém", "Goiânia",
    "Florianópolis", "Vitória", "Campo Grande", "Cuiabá",
    "Porto Velho", "Macapá", "Boa Vista", "Rio Branco", "Palmas",
}

LOCS_ESTADO = {
    # Por extenso
    "Rio Grande do Norte", "Paraíba", "Ceará", "Pernambuco",
    "Alagoas", "Bahia", "Sergipe", "Piauí", "Maranhão",
    "São Paulo", "Minas Gerais", "Rio de Janeiro", "Paraná",
    "Rio Grande do Sul", "Santa Catarina", "Goiás", "Mato Grosso",
    "Mato Grosso do Sul", "Espírito Santo", "Pará", "Amazonas",
    "Roraima", "Amapá", "Tocantins", "Rondônia", "Acre",
    # Siglas
    "RN", "PB", "CE", "PE", "AL", "BA", "SE", "PI", "MA",
    "SP", "MG", "RJ", "PR", "RS", "SC", "GO", "MT", "MS",
    "ES", "PA", "AM", "RR", "AP", "TO", "RO", "AC", "DF",
}

LOCS_REGIAO = {
    "Nordeste", "Norte", "Sudeste", "Sul", "Centro-Oeste",
    "Semiárido", "Semi-árido", "Agreste", "Sertão", "Litoral",
    "Brasil", "América do Sul", "América Latina",
}

# União de todos os LOC manuais (para busca)
LOCS_MANUAIS: set = LOCS_CIDADE | LOCS_ESTADO | LOCS_REGIAO

# NFD dos LOCs para busca case/acento-insensitive
LOCS_MANUAIS_NFD: dict = {}   # nfd(loc) → loc original

# ── construção do mapa unificado ──────────────────────────────────
TERMOS_ENG_COMP: dict = {}
for _t in TERMOS_TEC:        TERMOS_ENG_COMP[_t] = "TEC"
for _t in TERMOS_AREA:       TERMOS_ENG_COMP[_t] = "AREA"
for _t in TERMOS_FERRAMENTA: TERMOS_ENG_COMP[_t] = "FERRAMENTA"
for _t in TERMOS_HARDWARE:   TERMOS_ENG_COMP[_t] = "HARDWARE"
for _t in TERMOS_CONCEITO:   TERMOS_ENG_COMP[_t] = "CONCEITO"

# Autores dos trabalhos 
AUTORES_TCC = {
    "AnalisedeModelos_Oliveira_2025.txt": "José Augusto Agripino de Oliveira",
    "Chatbot Inteligente para Acesso a Regulamentos Acadêmicos_ Um Sistema de Recuperação de Informações Baseado em RAG - ChatbotInteligenteParaAcessoADocumentosAcademicos.txt": "João Pedro Freire Cabral",
    "CienciadeDadoAplicadaaSaudeOcupacional_Moreira_2025.txt": "Alisson Sousa Moreira",
    "Desenvolvimento de uma Ferramenta de Pintura 3D com Rastreamento por Visão Artificial para Determinação de Profundidade - TCC_Lucas_Lima.txt": "Lucas Lima",
    "Desenvolvimento de uma Solução Baseada em Blockchain para Armazenamento e Rastreamento de Dados Veiculares - TCC___Miguel___EngComp.txt": "Miguel Euripedes Nogueira do Amaral",
    "DesenvolvimentoDeUmSistemaSupervisorio_Amaral_2025.txt": "Adson Emanuel Santos Amaral",
    "DesenvolvimentoDeUmSistemaSupervisório_Ribeiro_2025.txt": "Caio Matheus Lopes Ribeiro",
    "DIANA_SOUSA_2025.txt": "Maria Alice de Melo Sousa",
    "EstimacaoAdaptativa_Rodrigues_2025.txt": "Matheus dos Santos Lopes Rodrigues",
    "Final_Gabriel_Lins_Monografia_Eng_Comp_UFRN - lins_gabriel_sistema_multiagente_LLM_2025_corrigido.txt": "Gabriel Barros Lins Lelis de Oliveira",
    "Interface para Imageamento Geológico utilizando Sensores Ultrassônicos em um Sistema de Caixa de Areia - InterfaceParaImageamento_Klyfton_2025.txt": "Klyfton Stanley Fernandes da Silva Queiroz",
    "Metodologia Orientada a Grandes Modelos de Linguagens para Extração de Conhecimento em Textos Acadêmicos - TCC___Andrade_Matheus_Gomes_Diniz.txt": "Matheus Gomes Diniz Andrade",
    "Microsoft Word - TCC_Eliza_Engecomp_21_01 - TCC_Eliza_Engecomp_final.txt": "Elizabete Cristina Venceslau de Lira",
    "Observabilidade_e_Monitoramento_Brito_2025.txt": "Reyne Jasson Marcelino de Brito",
    "PredicaodeRisco_Freitas_2025.txt": "Gabriel Ribeiro de Freitas",
    "TCC_EFRAIN_oficial.txt": "Efrain Marcelo Pulgar Pantaleon",
    "tcc_ficha.txt": "André Eduardo Meneses do Nascimento",
    "TCC_Final_Gilvandro_Com_Ficha.txt": "Gilvandro César Santos de Medeiros",
    "TCC-Gustavo.txt": "Gustavo Jerônimo Moura de França",
    "TCC_Henrique_Hideaki_UFRN_Final.txt": "Henrique Hideaki Koga",
    "TCC - MARIA J L MACEDO.txt": "Maria Jamilli Lemos de Macedo",
    "TCC_Vitor_Yeso___DCA_UFRN-8.txt": "Vítor Yeso Fidelis Freitas",
}


# Professores DCA (NFD+lower para busca robusta)
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

STOPWORDS_PT = {
    "para","como","onde","quando","este","essa","isso",
    "trabalho","monografia","graduação","resumo",
    "abstract","keywords","introdução","conclusão",
    "capitulo","figura","tabela"
}

STOPWORDS_EN = {
    "the","this","using","were","with","that","from","into","and",
    "for","are","has","have","been","its","their","these","those",
    "which","while","although","despite","must","even","along","where",
    "thus"
}

MAX_TOKENS_ENTIDADE = 4

_LABELS_CONFIAVEIS = {
    "TEC","AREA","FERRAMENTA","HARDWARE","CONCEITO",
    "INST","DEPT","CURSO","ORIENTADOR","LOC",
}

# ══════════════════════════════════════════════════════════════════
# 2. UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════

def _nfd(texto: str) -> str:
    """NFD sem diacríticos, lowercase."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    ).lower()


def _inicializar_locs_nfd() -> None:
    """Preenche LOCS_MANUAIS_NFD uma única vez na importação."""
    for loc in LOCS_MANUAIS:
        LOCS_MANUAIS_NFD[_nfd(loc)] = loc

_inicializar_locs_nfd()


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

    Ordem de aplicação (importa para filter_spans):
      1. LOC  — aplicado PRIMEIRO para proteger "Natal", "RN" etc.
      2. ORIENTADOR — nomes conhecidos do DCA
      3. INST, DEPT, CURSO — entidades acadêmicas
      4. TEC, AREA, FERRAMENTA, HARDWARE, CONCEITO — técnicas

    Por que LOC antes de ORIENTADOR?
      filter_spans escolhe o span MAIOR em caso de sobreposição.
      Se "Natal" (LOC, 5 chars) compete com "Luiz ... Natal" (PER, 30+
      chars), o PER maior venceria. Mas ao registrar LOC primeiro e
      depois chamar _separar_loc_de_nome(), garantimos que LOCs ao final
      de nomes de professores são separados corretamente.
    """
    novas = []
    texto_lower = doc.text.lower()
    texto_nfd   = _nfd(doc.text)

    # ── 1. LOC manual ─────────────────────────────────────────────
    for loc_nfd, loc_orig in LOCS_MANUAIS_NFD.items():
        # Siglas de estado: busca case-sensitive com word-boundary
        if len(loc_orig) <= 2:
            padrao = rf"\b{re.escape(loc_orig)}\b"
            texto_busca = doc.text
        else:
            # Cidades/estados/regiões: busca NFD para acento-insensitive
            padrao = rf"\b{re.escape(loc_nfd)}\b"
            texto_busca = texto_nfd

        for m in re.finditer(padrao, texto_busca):
            sp = doc.char_span(m.start(), m.end(), label="LOC")
            if sp:
                novas.append(sp)

    # ── 2. Professores DCA ────────────────────────────────────────
    for prof_nfd in PROFESSORES_DCA:
        for m in re.finditer(re.escape(prof_nfd), texto_nfd):
            sp = doc.char_span(m.start(), m.end(), label="ORIENTADOR")
            if sp:
                novas.append(sp)

    # ── 3. Entidades acadêmicas ───────────────────────────────────
    for sigla in UNIVERSIDADES:
        for m in re.finditer(rf"\b{re.escape(sigla)}\b", doc.text):
            sp = doc.char_span(m.start(), m.end(), label="INST")
            if sp: novas.append(sp)

    for nome_nfd in UNIVERSIDADES_NOMES_LONGOS:
        for m in re.finditer(re.escape(nome_nfd), texto_nfd):
            sp = doc.char_span(m.start(), m.end(), label="INST")
            if sp: novas.append(sp)

    for sigla in DEPARTAMENTOS:
        for m in re.finditer(rf"\b{re.escape(sigla)}\b", doc.text):
            sp = doc.char_span(m.start(), m.end(), label="DEPT")
            if sp: novas.append(sp)

    for nome_curso in CURSOS:
        for m in re.finditer(re.escape(_nfd(nome_curso)), texto_nfd):
            sp = doc.char_span(m.start(), m.end(), label="CURSO")
            if sp: novas.append(sp)

    # ── 4. Termos técnicos ────────────────────────────────────────
    for termo, label in TERMOS_ENG_COMP.items():
        for m in re.finditer(rf"\b{re.escape(termo.lower())}\b", texto_lower):
            sp = doc.char_span(m.start(), m.end(), label=label)
            if sp: novas.append(sp)

    # Resolve sobreposições (maior span vence)
    doc.ents = filter_spans(list(doc.ents) + novas)

    # ── 5. Pós-processamento: separa LOC de nomes de professores ──
    doc = _separar_loc_de_nome(doc)

    return doc


def _separar_loc_de_nome(doc: Doc) -> Doc:
    """
    Corrige o caso "Luiz Affonso Henderson Guedes De Oliveira Natal":
      - Se uma entidade PER/ORIENTADOR termina com um LOC conhecido,
        divide em duas: PER sem o LOC + LOC separado.
      - Também remove LOC que estejam sobrepostos dentro de um ORIENTADOR.

    Exemplo:
      ANTES:  [ORIENTADOR] "luiz affonso henderson guedes de oliveira natal"
      DEPOIS: [ORIENTADOR] "luiz affonso henderson guedes de oliveira"
              [LOC]        "natal"
    """
    ents_novas = []
    locs_nfd_set = set(LOCS_MANUAIS_NFD.keys())

    for ent in doc.ents:
        if ent.label_ not in ("PER", "ORIENTADOR"):
            ents_novas.append(ent)
            continue

        texto_ent_nfd = _nfd(ent.text)
        tokens_nfd    = texto_ent_nfd.split()

        # Verifica se o(s) último(s) token(s) formam um LOC conhecido
        loc_sufixo_nfd  = None
        loc_sufixo_len  = 0   # em chars do texto original

        # Tenta 1, 2 e 3 tokens finais (ex: "Rio Grande Do Norte")
        for n_tokens in (3, 2, 1):
            if len(tokens_nfd) <= n_tokens:
                continue
            candidato_nfd = " ".join(tokens_nfd[-n_tokens:])
            if candidato_nfd in locs_nfd_set:
                loc_sufixo_nfd = candidato_nfd
                # calcula o tamanho em chars originais
                # (re.search no texto do span)
                m = re.search(
                    rf"\b{re.escape(candidato_nfd)}\b",
                    _nfd(ent.text),
                )
                if m:
                    loc_sufixo_len = len(ent.text) - m.start()
                break

        if loc_sufixo_nfd is None:
            ents_novas.append(ent)
            continue

        # Cria span do nome sem o LOC
        fim_nome_char  = ent.end_char - loc_sufixo_len
        inicio_loc_char = fim_nome_char

        # Remove espaços do final do nome
        while fim_nome_char > ent.start_char and \
              doc.text[fim_nome_char - 1] == " ":
            fim_nome_char -= 1
            inicio_loc_char += 0   # loc começa depois do espaço

        # Remove espaços do início do LOC
        while inicio_loc_char < ent.end_char and \
              doc.text[inicio_loc_char] == " ":
            inicio_loc_char += 1

        sp_nome = doc.char_span(ent.start_char, fim_nome_char,
                                label=ent.label_)
        sp_loc  = doc.char_span(inicio_loc_char, ent.end_char,
                                label="LOC")

        if sp_nome and sp_loc:
            ents_novas.append(sp_nome)
            ents_novas.append(sp_loc)
        else:
            ents_novas.append(ent)   # fallback: mantém original

    doc.ents = filter_spans(ents_novas)
    return doc

def limpar_texto_entidade(t: str) -> str:
    t = t.strip()
    t = re.sub(r"[^\w\s\-ÁÉÍÓÚÀÂÊÔÃÕÇáéíóúàâêôãõç]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

# ══════════════════════════════════════════════════════════════════
# 5. FILTRO PÓS-NER
# ══════════════════════════════════════════════════════════════════

def filtrar_entidade(texto_ent: str, label: str) -> bool:
    t = limpar_texto_entidade(texto_ent)

    if not t:
        return False

    t_nfd = _nfd(t)

    # ─────────────────────────────────────────
    # 🚫 STOPWORDS
    # ─────────────────────────────────────────
    if t_nfd in STOPWORDS_PT or t_nfd in STOPWORDS_EN:
        return False

    # ─────────────────────────────────────────
    # 🚫 MUITO CURTO OU NUMÉRICO
    # ─────────────────────────────────────────
    if len(t) <= 2 or t.isdigit():
        return False

    # ─────────────────────────────────────────
    # 🚫 OCR RUIM (CRÍTICO)
    # ─────────────────────────────────────────
    proporcao_ruim = len(re.findall(r"[^A-Za-z0-9\s\-]", t)) / max(len(t),1)
    if proporcao_ruim > 0.2:
        return False

    if re.search(r"[ˆ˜´`¸]{1,}", t):
        return False

    # ─────────────────────────────────────────
    # 🚫 MUITO GRANDE OU MUITOS TOKENS
    # ─────────────────────────────────────────
    if len(t.split()) > 4 or len(t) > 40:
        return False

    # ─────────────────────────────────────────
    # 👤 PESSOAS (PER / ORIENTADOR)
    # ─────────────────────────────────────────
    if label in ("PER", "ORIENTADOR"):
        if len(t.split()) < 2:
            return False

        if not all(p[0].isupper() for p in t.split()):
            return False

        if re.search(r"\b[A-Z]\.", t):
            return False

        return True

    # ─────────────────────────────────────────
    # 📍 LOC (CRÍTICO)
    # ─────────────────────────────────────────
    if label == "LOC":
        if len(t) < 4:
            return False

        if t.lower() in STOPWORDS_PT or t.lower() in STOPWORDS_EN:
            return False

        # precisa parecer nome próprio
        if not any(c.isupper() for c in t):
            return False

        return True

    # ─────────────────────────────────────────
    # 🏢 ORG
    # ─────────────────────────────────────────
    if label == "ORG":
        if not any(c.isupper() for c in t):
            return False

    # ─────────────────────────────────────────
    # 🚫 FRASES / LIXO
    # ─────────────────────────────────────────
    if ENTIDADE_ARTEFATO.match(t):
        return False

    if RUIDO_ESTRUTURAL.search(t):
        return False

    return True

def filtrar_por_frequencia(doc):
    cont = Counter([_nfd(e.text) for e in doc.ents])
    novas = []

    for e in doc.ents:
        if cont[_nfd(e.text)] > 1:
            novas.append(e)

    doc.ents = novas
    return doc


# ══════════════════════════════════════════════════════════════════
# 6. NORMALIZAÇÃO DE NOMES
# ══════════════════════════════════════════════════════════════════

def normalizar_nome_pessoa(nome: str) -> str:
    """
    Remove prefixos acadêmicos e LOC de cauda, retorna Title Case.
    Ex: 'Prof. Dr. João da Silva Natal' → 'João Da Silva'
    """
    # Remove prefixos
    nome = re.sub(r"^" + PREFIXOS_RX, "", nome, flags=re.IGNORECASE).strip()
    nome = re.sub(r"^[.,;:\-]+", "", nome).strip()

    # Remove LOC conhecida no final (ex: "... Natal", "... RN")
    tokens = nome.split()
    for n_tokens in (3, 2, 1):
        if len(tokens) <= n_tokens:
            continue
        sufixo = " ".join(tokens[-n_tokens:])
        if _nfd(sufixo) in LOCS_MANUAIS_NFD:
            tokens = tokens[:-n_tokens]
            break
    nome = " ".join(tokens)

    return " ".join(p.capitalize() for p in nome.split())


# ══════════════════════════════════════════════════════════════════
# 7. PALAVRAS-CHAVE
# ══════════════════════════════════════════════════════════════════

def extrair_palavras_chave(doc) -> list:
    """
    Extrai palavras-chave do TCC (AREA > TEC > FERRAMENTA > CONCEITO).
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
    normalizar_nome_pessoa() já remove LOC de cauda.
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
    def normalizar_nome_arquivo(nome):
        return _nfd(nome).replace(" ", "").replace("_", "").replace("-", "")

    autor = None
    nome_proc = normalizar_nome_arquivo(nome_arquivo)

    for nome_ref, autor_ref in AUTORES_TCC.items():
        if normalizar_nome_arquivo(nome_ref) == nome_proc:
            autor = autor_ref
            break

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
    # substituições já existentes
    for padrao, substituto in SUBSTITUICOES:
        texto = re.sub(padrao, substituto, texto)

    # NOVO: normalização Unicode forte (remove lixo tipo ˜ ¸ ´)
    texto = unicodedata.normalize("NFKC", texto)

    # remove diacríticos quebrados isolados
    texto = re.sub(r"[ˆ˜´`¸]+", "", texto)

    return texto

def recortar_corpo(texto: str) -> str:
    texto = re.sub(
        r"LISTA\s+DE\s+(ABREVIATURAS?|SIGLAS?|SÍMBOLOS?).+?(?=RESUMO|ABSTRACT)",
        "", texto, flags=re.DOTALL | re.IGNORECASE,
    )
    inicio = re.search(r"\b(RESUMO|ABSTRACT)\b", texto)
    fim    = re.search(
        r"\b(REFER[ÊEĹ]NCIAS?|BIBLIOGRAPHY|BIBLIOGRAF[ÍI]A)\b", texto
    )
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
    texto = re.sub(r"^\s*\S\s*$", "", texto, flags=re.MULTILINE)# remove linhas com muitos símbolos estranhos (OCR quebrado)
    texto = re.sub(r"^[^A-Za-zÁÉÍÓÚÀÂÊÔÃÕÇ0-9\n]{3,}$", "", texto, flags=re.MULTILINE)

    # remove palavras com caracteres inválidos misturados
    texto = re.sub(r"\b\w*[ˆ˜´`¸]+\w*\b", "", texto)
    
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

    tokenizer = ner_bert.tokenizer
    max_len = tokenizer.model_max_length - 2  # 🔥 510
    

    texto = doc.text

    tokens = tokenizer(
        texto,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    input_ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]

    novas = []

    for i in range(0, len(input_ids), max_len):
        chunk_ids = input_ids[i:i+max_len]
        chunk_offsets = offsets[i:i+max_len]

        chunk_text = tokenizer.decode(chunk_ids)

        resultados = ner_bert(chunk_text)

        for r in resultados:
            label = r["entity_group"]

            if label not in ["PER", "LOC", "ORG"]:
                continue

            start_char = r["start"]
            end_char = r["end"]

            try:
                global_start = chunk_offsets[0][0] + start_char
                global_end = chunk_offsets[0][0] + end_char
            except:
                continue

            span = doc.char_span(global_start, global_end, label=label)

            if span:
                novas.append(span)

    doc.ents = filter_spans(list(doc.ents) + novas)
    return doc

def processar_spacy(texto: str):
    nlp = _carregar_modelo()
    if len(texto) >= nlp.max_length:
        nlp.max_length = len(texto) + 10_000
    doc = next(nlp.pipe([texto], batch_size=1))
    doc = aplicar_ner_manual(doc)
    doc.ents = filter_spans(doc.ents)  # garantir consistência
    doc = filtrar_por_frequencia(doc)
    total       = len(doc.ents)
    apos_filtro = [e for e in doc.ents if filtrar_entidade(e.text, e.label_)]
    print(f"\n  Sentenças : {len(list(doc.sents))}")
    print(f"  Entidades : {total} brutas → {len(apos_filtro)} após filtro")
    por_label: dict = {}
    for e in apos_filtro: por_label.setdefault(e.label_, []).append(e.text)
    for label, ents in sorted(por_label.items()):
        print(f"    {label:<12} ({len(ents):>3}): {', '.join(ents[:5])}")
    return doc