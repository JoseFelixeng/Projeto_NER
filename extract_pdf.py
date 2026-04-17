"""
extract_pdf_v2.py — Extração avançada de PDFs acadêmicos
"""

import os
import fitz  # PyMuPDF
import re

PDF_FOLDER = "TCC"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────
# LIMPEZA INTELIGENTE
# ─────────────────────────────────────────────
def limpar_texto_avancado(texto: str) -> str:
    # normaliza espaços
    texto = re.sub(r'[ \t]+', ' ', texto)

    # preserva parágrafos (2+ quebras = novo parágrafo)
    texto = re.sub(r'\n{3,}', '\n\n', texto)

    # remove linhas muito curtas (ruído)
    linhas = []
    for linha in texto.split("\n"):
        l = linha.strip()

        # remove linhas irrelevantes
        if len(l) <= 2:
            continue
        if re.match(r'^\d+$', l):  # número isolado
            continue
        if re.match(r'^(Figura|Tabela|Fonte)', l, re.IGNORECASE):
            continue

        linhas.append(l)

    texto = "\n".join(linhas)

    # remove múltiplos espaços novamente
    texto = re.sub(r' {2,}', ' ', texto)

    return texto.strip()


# ─────────────────────────────────────────────
# REMOÇÃO DE CABEÇALHO/RODAPÉ
# ─────────────────────────────────────────────
def remover_cabecalho_rodape(pagina):
    blocos = pagina.get_text("blocks")

    altura = pagina.rect.height

    texto_util = ""

    for b in blocos:
        x0, y0, x1, y1, texto, *_ = b

        # remove topo e rodapé (~10%)
        if y0 < altura * 0.1:
            continue
        if y1 > altura * 0.9:
            continue

        texto_util += texto + "\n"

    return texto_util


# ─────────────────────────────────────────────
# EXTRAÇÃO AVANÇADA
# ─────────────────────────────────────────────
def extrair_texto_pdf_avancado(pdf_path):
    texto = ""

    try:
        doc = fitz.open(pdf_path)

        for pagina in doc:
            texto += remover_cabecalho_rodape(pagina)
            texto += "\n\n"  # separa páginas

    except Exception as e:
        print(f"Erro ao ler {pdf_path}: {e}")

    return texto


# ─────────────────────────────────────────────
# PROCESSAMENTO EM LOTE
# ─────────────────────────────────────────────
def processar_pdfs():
    arquivos = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    print(f"📄 Encontrados {len(arquivos)} PDFs...\n")

    for arquivo in arquivos:
        caminho_pdf = os.path.join(PDF_FOLDER, arquivo)

        print(f"🔍 Processando: {arquivo}")

        texto = extrair_texto_pdf_avancado(caminho_pdf)
        texto_limpo = limpar_texto_avancado(texto)

        nome_saida = arquivo.replace(".pdf", ".txt")
        caminho_saida = os.path.join(OUTPUT_FOLDER, nome_saida)

        with open(caminho_saida, "w", encoding="utf-8") as f:
            f.write(texto_limpo)

        print(f"✅ Salvo em: {caminho_saida}\n")

    print("🎉 Todos os PDFs foram processados!")


if __name__ == "__main__":
    processar_pdfs()