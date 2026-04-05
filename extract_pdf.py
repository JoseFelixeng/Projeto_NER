import os
import fitz  # PyMuPDF
import re

# Caminhos
PDF_FOLDER = "TCC"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Função para limpar texto
def limpar_texto(texto):
    texto = re.sub(r'\n+', ' ', texto)        # remove quebras de linha
    texto = re.sub(r'\s+', ' ', texto)        # remove espaços duplicados
    texto = re.sub(r'\d+\s*', '', texto)      # remove números soltos
    return texto.strip()

# Função para extrair texto de um PDF
def extrair_texto_pdf(pdf_path):
    texto = ""
    try:
        doc = fitz.open(pdf_path)
        for pagina in doc:
            texto += pagina.get_text()
    except Exception as e:
        print(f"Erro ao ler {pdf_path}: {e}")
    return texto

# Processar todos os PDFs
def processar_pdfs():
    arquivos = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    print(f"📄 Encontrados {len(arquivos)} PDFs...\n")

    for arquivo in arquivos:
        caminho_pdf = os.path.join(PDF_FOLDER, arquivo)

        print(f"🔍 Processando: {arquivo}")

        texto = extrair_texto_pdf(caminho_pdf)
        texto_limpo = limpar_texto(texto)

        nome_saida = arquivo.replace(".pdf", ".txt")
        caminho_saida = os.path.join(OUTPUT_FOLDER, nome_saida)

        with open(caminho_saida, "w", encoding="utf-8") as f:
            f.write(texto_limpo)

        print(f"✅ Salvo em: {caminho_saida}\n")

    print("🎉 Todos os PDFs foram processados!")

# Executar
if __name__ == "__main__":
    processar_pdfs()