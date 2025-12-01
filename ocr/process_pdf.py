import os
import cv2
import pytesseract
import fitz           # PyMuPDF
from pdf2image import convert_from_path
import numpy as np

# ===========================
# IMPORTANDO SEUS MÓDULOS OCR
# ===========================
from ocr.ocr_cleaner import clean_ocr_text, ocr_pdf_clean
from ocr.ocr_table_detector import process_for_ocr
from ocr.postprocessor_fix_words import postprocess_ocr

# ================================
# IMPORTANDO SEUS MÓDULOS DE NOME
# ================================
from name_extractor.extract_name import extract_name
from name_extractor.clean_text import basic_clean_text

# ==================================
# CONFIG - ajuste para sua máquina
# ==================================
POPPLER_PATH = r"C:\Program Files\poppler-24.02.0\Library\bin"
TESSERACT_EXE = r"C:\Users\pedro.a\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


# =======================================================
# FUNÇÃO PRINCIPAL QUE PROCESSA UM PDF COMPLETO
# =======================================================
def process_pdf(path_pdf: str):
    """
    Pipeline completo:
        1) OCR limpo página por página
        2) Table detector (recortes úteis)
        3) Pós-processamento (fix_words)
        4) Extração de nome (name_extractor)

    Retorna:
        {
            "texto_limpo": ...,
            "texto_corrigido": ...,
            "nome": ...,
            "tabelas_detectadas": ...
        }
    """

    print(f"\n=== PROCESSANDO PDF ===\n{path_pdf}\n")

    # ------------------------------------------------------
    # 1) OCR PROFISSIONAL DO SEU ARQUIVO (ocr_cleaner.py)
    # ------------------------------------------------------
    print("[1] Rodando OCR limpo...")
    full_text = ocr_pdf_clean(path_pdf)

    if not full_text.strip():
        print("⚠ OCR retornou texto vazio!")
    else:
        print("✔ OCR concluído.")

    # ------------------------------------------------------
    # 2) PRÉ-PROCESSAMENTO VISUAL (ocr_table_detector.py)
    # ------------------------------------------------------
    print("[2] Detectando tabelas e bloco de nome...")

    # converter 1ª página em imagem temp
    pages = convert_from_path(path_pdf, dpi=300, poppler_path=POPPLER_PATH)
    first_page_path = "_temp_first_page.jpg"
    pages[0].save(first_page_path)

    rotated, table_crop, name_crop = process_for_ocr(first_page_path)

    print("✔ Table detector concluído.")

    # ------------------------------------------------------
    # 3) PÓS-PROCESSAMENTO DE TEXTO (postprocessor_fix_words.py)
    # ------------------------------------------------------
    print("[3] Limpando texto do OCR (fix_words)...")
    text_corrigido = postprocess_ocr(full_text)

    # ------------------------------------------------------
    # 4) EXTRAÇÃO DE NOME (name_extractor)
    # ------------------------------------------------------
    print("[4] Extraindo nome do texto...")
    nome = extract_name(text_corrigido)

    print(f"✔ Nome detectado: {nome}")

    # ------------------------------------------------------
    # 5) RETORNO FINAL
    # ------------------------------------------------------
    result = {
        "texto_limpo": full_text,
        "texto_corrigido": text_corrigido,
        "nome": nome,
        "tabelas_detectadas": {
            "rotacionada": rotated,
            "tabela": table_crop,
            "nome_crop": name_crop
        }
    }

    return result


# ===================================================
# TESTE DIRETO (como o usuário executaria)
# ===================================================
if __name__ == "__main__":
    arquivo = r"C:\temp\arquivo.pdf"
    dados = process_pdf(arquivo)

    print("\n=== RESULTADO FINAL ===")
    print("Nome:", dados["nome"])
    print("\nTexto Corrigido:\n", dados["texto_corrigido"][:1000], "...")
