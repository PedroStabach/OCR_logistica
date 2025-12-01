# ============================================================
# ocr_cleaner.py  |  Python 3.11
# Pipeline completo de OCR robusto + deskew + limpeza profunda
# ============================================================

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import re
import unicodedata

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
POPPLER_PATH = r"C:\Program Files\poppler-24.02.0\Library\bin"
TESSERACT_EXE = r"C:\Users\pedro.a\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


# ------------------------------------------------------------
# NORMALIZAÇÃO — remove acentos + normaliza caracteres
# ------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


# ------------------------------------------------------------
# DESKEW – corrigir inclinação automaticamente
# ------------------------------------------------------------
def deskew_cv(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ------------------------------------------------------------
# LIMPEZA PROFUNDA DE TEXTO OCR
# ------------------------------------------------------------
def clean_ocr_text(text: str) -> str:

    # remover caracteres invisíveis
    text = text.replace("\x0c", " ")

    # remover múltiplos espaços
    text = re.sub(r"\s+", " ", text)

    # remover lixo comum de OCR
    text = re.sub(r"[^\w\s.,:/\-ºª%()]", "", text)

    # remover quebras estranhas
    text = text.replace(" \n", " ").replace("\n ", " ")

    # remover linhas vazias
    text = "\n".join([l.strip() for l in text.split("\n") if l.strip()])

    # normalizar acentos
    text = normalize_unicode(text)

    # remover hífens no fim de linha
    text = re.sub(r"-\s*\n", "", text)

    return text.strip()


# ------------------------------------------------------------
# OCR PRINCIPAL — PDF → texto limpo
# ------------------------------------------------------------
def ocr_pdf_clean(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)

    final_text = []

    for pil_img in pages:
        img = np.array(pil_img)

        # deskew
        img = deskew_cv(img)

        # OCR
        raw_text = pytesseract.image_to_string(img, lang="por")

        # limpar
        cleaned = clean_ocr_text(raw_text)
        final_text.append(cleaned)

    return "\n\n".join(final_text).strip()


# ------------------------------------------------------------
# MAIN (exemplo de uso)
# ------------------------------------------------------------
if __name__ == "__main__":
    file = r"C:\temp\arquivo.pdf"
    print(ocr_pdf_clean(file))
