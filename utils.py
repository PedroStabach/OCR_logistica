import os
import tempfile
import ocrmypdf
import fitz
import traceback
import numpy as np
import cv2
from PIL import Image
import pytesseract
import re
import pypdf
import unicodedata

# EXTRACAO DE TEXTO

from PyPDF2 import PdfReader


def extract_text_with_ocrmypdf(path):
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            out = tmp.name

        ocrmypdf.ocr(
            input_file=path,
            output_file=out,
            force_ocr=True,
            skip_text=True,
            optimize=0,          # <<< DESLIGA PNGQUANT/JBIG2 (parou o WinError)
            use_threads=True,
            progress_bar=False
        )

        # Lê o PDF gerado pelo OCR
        from PyPDF2 import PdfReader
        reader = PdfReader(out)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        return text.strip()

    except Exception as e:
        return ""


# PREPROCESSAMENTO
def preprocess_image_opencv(pil_image):
    img = np.array(pil_image.convert("L"))  # Grayscale
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.fastNlMeansDenoising(img, None, 15, 7, 21)

    coords = np.column_stack(np.where(img < 255))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(img)

# ORIENTACAO TEXTO
def detect_orientation_tesseract(pil_image):
    try:
        osd = pytesseract.image_to_osd(pil_image)
        angle = int(re.search(r'Rotate: (\d+)', osd).group(1))

        if angle in (90, 270):
            return "horizontal"
        return "vertical"
    except:
        return "vertical"
    
def clean_text(text):
    text = text.replace('’', "'").replace('“', '"').replace('”', '"')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_num_pages(file_path):
    """Retorna o número total de páginas de um arquivo PDF."""
    try:
        reader = pypdf.PdfReader(file_path)
        return len(reader.pages)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {file_path}")
        return 0
    except Exception as e:
        print(f"Erro ao ler o número de páginas de {file_path}: {e}")
        return 1
    
def normalize_text(t):
    return ''.join(c for c in unicodedata.normalize('NFD', t)
                   if unicodedata.category(c) != 'Mn').lower()

# utils.py
import pytesseract
import re
import cv2
import numpy as np

def extract_date_with_special_ocr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (3,3), 0)

    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/.-"
    raw = pytesseract.image_to_string(img, config=config)

    date_regex = r"\b(?:[0-3]?\d[\/.\-][0-1]?\d[\/.\-](?:\d{2}|\d{4}))\b"
    return re.findall(date_regex, raw)
