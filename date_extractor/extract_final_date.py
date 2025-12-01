from .normalize_date_for_tipo import normalize_date_for_tipo
from .parse_posible_date import parse_possible_date
from utils import extract_date_with_special_ocr
import re

def extract_final_date(text, tipo, ocr_image_path=None):

    found = []

    # 1) OCR dedicado para datas
    if ocr_image_path:
        ocr_dates = extract_date_with_special_ocr(ocr_image_path)
        for raw in ocr_dates:
            dt = parse_possible_date(raw)
            if dt:
                found.append(dt)

    # 2) Regex no texto principal
    date_patterns = [
        r'\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4}',
        r'\d{4}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{1,2}',
        r'\d{1,2}\s+de\s+[a-zç]+(?:\s+de\s+\d{4})?'
    ]

    for patt in date_patterns:
        for m in re.finditer(patt, text.lower()):
            dt = parse_possible_date(m.group(0))
            if dt:
                found.append(dt)

    # 3) Fallback
    if not found:
        dt = normalize_date_for_tipo(None, tipo)
        return dt.strftime("%d-%m-%Y")

    final_dt = normalize_date_for_tipo(max(found), tipo)
    return final_dt.strftime("%d-%m-%Y")
