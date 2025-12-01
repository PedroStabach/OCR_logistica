import cv2
import numpy as np
from typing import Tuple, Optional

# Necessario para pontos em listas coloridas ou com marcas

"""
OCR TABLE DETECTOR
------------------
Detector especializado para PDFs/imagens de RELATÓRIO DE PONTO.
Objetivo:
  - Detectar área útil
  - Localizar bloco vertical onde geralmente está o nome do motorista
  - Corrigir rotação/deskew
  - Finalidade: melhorar OCR posterior

Funções principais:
  detect_table_region(img)    -> retorna crop da área útil
  correct_rotation(img)       -> deskew
  detect_name_block(img)      -> crop focado no bloco do nome
  process_for_ocr(path)       -> pipeline completa

IMPORTANTE:
  Este módulo NÃO faz OCR — ele prepara a imagem para o OCR externo.
"""

# ---------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------
def _to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def correct_rotation(img: np.ndarray) -> np.ndarray:
    """
    Deskew baseado em projeção + Hough.
    Funciona bem para documentos de ponto.
    """
    gray = _to_gray(img)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)
    if lines is None:
        return img

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return img

    median_angle = np.median(angles)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ---------------------------------------------------------------
# Detectar área útil (tabela principal)
# ---------------------------------------------------------------
def detect_table_region(img: np.ndarray) -> np.ndarray:
    """
    A tabela de ponto ocupa quase toda a página, mas tem margens.
    Estratégia:
      - transformar para binário
      - detectar contornos grandes
      - pegar o maior retângulo interno
    """
    gray = _to_gray(img)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 10)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    # Ordena contornos por área
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])

    # margem mínima para evitar pegar borda demais
    pad = int(min(w, h) * 0.01)
    x2 = max(0, x - pad)
    y2 = max(0, y - pad)

    w2 = min(img.shape[1] - x2, w + 2 * pad)
    h2 = min(img.shape[0] - y2, h + 2 * pad)

    return img[y2:y2 + h2, x2:x2 + w2]


# ---------------------------------------------------------------
# Detectar bloco do nome do motorista
# ---------------------------------------------------------------
def detect_name_block(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Em relatórios de ponto, o "Funcionário:" ou nome fica no TOPO
    ou lado superior-esquerdo.

    Heurística:
      - pegar 25% superior
      - procurar por área de maior densidade de caracteres
      - retornar região mais provável
    """
    h, w = img.shape[:2]
    top_region = img[0:int(h * 0.28), :]
    gray = _to_gray(top_region)

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 21, 10)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return top_region

    # pegar contorno mais largo (texto geralmente fica em linha ampla)
    best = None
    best_w = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > best_w:
            best = (x, y, cw, ch)
            best_w = cw

    if best is None:
        return top_region

    x, y, cw, ch = best

    # retorna com algum padding
    pad = int(w * 0.02)
    x2 = max(0, x - pad)
    y2 = max(0, y - pad)
    cw2 = min(w - x2, cw + pad * 2)
    ch2 = min(top_region.shape[0] - y2, ch + pad * 2)

    return top_region[y2:y2 + ch2, x2:x2 + cw2]


# ---------------------------------------------------------------
# PIPELINE COMPLETA PARA OCR
# ---------------------------------------------------------------
def process_for_ocr(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega imagem → corrige rotação → detecta tabela → localiza bloco do nome
    Retorna:
        img_rotated, table_crop, name_crop
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar: {path}")

    rotated = correct_rotation(img)
    table = detect_table_region(rotated)
    name = detect_name_block(table)
    return rotated, table, name


# ---------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------
def save_debug(rotated, table, name, prefix="debug"):
    cv2.imwrite(f"{prefix}_rotated.jpg", rotated)
    cv2.imwrite(f"{prefix}_table.jpg", table)
    cv2.imwrite(f"{prefix}_name.jpg", name)
