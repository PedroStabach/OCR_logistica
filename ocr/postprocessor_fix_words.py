import re
import unicodedata
from typing import List

"""
POSTPROCESSADOR FIX WORDS
-------------------------
Corrige texto vindo do OCR Tesseract, com foco em:
  - palavras quebradas
  - unir fragmentos de nomes
  - remover lixo
  - normalizar acentuação
  - colapsar múltiplos espaços
  - corrigir padrões frequentes em relatórios de ponto

Funções principais:
  clean_ocr_text(text)            -> limpa geral
  fix_word_splits(text)           -> une palavras quebradas idiotamente pelo OCR
  fix_common_breaks(text)         -> heurísticas focadas em nomes
  postprocess_ocr(text, nomes)    -> pipeline completa
"""

# ---------------------------------------------------------------
# Normalização básica
# ---------------------------------------------------------------
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------------------------------------------------------------
# Remover lixo característico de OCR
# ---------------------------------------------------------------
def clean_ocr_text(text: str) -> str:
    text = text.lower()
    # remove símbolos estranhos
    text = re.sub(r"[^a-zà-ÿ0-9 \-\/\\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------
# Corrigir palavras quebradas 
# ---------------------------------------------------------------
def fix_word_splits(text: str) -> str:
    # junta "pe droso uz a" → "pedro souza"
    text = re.sub(r"([a-zà-ÿ])\s+([a-zà-ÿ])", r"\1\2", text)
    # mas recoloca espaço entre nomes com mais de 3 letras
    text = re.sub(r"([a-zà-ÿ]{3,})([A-ZÀ-Ý][a-zà-ÿ]{3,})", r"\1 \2", text)
    return text


# ---------------------------------------------------------------
# Foco nos nomes: corrigir quebras e espaçamentos errados
# ---------------------------------------------------------------
def fix_common_breaks(text: str) -> str:
    # une nomes muito separados
    text = re.sub(r"\b([a-zà-ÿ]{2,})\s{1,2}([a-zà-ÿ]{2,})\s{1,2}([a-zà-ÿ]{2,})\b",
                  lambda m: " ".join([m.group(1), m.group(2), m.group(3)]),
                  text)

    # Corrigir sequências do tipo "cla y ton nu nes do nas ci men to"
    text = re.sub(r"(\b[a-zà-ÿ]{1,3})\s+(?=[a-zà-ÿ]{1,3}\b)", lambda m: m.group(1), text)

    return text


# ---------------------------------------------------------------
# Pipeline completa
# ---------------------------------------------------------------
def postprocess_ocr(text: str, motoristas: List[str] = None) -> str:
    """
    Pipeline geral de correção.

    Se motoristas for passado:
       → tenta alinhar texto aos nomes da lista
    """
    if not text:
        return ""

    t = normalize_text(text)
    t = clean_ocr_text(t)
    t = fix_word_splits(t)
    t = fix_common_breaks(t)

    # recolocar capitalização
    t = " ".join([w.capitalize() for w in t.split()])

    if motoristas:
        # tentativa simples de alinhar texto a algum nome conhecido
        norm_list = [normalize_text(m[1].lower()) for m in motoristas]
        t_norm = normalize_text(t.lower())
        for orig, norm in zip(motoristas, norm_list):
            if norm in t_norm:
                return orig

    return t