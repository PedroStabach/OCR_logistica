# === extract_name (versão otimizada e corrigida) ===
import spacy
import re
import numpy as np
from typing import List, Optional
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
from .clean_text import normalize_text_simple, limpar_ruido_name
from .matcher import build_spacy_matchers
from .embeddings import load_or_build_embeddings, _HAS_ST
# No topo de name_extractor/extract_name.py
import spacy
import re
import numpy as np
from config import NER_MODEL_PATH
from typing import List, Optional
# ... outras importações ...

# IMPORTAÇÕES ADICIONAIS NECESSÁRIAS:
try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch
    _HAS_HF = True
except ImportError:
    # Se as bibliotecas não estiverem instaladas, defina um flag e o modelo como None
    print("AVISO: Bibliotecas 'transformers' e 'torch' não encontradas. O NER do HuggingFace será desativado.")
    _HAS_HF = False
# ==============
# CONFIG
# ==============
FUZZY_THRESHOLD = 97
EMBED_SIM_THRESHOLD = 0.93


# ==============================
#   Helpers essenciais
# ==============================
def motorista_nome(m):
    """Retorna somente o nome independente de (cod, nome)."""
    if isinstance(m, (list, tuple)):
        return m[1]
    return str(m)


def split_first_last(name: str):
    """Divide nome em primeiro e último (normalizados)."""
    if not name:
        return "", ""
    parts = normalize_text_simple(name).split()
    if not parts:
        return "", ""
    return parts[0], parts[-1]


def edit_distance(a, b):
    """Distância Levenshtein com fallback."""
    try:
        import jellyfish
        return jellyfish.levenshtein_distance(a, b)
    except:
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, a, b).ratio()
        maxlen = max(len(a), len(b), 1)
        return int(round((1 - ratio) * maxlen))


# ==============================
#   Detector de códigos 100% melhorado
# ==============================
def detectar_codigo(text):
    """
    Encontra códigos nos formatos:
    NOME – 12345
    12345 - NOME
    matricula: 12345
    isolados 4–10 dígitos
    """
    if not text:
        return None

    padroes = [
        r"[A-ZÀ-Ý][A-Za-zÀ-ÿ\s]{3,60}[\-\–\—\: ]+(\d{4,10})",   # nome — 12345
        r"(\d{4,10})[\-\–\—\: ]+[A-ZÀ-Ý][A-Za-zÀ-ÿ\s]{3,60}",   # 12345 — nome
        r"(?:matr[ií]cula|registro|id)[\s\:]*?(\d{4,10})",       # campo específico
        r"\b(\d{4,10})\b"                                        # isolado
    ]

    for p in padroes:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1)

    return None


# ==============================
#   Pipeline setup
# ==============================
def load_hf_ner(model_path: str = "marcosgg/bert-base-pt-ner-enamex"):
    """
    Carrega o modelo BERT/NER do HuggingFace.
    Retorna (model, tokenizer) ou (None, None) em caso de falha.
    """
    if not _HAS_HF:
        return None, None

    print(f"Carregando modelo HuggingFace NER: {model_path}...")
    
    try:
        # Carrega o Tokenizer (necessário para pré-processar o texto para o BERT)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Carrega o Modelo (específico para a tarefa de Token Classification/NER)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        # Define o modo de avaliação
        model.eval()
        
        # Retorna o modelo e o tokenizer
        return model, tokenizer
        
    except Exception as e:
        print(f"ERRO ao carregar o modelo HuggingFace: {e}")
        return None, None
def setup_name_pipeline(motoristas: List[str], nlp_model_path: Optional[str] = None) -> tuple:
    """
    Configura pipeline de NER com Hugging Face + spaCy + embeddings.
    Retorna: nlp_name, phrase_matcher, ruler, embed_model, embed_matrix, hf_ner
    """

    # Carrega SpaCy
    if nlp_model_path:
        try:
            nlp = spacy.load(nlp_model_path)
        except:
            nlp = spacy.blank("pt")
    else:
        try:
            nlp = spacy.load("pt_core_news_sm")
        except:
            nlp = spacy.blank("pt")

    # PhraseMatcher + EntityRuler
    phrase_matcher, ruler = build_spacy_matchers(nlp, motoristas)

    try:
        if "entity_ruler" in nlp.pipe_names:
            nlp.remove_pipe("entity_ruler")
    except:
        pass

    nlp.add_pipe("entity_ruler", before="ner" if "ner" in nlp.pipe_names else None)
    ruler_pipe = nlp.get_pipe("entity_ruler")

    patterns = [
        {"label": "NOME", "pattern": nome}
        for cod, nome in motoristas
        if nome and len(nome) >= 3
    ]

    if patterns:
        ruler_pipe.add_patterns(patterns)

    # Embeddings
    motoristas_norm = [normalize_text_simple(m[1]) for m in motoristas]
    embed_model, embed_matrix = None, None

    if _HAS_ST:
        try:
            embed_model, embed_matrix = load_or_build_embeddings(motoristas_norm)
        except:
            pass

    # HuggingFace NER
    hf_ner = load_hf_ner()  # ← GARANTE A EXISTÊNCIA

    # 🔥 GARANTE RETORNO DE 6 VALORES SEMPRE
    return nlp, phrase_matcher, ruler, embed_model, embed_matrix, hf_ner


# ==============================
#  Pipeline principal reduzido e limpo
# ==============================
def _pipeline_extract(text, motoristas, nlp, phrase_matcher, embed_model, embed_matrix):
    """
    Tenta:
    1) PhraseMatcher
    2) spaCy NER
    3) Fuzzy
    4) Embeddings
    """
    text_orig = text or ""
    text = limpar_ruido_name(text_orig)

    noms = [normalize_text_simple(m[1]) for m in motoristas]

    # 1) PhraseMatcher
    if phrase_matcher and nlp:
        doc = nlp(text)
        matches = phrase_matcher(doc)
        if matches:
            span = doc[matches[0][1]:matches[0][2]].text.strip()
            norm = normalize_text_simple(span)
            if norm in noms:
                return motoristas[noms.index(norm)]

    # 2) spaCy NER
    if nlp:
        doc = nlp(text_orig)
        for ent in doc.ents:
            if ent.label_ in ("NOME", "PER", "PERSON"):
                cand = normalize_text_simple(ent.text)
                res = rf_process.extractOne(cand, noms, scorer=rf_fuzz.token_set_ratio)
                if res and res[1] >= FUZZY_THRESHOLD:
                    return motoristas[noms.index(res[0])]

    # 3) Fuzzy global
    res = rf_process.extractOne(normalize_text_simple(text), noms, scorer=rf_fuzz.token_set_ratio)
    if res and res[1] >= FUZZY_THRESHOLD:
        return motoristas[noms.index(res[0])]

    # 4) Embeddings
    if _HAS_ST and embed_model is not None and embed_matrix is not None:
        spans = [s for s in text.split() if len(s) > 3]
        if spans:
            vecs = embed_model.encode(spans, convert_to_numpy=True)
            emb_norm = embed_matrix / np.linalg.norm(embed_matrix, axis=1, keepdims=True)
            vec_norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            sims = np.dot(vec_norm, emb_norm.T)
            best_idx = sims.max(axis=1).argmax()
            if sims[best_idx].max() >= EMBED_SIM_THRESHOLD:
                return motoristas[sims[best_idx].argmax()]

    return None


# ==============================
#  EXTRACT NAME FINAL 100% ESTÁVEL
# ==============================
def extract_name(text, motoristas,
                 nlp=None, phrase_matcher=None, ruler=None,
                 embed_model=None, embed_matrix=None,
                 is_pdf_path=False):

    text_raw = text

    # 1) Código primeiro (melhor caminho)
    codigo = detectar_codigo(text_raw)
    if codigo:
        for cod, nome in motoristas:
            if cod == codigo:
                return nome

    # 2) Pipeline principal
    nome = _pipeline_extract(text_raw, motoristas, nlp, phrase_matcher,
                             embed_model, embed_matrix)
    if nome:
        return nome

    # 3) Tentativa final: fuzzy estrito sobre texto inteiro
    nomes_norm = [normalize_text_simple(n) for _, n in motoristas]
    text_norm = normalize_text_simple(text_raw)

    best = rf_process.extractOne(text_norm, nomes_norm, scorer=rf_fuzz.token_set_ratio)
    if best and best[1] >= 98:
        return motoristas[nomes_norm.index(best[0])]

    return "DESCONHECIDO"
