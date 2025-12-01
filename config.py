import os
import pytesseract
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Pode atualizar modelos e caminhos, utilizei os caminhos do poppler e do ghostscript por nao ter acesso ao path do sistema.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_CACHE_FILENAME = "motoristas_embeddings.pkl"
EMBED_SIM_THRESHOLD = 0.93
FUZZY_THRESHOLD = 98
PHRASE_MATCH_EXACT = True
MIN_NAME_LEN = 5

INPUT_FOLDER = r"P:\Transporte\15-FROTA\23 - JORNADA\BAIXAS\ADVERTENCIAS\ADVERTENCIAS DIGITALIZADAS PEDRO 15.8.2025"
POPPLER_PATH = r"C:\Users\pedro.a\Downloads\Release-25.07.0-0 (1)\poppler-25.07.0\Library\bin"
MAX_PROCESSES = max(1, cpu_count() - 1)

# Tesseract OCR
TESSERACT_EXE_PATH = r"C:\Users\pedro.a\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE_PATH
os.environ["PATH"] += os.pathsep + os.path.dirname(TESSERACT_EXE_PATH)

# Ghostscript (se necessário por outras libs)
GHOSTSCRIPT_PATH = r"C:\Program Files\gs\gs10.06.0\bin\gswin64c.exe"
os.environ["PATH"] += os.pathsep + os.path.dirname(GHOSTSCRIPT_PATH)

input_folder_documents = r"C:\Users\pedro.a\Documents\documentos"
output_folder = r"C:\Users\pedro.a\Documents\documentos_rename"

# Modelo atual, se preferir pode mudar para grandes modelos pagos
NER_MODEL_PATH = r"marcosgg/bert-base-pt-ner-enamex"

tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)

nlp_name = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"  # agrupa tokens em entidades completas
)