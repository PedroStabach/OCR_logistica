from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from config import NER_MODEL_PATH

def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return nlp
