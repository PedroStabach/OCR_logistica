from utils import normalize_text

def extract_motivo(text):
    t = normalize_text(text)
    if "celular" in t:
        return "CELULAR"
    elif "distracao" in t or "distra" in t:
        return "DISTRACAO"
    elif "cinto" in t:
        return "CINTO"
    elif "velocidade" in t:
        return "VELOCIDADE"
    elif "drogas" in t or "fumando" in t or "bebendo" in t:
        return "DROGAS"
    elif "jornada" in t:
        return "JORNADA"
    else:
        return "INTRAJORNADA"