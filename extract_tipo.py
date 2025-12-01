import re
from utils import normalize_text

def extract_tipo(text, orientation, num_pages):
    t = normalize_text(text)

    # Scores
    score = {
        "ponto": 0,
        "advertencia": 0,
        "atestado": 0,
        "carta": 0
    }

    # 1) Palavras-chave de Forte Evidência
    if "atestado" in t or "cid" in t or "dr" in t or "consultorio" in t:
        score["atestado"] += 40

    if "advertencia" in t or "intrajornada" in t or "descumprimento" in t or "advertido" in t:
        score["advertencia"] += 40

    if "prezado" in t or "venho por meio desta" in t or "ao senhor" in t:
        score["carta"] += 40

    if "registro de ponto" in t or "diario de bordo" in t or "horas" in t or "Controle de Jornada" in t:
        score["ponto"] += 40

    # 2) Estrutura do Documento
    if num_pages > 1:
        score["ponto"] += 30

    numeros = len(re.findall(r"\d{1,2}[:h]\d{2}", t))
    if numeros >= 5:
        score["ponto"] += 40

    # verificar presença de padrões horários/tabulares comuns
    linhas = t.count("00:00:00") + t.count("00:00")
    if linhas >= 3:
        score["ponto"] += 20

    if len(text.split("\n")) < 10 and len(text) > 400:
        score["advertencia"] += 10
        score["carta"] += 10

    if any(k in t for k in ["colaborador", "conduta", "motivo", "notificado", "empresa"]):
        score["advertencia"] += 20

    # 3) Indícios médios
    if any(k in t for k in ["compareceu", "afastamento", "atend", "clinica", "medico", "cid"]):
        score["atestado"] += 15

    # 4) Orientação da página (último recurso)
    if orientation == "horizontal":
        score["ponto"] += 5
    else:
        score["advertencia"] += 5

    # 5) Decisão Final
    tipo_final = max(score, key=score.get)
    if score[tipo_final] < 20:
        return "desconhecido"
    return tipo_final