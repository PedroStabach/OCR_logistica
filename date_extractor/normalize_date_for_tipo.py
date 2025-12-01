from datetime import datetime

def normalize_date_for_tipo(date_obj, tipo):
    hoje = datetime.today()
    if date_obj is None:
        date_obj = hoje

    # Força dia 15 para pontos, advertências e cartas
    if tipo in ["ponto", "advertencia", "carta"]:
        try:
            date_obj = date_obj.replace(day=15)
        except Exception:
            pass

    # Corrige ano mínimo e máximo
    if date_obj.year < 2023:
        date_obj = date_obj.replace(year=2023)
    if date_obj > hoje:
        date_obj = date_obj.replace(year=hoje.year)
        if date_obj > hoje:
            month = min(date_obj.month, hoje.month)
            date_obj = date_obj.replace(month=month)

    return date_obj