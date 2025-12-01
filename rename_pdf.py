import os

# Atualizar para padrao individual
def rename_pdf(file_path, tipo, nome, data_str, motivo_str):
    folder = os.path.dirname(file_path)
    if isinstance(nome, list):
    # pega só o nome
        nome = nome[1] if len(nome) > 1 else nome[0]
    nome_safe = (nome or "DESCONHECIDO").strip()
    data_safe = (data_str or "").replace("/", "-").replace(".", "-").strip()
    motivo_safe = (motivo_str or "").strip()

    if tipo == "carta":
        base_name = f"CARTA {nome_safe} {data_safe}.pdf"
    elif tipo == "advertencia":
        base_name = f"ADVERTENCIA {motivo_safe} {data_safe} {nome_safe}.pdf"
    elif tipo == "ponto":
        base_name = f"PONTO {data_safe} {nome_safe}.pdf"
    elif tipo == "atestado":
        base_name = f"ATESTADO {data_safe} {nome_safe}.pdf"
    else:
        base_name = f"DESCONHECIDO {data_safe} {nome_safe}.pdf"

    new_path = os.path.join(folder, base_name)
    count = 1
    while os.path.exists(new_path):
        name_no_ext, ext = os.path.splitext(base_name)
        new_path = os.path.join(folder, f"{name_no_ext}({count}){ext}")
        count += 1
    try:
        os.rename(file_path, new_path)
        print(f"✅ Renomeado para: {os.path.basename(new_path)}")
    except Exception as e:
        print(f"⚠️ Erro ao renomear {file_path}: {e}")