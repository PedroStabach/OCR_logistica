# === file: name_extractor/clean_text.py ===
import re
import unicodedata

MIN_NAME_LEN = 5

# Palavras que normalmente causam FALSOS POSITIVOS
STOPWORDS_HEADER = {
    "estado", "prefeitura", "municipio", "secretaria", "departamento",
    "cpf", "rg", "data", "assinatura", "endereco", "telefone",
    "brasil", "sistema", "modelo", "empresa", "documento",
    "requerimento", "orgao", "funcionario"
}


def normalize_text_simple(t: str) -> str:
    """Normalize text for matching: remove diacritics, lower, keep alnum + spaces."""
    if not t:
        return ""
    t = unicodedata.normalize('NFD', t)
    t = t.encode('ascii', 'ignore').decode('utf-8', 'ignore')
    t = re.sub(r'[^a-z0-9\s]', ' ', t.lower())
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def limpar_ruido_name(t: str) -> str:
    """Remove símbolos, múltiplos espaços e linhas suspeitas."""
    if not t:
        return ""

    # símbolos
    t = t.replace('’', "'").replace('“', '"').replace('”', '"')
    t = re.sub(r'[_~^°§#@*+=<>|\\{}\[\]`´]+', ' ', t)

    linhas = []
    for linha in t.splitlines():
        ln_norm = normalize_text_simple(linha)

        # linha inteira em CAPS = provável header / título
        if linha.strip().upper() == linha.strip() and len(linha.strip()) > 6:
            continue

        # linha que contém stopwords estruturais
        if any(sw in ln_norm.split() for sw in STOPWORDS_HEADER):
            continue

        linhas.append(linha)

    t = "\n".join(linhas)
    t = re.sub(r'\s{2,}', ' ', t)
    return t.strip()


def nome_valido(nome: str) -> bool:
    """Valida se parece um nome humano real."""
    if not nome or len(nome.strip()) < MIN_NAME_LEN:
        return False

    nome_limpo = normalize_text_simple(nome)

    # contém número → não é nome
    if re.search(r"\d", nome_limpo):
        return False

    partes = nome_limpo.split()
    if len(partes) < 2:
        return False

    # todas as partes precisam ter tamanho razoável
    if len(max(partes, key=len)) < 3:
        return False

    return True
