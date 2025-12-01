import traceback
from name_extractor import setup_name_pipeline, extract_name
from config import NER_MODEL_PATH
from motoristas import motoristas
# ---------------- EXTRAÇÃO DE NOME (INTEGRAÇÃO COM name_extractor) ----------------

# O import e setup do pipeline devem ocorrer depois de definir 'motoristas'
# (se já importou no topo, apenas crie o pipeline aqui)
try:
    # name_extractor.setup_name_pipeline and name_extractor.extract_name were imported earlier
    # name_extractor.extract_name foi importado como 'extract_name' no topo; guardamos referência
    name_extractor_fn = extract_name  # função importada do package
except NameError:
    name_extractor_fn = None

# Inicializa pipeline do name_extractor (usa seu NER custom se fornecido)
try:
    nlp_name, phrase_matcher, ruler, emb_model, emb_matrix, hf_ner = setup_name_pipeline(motoristas, nlp_model_path=NER_MODEL_PATH)
    print(f"✅ name_extractor pipeline inicializada (modelo: {NER_MODEL_PATH})")
except Exception as e:
    print(f"⚠️ Falha ao inicializar name_extractor pipeline: {e}")
    traceback.print_exc()
    nlp_name = phrase_matcher = ruler = emb_model = emb_matrix = None


def extract_name_pipeline(text):
    """
    Wrapper que chama a função extract_name do módulo name_extractor com os objetos do pipeline.
    Retorna nome do motorista (exato da lista) ou 'DESCONHECIDO'.
    """
    try:
        if name_extractor_fn is None:
            print("⚠️ Função name_extractor não encontrada. Retornando DESCONHECIDO.")
            return "DESCONHECIDO"
        return name_extractor_fn(text, motoristas,
                                 nlp=nlp_name,
                                 phrase_matcher=phrase_matcher,
                                 ruler=ruler,
                                 embed_model=emb_model,
                                 embed_matrix=emb_matrix)
    except Exception as e:
        print(f"❌ Erro no extract_name_pipeline: {e}")
        traceback.print_exc()
        return "DESCONHECIDO"