import spacy
from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler

def build_spacy_matchers(nlp, motoristas):
    """
    Espera motoristas no formato:
    [ ["123", "CLAYTON NUNES"], ["456", "JOSE SILVA"], ... ]
    Retorna (phrase_matcher, entity_ruler)
    """

    # transformar apenas o nome em padrões
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(nome) for codigo, nome in motoristas]

    if patterns:
        phrase_matcher.add("MOTORISTA", patterns)

    # EntityRuler
    if "ner" in nlp.pipe_names:
        entity_ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        entity_ruler = nlp.add_pipe("entity_ruler")

    rules = []

    for codigo, nome in motoristas:
        rules.append({"label": "NOME", "pattern": nome})

        partes = nome.split()
        if len(partes) >= 2:
            rules.append({"label": "NOME", "pattern": f"sr. {partes[0]} {partes[-1]}"})
            rules.append({"label": "NOME", "pattern": f"sra. {partes[0]} {partes[-1]}"})
    
    if rules:
        entity_ruler.add_patterns(rules)

    return phrase_matcher, entity_ruler
