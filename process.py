import os
from utils import extract_text_with_ocrmypdf, preprocess_image_opencv, detect_orientation_tesseract, clean_text, get_num_pages
from extract_name_pipeline import extract_name_pipeline
from pdf2image import convert_from_path
from config import INPUT_FOLDER, POPPLER_PATH, MIN_NAME_LEN
import pytesseract
import traceback
import numpy as np
import cv2
from PIL import Image
from motoristas import motoristas
from extract_tipo import extract_tipo
from date_extractor import extract_final_date
from extract_motivo import extract_motivo
from rename_pdf import rename_pdf
from nlp_loader import load_ner_model
nlp_name = load_ner_model()

def process_pdf(file):
    """
    Processa um PDF: OCR (OCRmyPDF -> fallback) -> aplica ocr_cleaner
    -> usa ocr_table_detector para melhorar o crop (se possível)
    -> aplica postprocessor_fix_words -> extrai tipo/data/nome/motivo -> renomeia.

    Recebe o nome do arquivo (apenas o nome, não o caminho).
    Usa variáveis globais definidas no topo do seu script como INPUT_FOLDER, POPPLER_PATH, etc.
    """
    path = os.path.join(INPUT_FOLDER, file)
    temp_img_path = None

    try:
        print(f"[DEBUG] Iniciando processamento de: {file}", flush=True)
        print(f"\n📄 Processando: {file}")

        # 1) Tenta OCRmyPDF primeiro (função já definida no arquivo)
        text = ""
        try:
            text = extract_text_with_ocrmypdf(path)
            if text:
                print("🔎 Texto obtido via OCRmyPDF.")
        except Exception as e:
            print(f"⚠️ OCRmyPDF falhou: {e}")
        

        # 2) Se OCRmyPDF vazio, tenta o pipeline robusto do seu ocr_cleaner (se disponível)
        oc_cleaner = None
        table_detector = None
        postproc = None
        try:
            import ocr_cleaner as oc_cleaner
        except Exception:
            oc_cleaner = None

        try:
            import ocr_table_detector as table_detector
        except Exception:
            table_detector = None

        try:
            import postprocessor_fix_words as postproc
        except Exception:
            postproc = None

        if not text or not text.strip():
            if oc_cleaner and hasattr(oc_cleaner, "ocr_pdf_clean"):
                try:
                    print("🔁 Tentando OCR com ocr_cleaner.ocr_pdf_clean(...)")
                    text = oc_cleaner.ocr_pdf_clean(path)
                except Exception as e:
                    print(f"⚠️ ocr_cleaner.ocr_pdf_clean falhou: {e}")
            else:
                # fallback: converte páginas e usa pytesseract direto (igual antes)
                try:
                    print("🔁 Fallback: convertendo páginas e usando pytesseract (imagem).")
                    pages = convert_from_path(path, dpi=300, poppler_path=POPPLER_PATH)
                    text_lines = []
                    for p in pages:
                        img_proc = preprocess_image_opencv(p)
                        page_text = pytesseract.image_to_string(img_proc, lang='por')
                        text_lines.append(page_text)
                    text = "\n".join(text_lines)
                except Exception as fe:
                    print(f"❌ Fallback de OCR por imagens falhou: {fe}")
                    traceback.print_exc()
                    text = ""

        # 3) Preparar imagem para detector de tabela / bloco de nome (se disponível)
        # Vamos extrair a primeira página como imagem temporária e chamar process_for_ocr(path)
        try:
            pages = convert_from_path(path, dpi=300, poppler_path=POPPLER_PATH)
            first_page_pil = pages[0] if pages else None
        except Exception:
            first_page_pil = None

        rotated = table_crop = name_crop = None
        if table_detector and first_page_pil is not None:
            try:
                # salva temporariamente a primeira página para que o module use cv2.imread(path)
                import tempfile
                tempf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_img_path = tempf.name
                tempf.close()

                # converter PIL -> BGR e salvar com cv2
                np_img = np.array(first_page_pil)
                # PIL dá RGB; cv2 espera BGR ao salvar — convertemos
                np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(temp_img_path, np_img_bgr)

                rotated, table_crop, name_crop = table_detector.process_for_ocr(temp_img_path)
                print("🔎 Detector de tabela executado com sucesso.")
            except Exception as e:
                print(f"⚠️ ocr_table_detector falhou: {e}")
                rotated = table_crop = name_crop = None

        # 4) Se detector devolveu um crop de nome ou tabela, tentar extrair OCR específico dessas regiões
        text_from_regions = ""
        try:
            if name_crop is not None:
                # name_crop é numpy array (BGR). Converter para PIL e rodar Tesseract
                pil_name = Image.fromarray(cv2.cvtColor(name_crop, cv2.COLOR_BGR2RGB))
                processed_name_img = preprocess_image_opencv(pil_name)
                text_name = pytesseract.image_to_string(processed_name_img, lang='por')
                text_from_regions += ("\n" + text_name) if text_name else ""
            if table_crop is not None:
                pil_table = Image.fromarray(cv2.cvtColor(table_crop, cv2.COLOR_BGR2RGB))
                processed_table_img = preprocess_image_opencv(pil_table)
                text_table = pytesseract.image_to_string(processed_table_img, lang='por')
                text_from_regions += ("\n" + text_table) if text_table else ""
        except Exception as e:
            print(f"[DEBUG] Iniciando processamento de: {filename}", flush=True)
            print(f"⚠️ Extração via regiões detectadas falhou: {e}")

        # 5) Combine textos: prioriza texto das regiões se forem mais longos/úteis
        if text_from_regions and len(text_from_regions.strip()) > len(text.strip()) / 2:
            # usar texto das regiões (quando faz sentido)
            text = text + "\n\n" + text_from_regions

        # 6) Detectar orientação (usa sua função existente). Se não houver página, assume vertical.
        try:
            if first_page_pil is not None:
                orientation = detect_orientation_tesseract(first_page_pil)
            else:
                orientation = "vertical"
        except Exception:
            orientation = "vertical"

        # 7) Limpeza com ocr_cleaner.clean_ocr_text (se disponível)
        try:
            if oc_cleaner and hasattr(oc_cleaner, "clean_ocr_text"):
                text = oc_cleaner.clean_ocr_text(text)
            else:
                # sua função local clean_text já existe — usamos como fallback leve
                text = clean_text(text)
        except Exception as e:
            print(f"⚠️ Falha ao aplicar ocr_cleaner.clean_ocr_text: {e}")

        # 8) Pós-processamento de palavras (postprocessor_fix_words.postprocess_ocr), se disponível
        try:
            if postproc and hasattr(postproc, "postprocess_ocr"):
                # postprocess_ocr pode aceitar lista motoristas para alinhamento de nomes
                text_post = postproc.postprocess_ocr(text, motoristas)
                if text_post:
                    text = text_post
        except Exception as e:
            print(f"⚠️ Falha no postprocessor_fix_words.postprocess_ocr: {e}")

        # 9) Número de páginas (função já presente)
        num_pages = get_num_pages(path)

        # 10) Extrair tipo/data/nome/motivo
        tipo = extract_tipo(text, orientation, num_pages)
        date = extract_final_date(
            text,
            tipo,
            ocr_image_path=temp_img_path
        )

        # Para extração de nome: se postprocessor devolveu um nome exato (alguns postprocessors retornam orig),
        # preferimos validar com seu name_extractor pipeline. Se texto contém nome isolado (postproc returned),
        # tentamos usar extract_name_pipeline; caso retorne DESCONHECIDO, tentamos heurística rápida.
        nome = extract_name_pipeline(text)
        if (not nome or nome == "DESCONHECIDO") and postproc and hasattr(postproc, "postprocess_ocr"):
            # postproc.postprocess_ocr pode ter retornado um nome exato — já aplicado acima.
            # Se ainda desconhecido, vamos tentar buscar palavras capitalizadas no top region (se disponível).
            try:
                if name_crop is not None:
                    pil_name = Image.fromarray(cv2.cvtColor(name_crop, cv2.COLOR_BGR2RGB))
                    small_text = pytesseract.image_to_string(preprocess_image_opencv(pil_name), lang='por')
                    small_text = small_text.strip()
                    if small_text and len(small_text) > MIN_NAME_LEN:
                        candidate = postproc.postprocess_ocr(small_text, motoristas) if postproc else small_text
                        if candidate and candidate != "":
                            nome = candidate
            except Exception:
                pass

        motivo = extract_motivo(text) 

        print(f"➡ Tipo: {tipo} | Data: {date} | Nome: {nome} | Motivo: {motivo}")

        # 11) Renomear arquivo
        rename_pdf(path, tipo, nome, date, motivo)

    except Exception as e:
        print(f"❌ Erro ao processar {file}: {e}")
        traceback.print_exc()

    finally:
        # remover arquivo temporário de imagem se criado
        try:
            if temp_img_path and os.path.exists(temp_img_path):
                os.remove(temp_img_path)
        except Exception:
            pass