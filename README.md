# 📄 OCR_Logistica

Processamento automatizado de PDFs e imagens utilizando **Pytesseract**, formatação inteligente de datas e eventos (Pontos e Advertências) e um **modelo NER treinado com milhares de exemplos** para extrair informações críticas de forma rápida e estruturada.

---

## 🚀 Como utilizar?

1. Instale todas as dependências indicadas no arquivo `config.py`.
2. Instale também, via pip, os seguintes pacotes adicionais:

```bash
pip install multiprocessing os tempfile ocrmypdf numpy cv2 Pillow pytesseract re pypdf unicodedata PyPDF2 pdf2image transformers traceback typing fitz rapidfuzz spacy datetime python-dateutil
```
3. No terminal
```bash
python main.py
```
⚙️ Funcionalidades
1. Otimização de Tempo

Elimine horas de trabalho manual: o script processa grandes volumes de arquivos, padronizando e extrai informações automaticamente.

2. Flexível e Adaptável

Por ser totalmente em Python, você consegue integrar:

Agentes de IA

Fluxos automatizados

Padrões personalizados para diferentes áreas ou regras de negócio

E pode ajustar o comportamento para qualquer estrutura de organização de arquivos.

3. Totalmente Gratuito

A ferramenta utiliza apenas os núcleos do seu computador.
Se quiser, pode subir para a nuvem (Function, VM etc.) para escalar e economizar ainda mais tempo.

4. Desempenho Atual

Tipos: ~90%

Nomes: ~70%

Datas: ~60%

60% dos documentos completos saem perfeitos

O restante exige validação humana, mas o processo já poupa grande parte do trabalho.

Para melhorias, você pode integrar APIs pagas de OCR/NLP (Azure, Google, AWS), elevando drasticamente a precisão.

5. Matriz de Motoristas

Minha lista de motoristas é armazenada como matriz contendo código e nome.
Para adaptar ao seu uso, basta ajustar o código responsável por essa estrutura.

6. Economize Tempo

A intenção do projeto é simples: te libertar de retrabalho manual.
Rápido, personalizável e expansível.
