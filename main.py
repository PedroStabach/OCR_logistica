import os

# ---------------- CONFIGURAÇÕES (ajuste conforme seu ambiente) ----------------
from config import INPUT_FOLDER

# ---------------- PROCESSAMENTO ----------------
from process import process_pdf

# ---------------- MAIN ----------------
import multiprocessing

def main():
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("Nenhum PDF encontrado em:", INPUT_FOLDER)
        return

    print(f"\n🚀 Iniciando processamento EM PARALELO usando todos os núcleos...\n")

    num_cpus = multiprocessing.cpu_count()
    print(f"🧠 Núcleos detectados: {num_cpus}")

    # Adaptar para nuvem
    multiprocessing.freeze_support()  # Necessário no Windows

    # Pool recebe como argumento a função e a lista de PDFs
    with multiprocessing.Pool(processes=num_cpus) as pool:
        pool.map(process_pdf, pdf_files)

    print("\n✅ Processamento concluído (Paralelo)!")


if __name__ == "__main__":
    main()
