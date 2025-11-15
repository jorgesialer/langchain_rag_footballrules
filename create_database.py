from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    print("Cargando documentos desde la carpeta 'data'...")
    # Esto le dice a DirectoryLoader que USE PyPDFLoader para los PDF.
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print(f"¡Error! No se encontraron PDFs en la carpeta '{DATA_PATH}'.")
        print("Asegúrate de que 'Laws of the Game 2025_26.pdf' esté ahí.")
        return []

    print(f"Se cargaron {len(documents)} documento(s) PDF.")
    return documents


def split_text(documents: list[Document]):
    print("Dividiendo documentos en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    
    if chunks:
        print("\n--- Ejemplo de Chunk ---")
        # Mostramos el primer chunk como ejemplo
        document = chunks[0]
        print(document.page_content)
        print(document.metadata)
        print("--------------------------\n")

    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    print("Inicializando modelo de embeddings local (HuggingFace)...")
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    print(f"Creando base de datos Chroma con {len(chunks)} chunks (usando modelo local)...")
    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH 
    )
    db.persist()
    print(f"Se guardaron {len(chunks)} chunks en {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
