import argparse
# CAMBIO: Chroma se importa de su nuevo paquete
from langchain_chroma import Chroma 
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    print("Cargando modelo de embeddings local...")
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    print("Buscando en la base de datos...")
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # CAMBIO CLAVE: Invertimos '<' por '>'.
    # Descartamos resultados si el puntaje (distancia) es MAYOR a 0.7
    if len(results) == 0 or results[0][1] > 0.7: 
        print(f"No se encontraron resultados relevantes para: '{query_text}'")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("\n--- Contexto enviado al modelo ---")
    print(prompt)
    print("----------------------------------\n")

    # LÍNEA NUEVA (CORRECTA)
    model = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",  # <-- ¡USA ESTE NOMBRE DE TU LISTA! 
    google_api_key=os.environ['GOOGLE_API_KEY'],
    temperature=0.7 
    )

    # LÍNEAS NUEVAS (¡LA SOLUCIÓN!)
    response = model.invoke(prompt)

    # Revisa si la respuesta es una lista (el formato feo)
    if isinstance(response.content, list):
        # Saca el texto del primer elemento de la lista
        response_text = response.content[0].get('text', '')
    else:
        # Si no, usa el contenido directamente (es un string)
        response_text = response.content


if __name__ == "__main__":
    main()