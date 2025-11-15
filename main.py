import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma 
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------------------------------------------
# 1. OPTIMIZACIÓN: Cargar los modelos UNA SOLA VEZ
# -----------------------------------------------------------------

print("Cargando variables de entorno...")
load_dotenv()

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

print("Cargando modelo de embeddings (HuggingFace)... Esto puede tardar un momento.")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Cargando base de datos Chroma...")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

print("Cargando modelo de Chat (Google Gemini)...")
model = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest", 
    google_api_key=os.environ['GOOGLE_API_KEY'],
    temperature=0.7
)
print("✅ ¡Servidor y modelos listos!")

# -----------------------------------------------------------------
# 2. INICIALIZAR LA APLICACIÓN FastAPI
# -----------------------------------------------------------------
app = FastAPI(
    title="Agente RAG de Reglas de Fútbol",
    description="Una API para responder preguntas sobre el reglamento de fútbol usando RAG."
)

# -----------------------------------------------------------------
# 3. DEFINIR LOS FORMATOS DE PREGUNTA Y RESPUESTA
# -----------------------------------------------------------------
# Esto le dice a FastAPI cómo debe verse el JSON que entra y el que sale.

class QueryRequest(BaseModel):
    texto: str  # El usuario debe enviar un JSON como: {"texto": "..."}

class QueryResponse(BaseModel):
    respuesta: str
    fuentes: list[str] # Devolveremos un JSON como: {"respuesta": "...", "fuentes": [...]}

# -----------------------------------------------------------------
# 4. CREAR EL "ENDPOINT" (LA URL DE LA API)
# -----------------------------------------------------------------

@app.post("/preguntar", response_model=QueryResponse)
def preguntar(query: QueryRequest):
    """
    Recibe una pregunta y devuelve una respuesta usando RAG.
    """
    query_text = query.texto

    # 1. Buscar en la DB (Igual que en query_data.py)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0 or results[0][1] > 0.7:
        return QueryResponse(
            respuesta=f"No se encontraron resultados relevantes para: '{query_text}'",
            fuentes=[]
        )

    # 2. Construir el contexto y el prompt (Igual que en query_data.py)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # 3. Generar la respuesta (Igual que en query_data.py)
    response = model.invoke(prompt)
    
    if isinstance(response.content, list):
        # Saca el texto del primer elemento de la lista
        response_text = response.content[0].get('text', 'Error: No se pudo extraer texto')
    else:
        # Si no, usa el contenido directamente (es un string)
        response_text = response.content
    # ----------------------------------------------
        
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    
    # 4. Devolver la respuesta en formato JSON (QueryResponse)
    return QueryResponse(
        respuesta=response_text,
        fuentes=sources
    )

# -----------------------------------------------------------------
# 5. CÓMO CORRER EL SERVIDOR 
# -----------------------------------------------------------------

if __name__ == "__main__":
    # Inicia el servidor en el puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
