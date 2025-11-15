# 🤖 API de RAG para Consultar Reglas de Fútbol

Este es un proyecto personal enfocado en explorar el stack moderno de IA para construir un sistema de "Preguntas y Respuestas" (Q&A). El objetivo principal fue aprender e implementar una arquitectura **RAG (Retrieval-Augmented Generation)** y exponerla a través de una **API REST**.

El proyecto responde preguntas sobre el reglamento oficial de fútbol ("Laws of the Game 2025/26").

## 💡 Inspiración y Referencias

La lógica base para la implementación del RAG fue adaptada del excelente tutorial de **Pixegami**. Este proyecto expande esa base al:
1.  Cambiar el modelo de embeddings (de OpenAI a HuggingFace local).
2.  Cambiar el modelo de chat (de OpenAI a Google Gemini).
3.  "Envolver" toda la lógica de consulta en una API de FastAPI.

* **Video Tutorial Original:** [RAG + Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch/tcqEUSNCn8I)
* **Repositorio Original:** [github.com/pixegami/langchain-rag-tutorial](https://github.com/pixegami/langchain-rag-tutorial)

---

## 📋 Descripción del Proyecto

Este proyecto es una API REST que responde preguntas sobre el reglamento de fútbol. Utiliza una arquitectura RAG para asegurar que las respuestas se basen únicamente en el contenido del documento.

### Arquitectura
1.  **Carga de Datos (`create_db.py`):** Un script de Python lee un PDF (`Laws of the Game 2025_26.pdf`), lo divide en "chunks" (trozos) de texto y los procesa.
2.  **Embeddings (Local):** Se utiliza un modelo de embeddings de HuggingFace (`all-MiniLM-L6-v2`) para convertir cada chunk de texto en vectores numéricos. Esto se ejecuta localmente y es gratuito.
3.  **Base de Datos Vectorial (`Chroma`):** Los vectores se almacenan en una base de datos vectorial local (ChromaDB) para permitir búsquedas de similitud rápidas.
4.  **API REST (`main.py`):** Una API construida con **FastAPI** expone un endpoint `/preguntar`.
5.  **Proceso RAG (en la API):**
    * Cuando la API recibe una pregunta (ej. *"What is an offside?"*), la convierte en un vector usando el mismo modelo de embeddings.
    * Busca en la base de datos `Chroma` los chunks de texto más similares a la pregunta.
    * Construye un prompt que incluye la pregunta del usuario y el contexto encontrado.
    * Envía el prompt a la API de **Google Gemini** (`gemini-flash-latest`) para generar una respuesta.
    * La API devuelve la respuesta y las fuentes (el PDF) en un formato JSON.

---

## 🚀 Cómo ejecutar el proyecto

Este proyecto está dividido en dos partes: crear la base de datos y correr la API.

### Requisitos

Asegúrate de tener un archivo `.env` en la raíz con tu clave de API de Google:
GOOGLE_API_KEY="AIzaSy..."


Luego, instala las dependencias (se recomienda usar un entorno virtual):
```bash
pip install -r requirements.txt
1. Crear la Base de Datos
(Este paso solo se hace una vez)

Bash

python create_db.py
Esto leerá el PDF en la carpeta /data y creará la base de datos en la carpeta /chroma.

2. Correr la API
Bash

python main.py
El servidor se iniciará en http://127.0.0.1:8000.

3. Probar la API
Puedes usar la documentación interactiva de FastAPI que se genera automáticamente.

Abre tu navegador y ve a: http://127.0.0.1:8000/docs

Haz clic en el endpoint /preguntar y luego en "Try it out".

Escribe tu pregunta en el "Request body":

JSON

{
  "texto": "What is an offside offence?"
}
Presiona "Execute" y verás la respuesta en JSON.