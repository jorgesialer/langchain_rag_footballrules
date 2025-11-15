# API de RAG para Consultar Reglas de Fútbol

Este es un proyecto personal para explorar el stack de IA moderno, enfocado en construir un sistema Q&A (Preguntas y Respuestas). El objetivo principal fue aprender e implementar una arquitectura RAG (Retrieval-Augmented Generation) y exponerla a través de una API REST.

El proyecto responde preguntas sobre el reglamento oficial de fútbol ("Laws of the Game 2025/26").

## Inspiración y Referencias

La lógica base del RAG fue adaptada del tutorial de **Pixegami**. Este proyecto expande esa base en varios puntos clave:
* Se reemplazaron los embeddings de OpenAI por un modelo local de HuggingFace.
* Se cambió el modelo de chat de OpenAI por la API de Google Gemini.
* Se "envolvió" toda la lógica de consulta en una API REST usando FastAPI.

* **Video Tutorial Original:** [RAG + Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch/tcqEUSNCn8I)
* **Repositorio Original:** [github.com/pixegami/langchain-rag-tutorial](https://github.com/pixegami/langchain-rag-tutorial)

---

## Arquitectura del Proyecto

El sistema funciona en dos fases: indexación y consulta.

1.  **Indexación (`create_db.py`):**
    * Un script lee un PDF (`Laws of the Game 2025_26.pdf`).
    * El texto se divide en "chunks" (trozos) de 1000 caracteres.
    * Se usa un modelo local de HuggingFace (`all-MiniLM-L6-v2`) para crear embeddings (vectores) de cada chunk.
    * Los vectores se guardan en una base de datos vectorial **ChromaDB** en la carpeta local `/chroma`.

2.  **API de Consulta (`main.py`):**
    * Se levanta un servidor **FastAPI** que expone un endpoint `/preguntar`.
    * Cuando el endpoint recibe una pregunta, la convierte en un vector usando el mismo modelo de HuggingFace.
    * Realiza una búsqueda de similitud en `ChromaDB` para encontrar los 3 chunks más relevantes.
    * Construye un prompt con la pregunta del usuario y el contexto encontrado.
    * Llama a la API de **Google Gemini** (`gemini-flash-latest`) para generar una respuesta basada solo en ese contexto.
    * La API devuelve la respuesta y las fuentes en formato JSON.

---

## Cómo ejecutar el proyecto

### Requisitos

* Tener un archivo `.env` en la raíz con una clave de API de Google:
    ```
    GOOGLE_API_KEY="AIzaSy..."
    ```
* Instalar las dependencias (se recomienda usar un entorno virtual):
    ```bash
    pip install -r requirements.txt
    ```

### 1. Crear la Base de Datos
(Este paso solo se hace una vez)

```bash
python create_db.py

Esto leerá el PDF en /data y creará la base de datos en la carpeta /chroma.

2. Correr la API
Bash

python main.py
El servidor se iniciará en http://127.0.0.1:8000.

3. Probar la API
Se puede usar la documentación interactiva de FastAPI.

Abre tu navegador y ve a: http://127.0.0.1:8000/docs

Haz clic en el endpoint /preguntar y en "Try it out".

Escribe tu pregunta en el "Request body":

{
  "texto": "What is an offside offence?"
}