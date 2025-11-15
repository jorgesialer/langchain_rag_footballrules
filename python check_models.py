import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carga tu clave de API desde el archivo .env
load_dotenv()
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("¡Error! No se encontró 'GOOGLE_API_KEY' en tu archivo .env")
    exit()

print("Buscando todos los modelos disponibles para tu clave de API...\n")

# Llama a la función ListModels
try:
    for m in genai.list_models():
        # Verificamos si el modelo soporta el método 'generateContent'
        if 'generateContent' in m.supported_generation_methods:
            print("---------------------------------------------")
            print(f"✅ Modelo DISPONIBLE: {m.name}")
            print(f"    Descripción: {m.description}")
            print(f"    Métodos Soportados: {m.supported_generation_methods}")
            
except Exception as e:
    print(f"\nSe produjo un error al contactar la API de Google:")
    print(e)
    print("\nPor favor, verifica que tu GOOGLE_API_KEY en el archivo .env sea correcta.")

print("\n---------------------------------------------")
print("\nRevisa la lista de arriba 👆. Tu script 'query_data.py' debe usar uno de los nombres de modelo que dicen '✅ Modelo DISPONIBLE'.")
print("El nombre correcto es probablemente 'models/gemini-1.0-pro'.")