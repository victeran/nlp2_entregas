import streamlit as st
import re
import unicodedata
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import openai

# ==================== keys ======================
PINECONE_API_KEY = ""
GROQ_API_KEY =""
# ===============================================================

# ==================== NODO DECISOR ====================
def normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    return ''.join(c for c in texto if unicodedata.category(c) != 'Mn')

person_patterns = {
    "cv-lara-index": re.compile(r"\blara(\s+rosenberg)?\b|\brosenberg\b"),
    "cv-index": re.compile(r"\bvictoria(\s+teran)?\b|\bteran\b"),
    "cv-claudio-index": re.compile(r"\bclaudio(\s+barril)?\b|\bbarril\b")
}

def decidir_indice(pregunta: str) -> str:
    nq = normalizar(pregunta)
    for idx, pat in person_patterns.items():
        if pat.search(nq):
            return idx
    return "cv-index"  # fallback por defecto → índice de Victoria

# ==================== AGENTE ====================
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = groq.chat.completions.create(
            model="llama3-70b-8192",
            temperature=0.2,
            messages=self.messages
        )
        return completion.choices[0].message.content

# ==================== CARGA DE MODELOS/APIs ====================
@st.cache_resource
def cargar_todo():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
    return pc, st_model, groq_client

pc, st_model, groq = cargar_todo()

# ==================== ACCIÓN DEL AGENTE ====================
def buscar_cv(pregunta: str):
    indice = decidir_indice(pregunta)
    index = pc.Index(indice)
    vector = st_model.encode(pregunta).tolist()
    response = index.query(vector=vector, top_k=5, include_metadata=True)
    contexto = "\n".join([m['metadata']['text'] for m in response['matches']])
    return contexto, indice

# ==================== PROMPT DEL AGENTE ====================
prompt = """
Corres en un ciclo de Pensamiento, Acción, PAUSA, Observación.
Al final del ciclo, das una Respuesta.

Tus acciones disponibles son:

buscar_cv:
Ejemplo: buscar_cv: ¿Qué experiencia laboral tiene Lara?
Devuelve el contexto relevante del CV consultando Pinecone y usará ese contexto para responder.
""".strip()

agente = Agent(system=prompt)


# ==================== INTERFAZ STREAMLIT ====================
st.title("Agente inteligente para consultas sobre CVs")
user_query = st.text_area("Pregunta o consulta:", "")

if st.button("Consultar") and user_query.strip():
    with st.spinner("Pensando..."):

        # Paso 1️⃣ → Buscar contexto según índice detectado
        contexto, indice_usado = buscar_cv(user_query)

        # Paso 2️⃣ → Generar respuesta usando el agente
        respuesta_final = agente(
            f"Observación: {contexto}\n"
            f"Responde la siguiente pregunta usando SOLO la observación:\n"
            f"{user_query}"
        )

        # Paso 3️⃣ → Mostrar solo la respuesta final
        st.markdown("### ✨ Respuesta:")
        st.write(respuesta_final)