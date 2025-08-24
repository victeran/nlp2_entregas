import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import openai

# 1. CLAVES Y MODELOS

# ==================== keys ======================
PINECONE_API_KEY = ""
PINECONE_INDEX = "cv-index"
GROQ_API_KEY =""
# ===============================================================

# Carga modelos/apis solo una vez
@st.cache_resource
def cargar_todo():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    groq = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
    index = pc.Index(PINECONE_INDEX)
    return st_model, groq, index

st_model, groq, index = cargar_todo()

# 2. INTERFAZ
st.title("Consultas a personalizadas a mi CV")
user_query = st.text_area("Pregunta o consulta:", "")

if st.button("Consultar") and user_query.strip():
    with st.spinner("Buscando respuesta..."):
        # EMBEDDING + RETRIEVAL
        q_vector = st_model.encode(user_query).tolist()
        response = index.query(vector=q_vector, top_k=5, include_metadata=True)
        contexto = "\n".join([m['metadata']['text'] for m in response['matches']])

        # PROMPT y GENERACIÓN (GROQ)
        prompt = f"""A continuación tienes información extraída de un currículum:

{contexto}

Responde SOLO usando la información de arriba:

Pregunta: {user_query}
Respuesta:"""
        completion = groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de currículums vitae."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.2
        )
        result = completion.choices[0].message.content

        st.markdown("**Respuesta:**")
        st.write(result)
        with st.expander("Ver contexto usado"):
            st.code(contexto)