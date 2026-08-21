import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURACIÓN DE SEGURIDAD REPARADA PARA LA NUBE ---
# Validamos si estamos en local/Colab o en Streamlit Cloud
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    # Clave temporal por si lo corres en Colab (reemplaza por tu 'gsk_...' real si pruebas en Colab)
    GROQ_API_KEY = "api-jgc" 

st.set_page_config(page_title="Asistente RAG", layout="wide")
st.title("🦙 Asistente Virtual de JGC")

# Validar que la API Key no sea la de ejemplo antes de llamar al modelo
if GROQ_API_KEY == "api-jgc":
    st.warning("⚠️ Usando clave de ejemplo. Si estás en la nube, asegúrate de configurar GROQ_API_KEY en Advanced Settings -> Secrets.")

# --- CAMBIO IMPORTANTE: Usamos un modelo oficial, activo y gratuito de Groq ---
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, groq_api_key=GROQ_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

with st.sidebar:
    st.header("📁 Base de Conocimiento")
    uploaded_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])
    if uploaded_file is not None:
        temp_file_path = f"temp_{uploaded_file.name}"
        
        # Escribir el archivo y asegurar el cierre del flujo de datos
        with open(temp_file_path, "wb") as f: 
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Procesando y segmentando documento..."):
            try:
                # Carga del PDF de forma segura
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                
                # Segmentación del texto en fragmentos
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # Construcción del índice vectorial indexado en memoria
                st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                st.success("¡Documento procesado con éxito!")
            except Exception as e: 
                st.error(f"Error al procesar el PDF: {e}")
            finally: 
                # Eliminación del residuo temporal en el servidor de la nube
                if os.path.exists(temp_file_path): 
                    os.remove(temp_file_path)

# Mostrar el historial del chat de forma limpia
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]): 
        st.write(message["content"])

user_query = st.chat_input("Hazme una pregunta sobre tus documentos...")
if user_query:
    with st.chat_message("user"): 
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    if st.session_state.vector_store is None:
        response_text = "Por favor, primero sube un archivo PDF en la barra lateral para poder extraer el contexto."
    else:
        with st.spinner("Buscando en la base de datos y redactando respuesta..."):
            with st.spinner("Buscando en la base de datos y redactando respuesta..."):
            try:
                # 1. Recuperación inteligente de fragmentos
                # Traemos los 4 más relevantes, pero si se pregunta por el título o inicio, incluimos los primeros chunks del PDF
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                docs_retrieved = retriever.invoke(user_query)
                
                # Respaldo: Si la pregunta es global o sobre el título, inyectamos los fragmentos iniciales de la primera página
                queries_globales = ["titulo", "title", "de que trata", "resumen", "articulo", "quien escribe"]
                if any(q in user_query.lower() for q in queries_globales):
                    # Obtenemos algunos fragmentos del principio asegurando que la portada esté presente
                    todos_los_docs = st.session_state.vector_store.similarity_search("", k=3)
                    docs_retrieved = list(set(docs_retrieved + todos_los_docs))

                contexto_final = "\n\n".join(d.page_content for d in docs_retrieved)
                
                # 2. PROMPT OPTIMIZADO CON PERSONALIDAD EMPÁTICA
                prompt = ChatPromptTemplate.from_template(
                    "Eres un asistente virtual empático, experto y altamente resolutivo. Tu objetivo es ayudar al usuario de forma clara y directa.\n"
                    "Analiza con atención el contexto provisto del documento PDF. Si te preguntan por el título, autor o resumen general, deduce la respuesta analizando la información de las primeras páginas que viene en el contexto.\n"
                    "Evita dar respuestas robóticas o negativas a menos que sea estrictamente imposible deducirlo.\n\n"
                    "Contexto del PDF:\n{context}\n\n"
                    "Pregunta del usuario:\n{question}\n\n"
                    "Respuesta analítica en español:"
                )
                
                # Cadena interactiva ejecutada directamente con el contexto adaptativo
                rag_chain = prompt | llm | StrOutputParser()
                response_text = rag_chain.invoke({"context": contexto_final, "question": user_query})
                
            except Exception as e:
                response_text = f"Ocurrió un error al procesar la consulta con el modelo de lenguaje: {e}"
