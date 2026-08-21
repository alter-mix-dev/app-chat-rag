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
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = "api-jgc" 

st.set_page_config(page_title="Asistente RAG", layout="wide")
st.title("🦙 Asistente Virtual de JGC")

if GROQ_API_KEY == "api-jgc":
    st.warning("⚠️ Usando clave de ejemplo. Si estás en la nube, asegúrate de configurar GROQ_API_KEY en Advanced Settings -> Secrets.")

# Inicializar modelos activos en Groq
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, groq_api_key=GROQ_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Inicializar estados de sesión
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "inicio_documento" not in st.session_state: st.session_state.inicio_documento = ""

with st.sidebar:
    st.header("📁 Base de Conocimiento")
    uploaded_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])
    if uploaded_file is not None:
        temp_file_path = f"temp_{uploaded_file.name}"
        
        with open(temp_file_path, "wb") as f: 
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Procesando y segmentando documento..."):
            try:
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                
                # REPARACIÓN GLOBAL: Guardamos el texto de las primeras páginas para el título
                texto_inicial = ""
                for i in range(min(2, len(docs))):
                    texto_inicial += docs[i].page_content + "\n"
                st.session_state.inicio_documento = texto_inicial
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                st.success("¡Documento procesado con éxito!")
            except Exception as e: 
                st.error(f"Error al procesar el PDF: {e}")
            finally: 
                if os.path.exists(temp_file_path): 
                    os.remove(temp_file_path)

# Mostrar el historial
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
            try:
                # Buscamos los fragmentos más parecidos
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                docs = retriever.invoke(user_query)
                contexto_busqueda = "\n\n".join(d.page_content for d in docs)
                
                # REPARACIÓN DE CONTEXTO: Sumamos siempre la portada al contexto analizado
                contexto_completo = f"--- INICIO/PORTADA DEL DOCUMENTO ---\n{st.session_state.inicio_documento}\n\n--- FRAGMENTOS RELEVANTES ---\n{contexto_busqueda}"
                
                prompt = ChatPromptTemplate.from_template(
                    "Eres un asistente virtual experto, empático y preciso. Responde de forma fluida y natural en español.\n"
                    "Utiliza la sección 'INICIO/PORTADA DEL DOCUMENTO' si el usuario te pregunta por el título, autores, revista o de qué trata el archivo en general.\n"
                    "Si la información no puede deducirse de ninguna parte del contexto, indícalo amablemente.\n\n"
                    "Contexto disponible:\n{context}\n\n"
                    "Pregunta del usuario:\n{question}\n\n"
                    "Respuesta analítica en español:"
                )
                
                rag_chain = prompt | llm | StrOutputParser()
                response_text = rag_chain.invoke({"context": contexto_completo, "question": user_query})
            except Exception as e:
                response_text = f"Ocurrió un error al procesar la consulta con el modelo de lenguaje: {e}"
            
    with st.chat_message("assistant"): 
        st.write(response_text)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

