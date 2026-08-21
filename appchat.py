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
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, groq_api_key=GROQ_API_KEY)
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
            try:
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                
                prompt = ChatPromptTemplate.from_template(
                    "Eres un asistente virtual experto y preciso. Responde estrictamente basado en el contexto provisto.\n"
                    "Si la respuesta no viene explícita en el contexto, di amablemente que no cuentas con esa información.\n\n"
                    "Contexto:\n{context}\n\n"
                    "Pregunta:\n{question}\n\n"
                    "Respuesta en español:"
                )
                
                # Cadena interactiva RAG empleando LCEL (LangChain Expression Language)
                rag_chain = (
                    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()} 
                    | prompt 
                    | llm 
                    | StrOutputParser()
                )
                response_text = rag_chain.invoke(user_query)
            except Exception as e:
                response_text = f"Ocurrió un error al procesar la consulta con el modelo de lenguaje: {e}"
            
    with st.chat_message("assistant"): 
        st.write(response_text)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
