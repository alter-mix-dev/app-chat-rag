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

# --- CONFIGURACIÓN DE SEGURIDAD PARA COLAB ---
# Como en Colab no tienes la interfaz de Streamlit Cloud para los secretos, 
# puedes pegar tu API key directamente aquí de forma temporal para probar:
GROQ_API_KEY = "api-jgc"

st.set_page_config(page_title="Asistente RAG", layout="wide")
st.title("🦙 Asistente Virtual en Colab")

# Inicializar modelos con la clave directa
llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, groq_api_key=GROQ_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

with st.sidebar:
    st.header("📁 Base de Conocimiento")
    uploaded_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])
    if uploaded_file is not None:
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        with st.spinner("Procesando..."):
            try:
                loader = PyPDFLoader(temp_file_path)
                splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
                st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                st.success("¡Listo!")
            except Exception as e: st.error(f"Error: {e}")
            finally: 
                if os.path.exists(temp_file_path): os.remove(temp_file_path)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]): st.write(message["content"])

user_query = st.chat_input("Hazme una pregunta...")
if user_query:
    with st.chat_message("user"): st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    if st.session_state.vector_store is None:
        response_text = "Sube un PDF primero."
    else:
        with st.spinner("Pensando..."):
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            prompt = ChatPromptTemplate.from_template("Responde basándote en el contexto:\nContexto:\n{context}\nPregunta:\n{question}")
            rag_chain = ({"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
            response_text = rag_chain.invoke(user_query)
            
    with st.chat_message("assistant"): st.write(response_text)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
Usa el código con precaución.Celda 3: Lanzar la aplicación a internet localmentePara poder ver y abrir la aplicación desde tu computadora o el celular mientras Colab esté encendido, necesitas crear un túnel público gratis usando Localtunnel. Pega esto en la tercera celda y ejecútalo:python# Corre streamlit en el fondo de Colab
!streamlit run app.py &>/dev/null &

# Obtén la dirección IP pública necesaria para desbloquear el túnel
import urllib
print("Tu contraseña de acceso es:", urllib.request.urlopen('https://icanhazip.com').read().decode('utf8').strip())

# Abre el enlace público para abrir tu app de Streamlit
!npx localtunnel --port 8501
#Cuando ejecutes la Celda 3, te dará una dirección IP (Contraseña) y un enlace azul. Al hacer clic en el enlace, te pedirá la contraseña; la pegas y podrás interactuar con tu aplicación en tiempo real.
