import tempfile
import os
import pandas as pd
import streamlit as st

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Configuração da página
st.set_page_config(page_title="ChatBot Personalizado", page_icon="🤖", layout="wide")

# Versão da API fixada internamente para simplificar a interface
API_VERSION = "2024-02-01"

# --- BARRA LATERAL: Configurações e Entradas (Apenas os 3 campos solicitados) ---
with st.sidebar:
    st.header("⚙️ Configurações Azure / Foundry")
    azure_api_key = st.text_input("Azure OpenAI API Key", type="password", help="Sua chave de API")
    azure_endpoint = st.text_input("Azure Endpoint", placeholder="https://seu-recurso.openai.azure.com/")
    azure_deployment = st.text_input("Nome do Deployment / Modelo", placeholder="gpt-4o ou gpt-35-turbo")
    
    st.markdown("---")
    st.header("🎭 Persona da IA")
    system_prompt = st.text_area(
        "Instruções do Sistema (System Prompt)",
        value="Você é um assistente virtual prestativo, preciso e amigável. Responda com base nos documentos fornecidos quando relevante.",
        height=100
    )

    st.markdown("---")
    st.header("📄 Base de Conhecimento (RAG)")
    uploaded_files = st.file_uploader(
        "Upload de arquivos de orientação",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True
    )
    
    btn_iniciar = st.button("🚀 Conectar e Inicializar Bot", use_container_width=True)

# --- GERENCIAMENTO DE ESTADO ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "chat_ready" not in st.session_state:
    st.session_state.chat_ready = False

# Histórico de conversas via Streamlit integration do LangChain
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# --- PROCESSAMENTO DAS ENTRADAS E CONEXÃO ---
if btn_iniciar:
    if not azure_api_key or not azure_endpoint or not azure_deployment:
        st.error("Por favor, preencha a API Key, o Endpoint e o Nome do Deployment.")
    else:
        try:
            with st.spinner("Conectando ao modelo e processando arquivos..."):
                # Inicializa LLM usando a versão fixa da API
                st.session_state.llm = AzureChatOpenAI(
                    azure_endpoint=azure_endpoint,
                    azure_deployment=azure_deployment,
                    openai_api_key=azure_api_key,
                    api_version=API_VERSION,
                    temperature=0.7
                )
                
                # Processamento RAG (Arquivos)
                documents = []
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        ext = uploaded_file.name.split(".")[-1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        try:
                            if ext == "pdf":
                                loader = PyPDFLoader(tmp_path)
                            elif ext == "txt":
                                loader = TextLoader(tmp_path, encoding="utf-8")
                            elif ext == "csv":
                                loader = CSVLoader(tmp_path, encoding="utf-8")
                            
                            documents.extend(loader.load())
                        finally:
                            os.remove(tmp_path)

                if documents:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    splits = text_splitter.split_documents(documents)
                    
                    # Usa modelo de Embeddings da Azure para indexação RAG
                    embeddings = AzureOpenAIEmbeddings(
                        azure_endpoint=azure_endpoint,
                        openai_api_key=azure_api_key,
                        api_version=API_VERSION,
                    )
                    st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                    st.success("Arquivos RAG processados com sucesso!")
                else:
                    st.session_state.vectorstore = None
                    st.info("Nenhum arquivo enviado. O bot responderá sem base de conhecimento externa.")

                st.session_state.chat_ready = True
                msgs.clear()
        except Exception as e:
            st.error(f"Erro ao inicializar conexão: {str(e)}")

# --- INTERFACE PRINCIPAL DO CHAT ---
st.title("🤖 ChatBot Azure OpenAI")

# Botão para Novo Chat / Limpar Histórico
col_title, col_btn = st.columns([4, 1])
with col_btn:
    if st.button("🔄 Novo Chat", use_container_width=True):
        msgs.clear()
        st.rerun()

if not st.session_state.chat_ready:
    st.info("👈 Insira os dados de conexão na barra lateral e clique em **Conectar e Inicializar Bot** para começar.")
else:
    # Exibe o histórico de mensagens
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Captura a entrada do usuário
    if user_input := st.chat_input("Digite sua pergunta..."):
        st.chat_message("user").write(user_input)

        # Processamento da resposta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    # Busca contexto via RAG (se houver vectorstore carregado)
                    context_text = ""
                    if st.session_state.vectorstore:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                        docs = retriever.invoke(user_input)
                        context_text = "\n\n".join([doc.page_content for doc in docs])

                    # Monta o prompt combinando a Persona e o Contexto (RAG)
                    full_system_prompt = system_prompt
                    if context_text:
                        full_system_prompt += f"\n\nContexto dos documentos fornecidos:\n{context_text}"

                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", full_system_prompt),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{input}")
                    ])

                    chain = prompt_template | st.session_state.llm

                    chain_with_history = RunnableWithMessageHistory(
                        chain,
                        lambda session_id: msgs,
                        input_messages_key="input",
                        history_messages_key="history",
                    )

                    response = chain_with_history.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": "default_session"}}
                    )
                    
                    st.write(response.content)

                except Exception as e:
                    st.error(f"Erro ao gerar resposta: {str(e)}")
