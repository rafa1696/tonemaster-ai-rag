import os
import io
import streamlit as st
from dotenv import load_dotenv

from loader import load_csv_document, load_pdf_document, create_vectorstore
from agent import build_qa_chain, query_agent

# Carrega variáveis de ambiente do .env
load_dotenv()

# Configuração da página Streamlit
st.set_page_config(
    page_title="Agente IA - Leitor de PDFs e CSVs",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização em CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6, #10B981, #F59E0B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .doc-badge {
        background-color: #1F2937;
        color: #F3F4F6;
        border: 1px solid #374151;
        padding: 0.4rem 0.75rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "loaded_docs_info" not in st.session_state:
    st.session_state.loaded_docs_info = []

if "last_failed_prompt" not in st.session_state:
    st.session_state.last_prompt = None

# Header Principal
st.markdown('<div class="main-title">🤖 Agente IA Multiprovedor - Leitor de PDFs & CSVs</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Consulte manuais e dados em tabelas utilizando modelos Groq, Google Gemini ou OpenAI.</div>', unsafe_allow_html=True)

# Sidebar - Seleção do Provedor e API Keys
with st.sidebar:
    st.header("⚙️ Provedor & Modelo de IA")
    
    provider = st.selectbox(
        "Selecione o Provedor de LLM:",
        options=["Groq", "Google Gemini", "OpenAI"],
        index=0
    )
    
    # Modelos e API Keys por provedor
    if provider == "Groq":
        model_options = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768", # Descontinuado
            "gemma2-9b-it"
        ]
        selected_model = st.selectbox("Modelo Groq:", model_options)
        env_key = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input("Groq API Key:", value=env_key, type="password", help="Obtenha gratuitamente em console.groq.com")
        
    elif provider == "Google Gemini":
        model_options = [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite"
        ]
        selected_model = st.selectbox("Modelo Gemini:", model_options)
        env_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
        api_key = st.text_input("Google API Key:", value=env_key, type="password", help="Obtenha no Google AI Studio")
        
    else:  # OpenAI
        model_options = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo"
        ]
        selected_model = st.selectbox("Modelo OpenAI:", model_options)
        env_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key:", value=env_key, type="password")

    st.divider()
    st.subheader("🧠 Provedor de Embeddings")
    
    emb_provider = st.selectbox(
        "Selecione o modelo para Embeddings (FAISS):",
        options=["HuggingFace (Local / Gratuito)", "Google Gemini Embeddings", "OpenAI Embeddings"],
        index=0 if provider == "Groq" else (1 if provider == "Google Gemini" else 2),
        help="HuggingFace roda localmente sem exigir API key de embeddings, ideal para usar junto com o Groq!"
    )
    
    emb_provider_code = "huggingface"
    if "Google" in emb_provider:
        emb_provider_code = "google"
    elif "OpenAI" in emb_provider:
        emb_provider_code = "openai"

    st.divider()
    st.subheader("📁 Documentos de Referência")
    
    default_docs_dir = os.path.join(os.path.dirname(__file__), "documents")
    found_default_files = []
    if os.path.exists(default_docs_dir):
        for f in os.listdir(default_docs_dir):
            if f.endswith(".csv") or f.endswith(".pdf"):
                found_default_files.append(os.path.join(default_docs_dir, f))
    
    use_default_files = False
    if found_default_files:
        use_default_files = st.checkbox("Usar arquivos da pasta `documents/`", value=True)
        for fpath in found_default_files:
            fname = os.path.basename(fpath)
            ftype = "📊 CSV" if fname.endswith(".csv") else "📄 PDF"
            st.markdown(f'<div class="doc-badge"><span>{ftype} {fname}</span></div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload de novos arquivos (PDF ou CSV):",
        type=["pdf", "csv"],
        accept_multiple_files=True
    )

    btn_index = st.button("🔄 Processar & Indexar Documentos", use_container_width=True, type="primary")

    if btn_index:
        if not api_key and provider != "HuggingFace":
            st.error(f"Por favor, forneça uma API Key válida para o provedor {provider}.")
        else:
            with st.spinner(f"Processando documentos e gerando embeddings ({emb_provider})..."):
                all_documents = []
                docs_summary = []
                
                # Documentos padrão
                if use_default_files and found_default_files:
                    for fpath in found_default_files:
                        fname = os.path.basename(fpath)
                        if fname.endswith(".csv"):
                            docs = load_csv_document(fpath, filename=fname)
                            all_documents.extend(docs)
                            docs_summary.append(f"📊 {fname} ({len(docs)} registros)")
                        elif fname.endswith(".pdf"):
                            docs = load_pdf_document(fpath, filename=fname)
                            all_documents.extend(docs)
                            docs_summary.append(f"📄 {fname} ({len(docs)} trechos)")

                # Documentos enviados
                if uploaded_files:
                    for up_file in uploaded_files:
                        fname = up_file.name
                        bytes_data = io.BytesIO(up_file.read())
                        if fname.endswith(".csv"):
                            docs = load_csv_document(bytes_data, filename=fname)
                            all_documents.extend(docs)
                            docs_summary.append(f"📊 {fname} ({len(docs)} registros)")
                        elif fname.endswith(".pdf"):
                            docs = load_pdf_document(bytes_data, filename=fname)
                            all_documents.extend(docs)
                            docs_summary.append(f"📄 {fname} ({len(docs)} trechos)")

                if not all_documents:
                    st.warning("Nenhum arquivo válido foi encontrado ou enviado.")
                else:
                    try:
                        vectorstore = create_vectorstore(
                            documents=all_documents,
                            embedding_provider=emb_provider_code,
                            api_key=api_key
                        )
                        rag_chain = build_qa_chain(
                            vectorstore=vectorstore,
                            provider=provider.lower().replace("google ", ""),
                            api_key=api_key,
                            model_name=selected_model
                        )
                        
                        st.session_state.vectorstore = vectorstore
                        st.session_state.rag_chain = rag_chain
                        st.session_state.loaded_docs_info = docs_summary
                        st.success(f"Índice vetorial criado com sucesso via {emb_provider}! ({len(all_documents)} trechos)")
                    except Exception as e:
                        st.error(f"Erro ao criar índice vetorial: {e}")

    if st.session_state.loaded_docs_info:
        st.divider()
        st.caption("📌 **Status da Indexação Atual:**")
        for item in st.session_state.loaded_docs_info:
            st.markdown(f"- {item}")

# Painel Principal - Área de Chat
if st.session_state.rag_chain is None:
    st.info("👋 **Pronto para começar!** Escolha o provedor de IA na barra lateral (Groq, Gemini ou OpenAI) e clique em **Processar & Indexar Documentos**.", icon="💡")
    
    st.markdown("### 💡 Perguntas de Exemplo que você pode testar:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- **“Quais guitarras e pedais o David Gilmour usou no álbum The Wall?”**")
        st.markdown("- **“Qual é a afinação principal utilizada pelo Slash e Van Halen?”**")
    with col2:
        st.markdown("- **“Compare os equipamentos do John Frusciante em Californication vs Stadium Arcadium.”**")
        st.markdown("- **“Quais efeitos de modulação Steve Vai utiliza?”**")

else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 Fontes Consultadas"):
                    for src in msg["sources"]:
                        st.markdown(f"**Fonte:** `{src['source']}` | **Tipo:** `{src['type']}`")
                        st.caption(src["content"])

    prompt = st.chat_input("Pergunte sobre os equipamentos, timbres ou manuais dos documentos..")

    is_retry = False
    if not prompt and st.session_state.get("retry_prompt"):
        prompt = st.session_state.last_failed_prompt
        st.session_state.retry_prompt = False
        is_retry = True
    
    if prompt:
        if not is_retry:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Consultando documentos com {provider} ({selected_model})..."):
                try:
                    current_rag_chain = build_qa_chain(
                        vectorstore=st.session_state.vectorstore, 
                        provider=provider.lower().replace("google ", ""), 
                        api_key=api_key, model_name=selected_model)

                    result = query_agent(
                        rag_chain=current_rag_chain,
                        question=prompt,
                        chat_history=st.session_state.chat_history
                    )
                    
                    answer = result["answer"]
                    sources = result["context"]
                    
                    st.markdown(answer)
                    
                    formatted_sources = []
                    if sources:
                        with st.expander("🔍 Fontes Consultadas"):
                            for doc in sources:
                                src_name = doc.metadata.get("source", "Desconhecido")
                                src_type = doc.metadata.get("type", "desconhecido")
                                snippet = doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
                                
                                formatted_sources.append({
                                    "source": src_name,
                                    "type": src_type,
                                    "content": snippet
                                })
                                st.markdown(f"**Fonte:** `{src_name}` | **Tipo:** `{src_type}`")
                                st.caption(snippet)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": formatted_sources
                    })
                    st.session_state.chat_history.append((prompt, answer))

                    st.session_state.last_failed_prompt = None
                    st.session_state.last_error_msg = None

                except Exception as e:
                    st.session_state.last_failed_prompt = prompt

                    st.error(f"Erro ao consultar modelo")
                    st.session_state.last_error_msg = e

    if st.session_state.get("last_failed_prompt"):
        st.warning(f" A consulta anterior falhou: {st.session_state.last_error_msg}")

        if st.button("🔄 Tentar novamente com o modelo selecionado", key="btn_retry"):
            st.session_state.retry_prompt = True
            st.rerun()