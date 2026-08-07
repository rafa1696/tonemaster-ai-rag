import io
import os
import pandas as pd
from typing import List, Union, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
import streamlit as st

# O @st.cache_resource faz com que o modelo seja carregado na memória apenas uma vez,
# otimizando o desempenho da aplicação.
@st.cache_resource

# Imports dinâmicos/condicionais para os modelos de embeddings
def get_embeddings_instance(provider: str = "huggingface", api_key: Optional[str] = None):
    """
    Retorna a instância de Embeddings selecionada pelo usuário.
    Provedores suportados: 'huggingface' (gratuito/local), 'google' (Gemini), 'openai'.
    """
    provider = provider.lower()
    
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY")
        )
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    else:
        # Padrão: HuggingFace (Local / Gratuito - excelente para usar junto com Groq)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )


def load_csv_document(source: Union[str, io.BytesIO], filename: str = "dados.csv") -> List[Document]:
    """
    Carrega e processa um arquivo CSV convertendo cada linha em um Document contendo
    uma representação em texto estruturado e metadados associados.
    """
    df = pd.read_csv(source)
    
    documents = []
    for idx, row in df.iterrows():
        metadata = {
            "source": filename,
            "row": idx + 1,
            "type": "csv"
        }

        content_lines = []

        for col in df.columns:
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                content_lines.append(f"• **{col}**: {val}")
                metadata[col.lower().replace(" ", "_")] = str(val)
        
        page_content = f"### [Registro CSV - Linha {idx + 1} ({filename})]\n" + "\n".join(content_lines)
        
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    return documents


def load_pdf_document(source: Union[str, io.BytesIO], filename: str = "documento.pdf") -> List[Document]:
    """
    Carrega e extrai texto de um arquivo PDF (via caminho de arquivo ou buffer de memória).
    """
    raw_documents = []
    
    if isinstance(source, str):
        loader = PyPDFLoader(source)
        raw_documents = loader.load()
        for doc in raw_documents:
            doc.metadata["source"] = filename
            doc.metadata["type"] = "pdf"
    else:
        reader = PdfReader(source)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                raw_documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": idx + 1,
                            "type": "pdf"
                        }
                    )
                )
                
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(raw_documents)
    return split_docs


def create_vectorstore(documents: List[Document], embedding_provider: str = "huggingface", api_key: Optional[str] = None) -> FAISS:
    """
    Gera um índice FAISS a partir dos documentos processados utilizando o provedor de embeddings especificado.
    """
    embeddings = get_embeddings_instance(provider=embedding_provider, api_key=api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
