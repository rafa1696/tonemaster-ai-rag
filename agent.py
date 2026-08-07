from typing import List, Tuple, Dict, Any, Optional
import os

from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS

def get_llm_instance(provider: str, model_name: Optional[str], api_key: str):
    """
    Instancia o LLM do provedor especificado ('groq', 'gemini', 'openai').
    """
    prov = provider.lower()
    
    if prov == "groq":
        from langchain_groq import ChatGroq
        selected_model = model_name or "llama-3.3-70b-versatile"
        return ChatGroq(
            model_name=selected_model,
            groq_api_key=api_key or os.getenv("GROQ_API_KEY"),
            temperature=0.2
        )
    elif prov in ["gemini", "google"]:
        from langchain_google_genai import ChatGoogleGenerativeAI
        selected_model = model_name or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )
    elif prov == "openai":
        from langchain_openai import ChatOpenAI
        selected_model = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=selected_model,
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.2
        )
    else:
        raise ValueError(f"Provedor de LLM inválido ou não suportado: {provider}")


def build_qa_chain(vectorstore: FAISS, provider: str, api_key: str, model_name: Optional[str] = None):
    """
    Cria a cadeia RAG ciente do histórico de conversa utilizando Groq, Gemini ou OpenAI.
    """
    llm = get_llm_instance(provider=provider, model_name=model_name, api_key=api_key)
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
    )
    
    contextualize_q_system_prompt = (
        "Dada uma história de conversa e a última pergunta do usuário, "
        "que pode fazer referência ao contexto da conversa, formule uma pergunta independente "
        "que possa ser entendida sem o histórico da conversa. NÃO responda à pergunta, "
        "apenas reformule-a se necessário e, caso contrário, retorne-a como está."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    system_prompt = (
        "Você é um assistente especialista em análise de documentos técnicos, manuais e tabelas de equipamentos de áudio/guitarras.\n"
        "Responda à pergunta do usuário utilizando prioritariamente o contexto fornecido abaixo (que inclui dados de PDFs e linhas de CSVs).\n\n"
        "Instruções:\n"
        "1. Se a informação estiver no contexto, forneça uma resposta precisa, detalhada e bem formatada em Markdown.\n"
        "2. Indique claramente os arquivos, modelos ou referências relevantes no seu texto quando aplicável.\n"
        "3. Se a informação não constar nos documentos carregados, esclareça de forma cordial.\n"
        "4. Mantenha um tom profissional e prestativo.\n\n"
        "Contexto:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


def query_agent(rag_chain, question: str, chat_history: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Executa uma consulta na cadeia RAG formatando o histórico de chat e retornando a resposta
    junto com os documentos de origem (fontes).
    """
    formatted_history = []
    for user_msg, ai_msg in chat_history:
        formatted_history.append(HumanMessage(content=user_msg))
        formatted_history.append(AIMessage(content=ai_msg))
        
    result = rag_chain.invoke({
        "input": question,
        "chat_history": formatted_history
    })
    
    return {
        "answer": result.get("answer", ""),
        "context": result.get("context", [])
    }
