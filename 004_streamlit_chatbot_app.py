import streamlit as st
import os
import tempfile

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from kiwipiepy import Kiwi

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# 한국어 Embedding 모델을 사용하는 경우
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 토크나이저의 병렬 처리를 비활성화하여 리소스 사용을 줄임 (안정화)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # LangSmith (필요 시 주석 해제): api 환경변수 설정
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"

def main():
    st.set_page_config(
        page_title="SKT-Frontier RAG Chat")

    st.title("SKT-Frontier RAG Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    with st.sidebar:
        st.selectbox(
            "Choose the language model",
            ("gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo-preview", "bnksys/yanolja-eeve-korean-instruct-10.8b"),
            key="model_selection"
        )
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)

        # 사용자가 OpenAI API 키를 입력할 수 있는 비밀번호 입력란을 생성
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")        
        
        # # LangSmith (필요 시 주석 해제): 환경 변수 입력을 위한 UI 추가
        # langchain_api_key = st.text_input("LangChain API Key", key="langchain_api_key", type="password")
        # langchain_project = st.text_input("LangChain Project", key="langchain_project")
        
        process = st.button("Process")
    
    # # LangSmith (필요 시 주석 해제): 입력받은 환경변수로 설정
    # os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    # os.environ["LANGCHAIN_PROJECT"] = langchain_project

    if process:
        # # LangSmith (필요 시 주석 해제): 관련 api key 및 project 정보가 없는 경우 경고 메시지 출력
        # if not openai_api_key or not langchain_api_key or not langchain_project:
        if not openai_api_key:
            st.info("Please add all necessary API keys and project information to continue.")
            st.stop()

        with st.spinner("Processing documents..."):
            # Load
            files_text = get_text(uploaded_files)

            # embedding_model_name = 'intfloat/multilingual-e5-large-instruct'
            embedding_model_name = ''
            # Split
            text_chunks = get_text_chunks(files_text)
            # Store
            vetorestore = get_vectorstore(text_chunks, openai_api_key)

            ensemble_retriever = get_ensemble_retriever(vetorestore, text_chunks)
            st.session_state.conversation = get_conversation_chain(ensemble_retriever, openai_api_key, st.session_state.model_selection)

            st.session_state.messages.clear()
            st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! SKT-Frontier RAG chatbot 입니다. 주어진 문서에 대해 궁금한 점을 물어보세요."}]

            st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! SKT-Frontier RAG chatbot 입니다. 주어진 문서에 대해 궁금한 점을 물어보세요."}]

    # 채팅 기록 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.processComplete:
        # 사용자가 채팅 입력란에 메시지를 입력하면 query 변수에 할당
        if query := st.chat_input("Message to chatbot"):
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation.invoke({"query": query})
                    result = response['result']
                    source_documents = response['source_documents']
                    st.markdown(result)
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            st.markdown(doc.metadata['source'], help=doc.page_content)
            # 어시스턴트 메시지를 채팅 기록에 추가
            st.session_state.messages.append({"role": "assistant", "content": result})
    else:
        st.info("질문하기 전에 문서를 업로드하고 'Process'를 클릭하세요.")


def load_document(doc):
    """
    문서 유형에 따라 문서를 로드하고 분할합니다.

    이 함수는 제공된 문서를 임시 디렉토리에 저장하고, 파일 유형을 결정한 후
    적절한 로더를 사용하여 문서를 로드하고 분할합니다. 지원되는 파일 유형은 PDF, DOCX, PPTX입니다.
    처리 후 임시 파일은 삭제됩니다.

    Args:
        doc (UploadedFile): 로드하고 분할할 문서입니다. 'name' 속성과 'getbuffer' 메서드를 통해
                            문서 내용을 가져올 수 있어야 합니다.

    Returns:
        list: 로드되고 분할된 문서 부분들의 리스트입니다. 파일 유형이 지원되지 않는 경우 빈 리스트를 반환합니다.
    """
    # 임시 디렉토리에 파일 저장
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, doc.name)

    # 파일 쓰기
    with open(file_path, "wb") as file:
        file.write(doc.getbuffer())  # 파일 내용을 임시 파일에 쓴다

    # 파일 유형에 따라 적절한 로더를 사용하여 문서 로드 및 분할
    try:
        if file_path.endswith('.pdf'):
            loaded_docs = PyPDFLoader(file_path).load_and_split()
        elif file_path.endswith('.docx'):
            loaded_docs = Docx2txtLoader(file_path).load_and_split()
        elif file_path.endswith('.pptx'):
            loaded_docs = UnstructuredPowerPointLoader(file_path).load_and_split()
        else:
            loaded_docs = []  # 지원되지 않는 파일 유형
    finally:
        os.remove(file_path)  # 작업 완료 후 임시 파일 삭제

    return loaded_docs

def get_text(docs):
    """
    문서 경로 목록을 처리하고 해당 내용의 결합된 목록을 반환합니다.

    Args:
        docs (list): 처리할 문서 경로 목록입니다.

    Returns:
        list: 모든 문서의 내용을 포함하는 결합된 목록입니다.
    """
    doc_list = []
    for doc in docs:
        doc_list.extend(load_document(doc))
    return doc_list

def get_text_chunks(text: str, embedding_model_name: str=''):
    """
    주어진 텍스트를 지정된 임베딩 모델의 토크나이저를 사용하여 청크로 분할합니다.

    Args:
        text (str): 청크로 분할할 입력 텍스트입니다.
        embedding_model_name (str): 토크나이저로 사용할 임베딩 모델의 이름입니다.

    Returns:
        List[str]: 텍스트 청크의 리스트입니다.
    """
    if embedding_model_name == '':
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    else:
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=900, 
            chunk_overlap=100
        )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks: str, openai_api_key: str, embedding_model_name: str = ''):
    """
    주어진 텍스트 청크를 지정된 임베딩 모델을 사용하여 벡터 저장소로 생성합니다.
    Args:
        text_chunks (str): 벡터로 변환할 텍스트 청크입니다.
        embedding_model_name (str): 사용할 임베딩 모델의 이름입니다.
    Returns:
        FAISS: 벡터화된 텍스트 청크를 포함하는 FAISS 벡터 저장소입니다.
    Raises:
        ValueError: text_chunks 입력이 비어 있는 경우 발생합니다.
    """
    if embedding_model_name == '':
        embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_ensemble_retriever(vetorestore, splitted_docs) -> EnsembleRetriever:
    doc_k = 3
    search_type = 'mmr' # 'similarity'
    faiss_retriever = vetorestore.as_retriever(search_type=search_type, search_kwargs={"k": doc_k})
    bm25_retriever = BM25Retriever.from_documents(splitted_docs, preprocess_func=kiwi_tokenize)
    bm25_retriever.k = doc_k
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
        search_type=search_type,
    )

    return ensemble_retriever

@st.cache_resource
def get_kiwi():
    return Kiwi()

def kiwi_tokenize(text):
    kiwi = get_kiwi()
    return [token.form for token in kiwi.tokenize(text)]

def get_conversation_chain(retriever, openai_api_key, model_selection):
    """
    OpenAI의 GPT 모델 또는 Ollama의 로컬 LLM을 사용하여 대화형 검색 체인을 생성합니다.
    Args:
        retriever: 관련 문서를 검색하는 데 사용되는 검색기 객체입니다.
        openai_api_key (str): OpenAI 서비스에 액세스하기 위한 API 키입니다.
        model_selection (str): 사용할 모델 이름입니다. 'gpt'로 시작하면 OpenAI의 GPT 모델이 사용되고, 그렇지 않으면 Ollama의 로컬 LLM이 사용됩니다.
    Returns:
        ConversationalRetrievalChain: 구성된 대화형 검색 체인 객체입니다.
    """
    if model_selection.startswith('gpt'): # OpenAI의 GPT 모델
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection, temperature=0)
    else: # Ollama 로컬LLM
        llm = OllamaLLM(model=model_selection, temperature=0)

    # 프롬프트 템플릿 정의
    rag_prompt = PromptTemplate(
        template="""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        Question: {question} 

        Context: {context} 

        Answer:
        """, 
        input_variables=["context", "question"]
    )

    # RetrievalQA 체인 생성
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        # chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt}
    )
    
    return rag_chain

if __name__ == '__main__':
    main()