# pip3 install langchain langchain_community langchain_chroma langchain-openai
# pip3 install rank_bm25 kiwipiepy langchain_huggingface

# 2. 환경변수 설정
import os

os.environ["OPENAI_API_KEY"] = ""

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"]="skt-04"

# Splitted Docs
# 기사 텍스트를 로드하여 작은 단위로 분리
# 기사 텍스트 파일 업로드

import os
from langchain_community.document_loaders import TextLoader

# from google.colab import files
# uploaded = files.upload()

file_path = '/Volumes/jhp/Study/AI_Frontier/5기/gpt-4o-mini.txt'
loader = TextLoader(file_path)
docs = loader.load()

print(docs)

# Text Split
# 한국어 지원이 용이한 임베딩 모델 사용
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

encode_kwargs = {"normalize_embeddings": True}

## 한국어 임베딩 랭킹이 높았던 모델을 허깅페이스를 통해 땡겨옴
embeddings = HuggingFaceEmbeddings(
    # model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    model_name="intfloat/multilingual-e5-large-instruct",
    encode_kwargs=encode_kwargs,
    show_progress=True
)

# AutoTokenizer는 model_name에 맞는 알아서 잘라야 하는 사이즈를 찾아줌.
tokenizer = AutoTokenizer.from_pretrained(embeddings.model_name)
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=200,
    chunk_overlap=30
)

splitted_docs = text_splitter.split_documents(docs)
print(len(splitted_docs))
print(splitted_docs)

# kiwi 한국 형태소 분석기 (https://github.com/bab2min/kiwipiepy)
# BM25는 특별한 설정이 없으면 ' '를 기준으로 키워드를 분리 (영어에는 맞음.)
# 한국어에는 맞지 않는 방식으로 한국어에 적합한 형태소 분석기를 사용

from kiwipiepy import Kiwi

kiwi = None
def kiwi_tokenize(text):
    global kiwi
    if kiwi is None:
        kiwi = Kiwi()

    return [token.form for token in kiwi.tokenize(text)]

# BM25 Retriever
# 문서 2개를 검색해서 반환
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(splitted_docs, preprocess_func=kiwi_tokenize)
bm25_retriever.k = 2

# Dense Retriever
# ChaomaDB 사용
# 유사도 검색 사용
# 문서 2개를 검색해서 반환
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=splitted_docs, embedding=embeddings)
chroma_retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 2})

# Ensemble Retriever로 묶음.
# BM25 + Chroma
# 가중치는 동일하게 0.5로 설정
from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5],
    # search_type="mmr",
    search_type="similarity",
)

# Ensemble Retriever 테스트
# Prompt
# 기존에 사용했던 rag prompt
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template ("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
""")

# 기존과 동일한 chain 구성
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
rag_chain = (
    {"context": ensemble_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("GPT-4o 미니의 컨텍스트 창 크기는 얼마이며, 이는 대략 어느 정도의 텍스트 양에 해당합니까?")
print(result)

result = rag_chain.invoke("오픈AI가 GPT-4o 미니의 안전성 향상을 위해 도입한 '지시 계층(The Instruction Hierarch)'은 어떤 기능을 합니까?")
print(result)