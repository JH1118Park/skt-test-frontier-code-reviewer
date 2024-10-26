import os

# API 키를 환경변수로 설정
os.environ["OPENAI_API_KEY"] = ""

import bs4  # Beautiful Soup는 HTML과 XML 파일에서 데이터를 추출하기 위한 라이브러리입니다.
from langchain import hub  # Langchain의 허브 모듈, 다양한 언어 처리 기능을 제공합니다.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 텍스트를 재귀적으로 분할하는 데 사용되는 Langchain의 텍스트 분할기입니다.
from langchain_community.document_loaders import WebBaseLoader  # 웹 기반 문서를 로드하기 위한 Langchain 커뮤니티 모듈입니다.
from langchain_chroma import Chroma # Chroma는 벡터 임베딩을 효율적으로 저장하고 쿼리할 수 있는 VectorDB 입니다.
from langchain_core.output_parsers import StrOutputParser  # 출력을 문자열로 파싱하는 Langchain 코어 모듈입니다.
from langchain_core.runnables import RunnablePassthrough  # 실행 가능한 코드를 통과시키는 데 사용되는 Langchain 코어 모듈입니다.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI의 챗봇과 임베딩 기능을 Langchain에서 사용할 수 있게 하는 모듈입니다.

# WebBaseLoader 인스턴스를 생성합니다. 이는 웹 페이지에서 문서를 로드하는 데 사용됩니다.
loader = WebBaseLoader(
    # web_paths는 로드할 웹 페이지의 URL을 지정합니다.
    web_paths=("https://n.news.naver.com/mnews/article/648/0000029698?sid=105",),
    # bs_kwargs는 Beautiful Soup의 파싱 옵션을 지정하는데 사용됩니다. 여기서는 특정 div 태그만 파싱하도록 설정합니다.
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
)
# loader를 사용하여 문서를 로드하고, 로드된 문서를 docs 변수에 저장합니다.
docs = loader.load()

# 로드된 문서의 수를 출력합니다.
print(f"문서의 수: {len(docs)}")

# 로드된 문서의 내용을 출력합니다.
docs

# RecursiveCharacterTextSplitter 인스턴스를 생성합니다. 이는 문서를 재귀적으로 문자 단위로 분할합니다.
# chunk_size는 각 청크의 크기를 지정하며, chunk_overlap은 청크 간 겹치는 문자의 수를 지정합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 생성된 text_splitter를 사용하여 docs에 포함된 문서들을 분할합니다.
# 결과는 분할된 텍스트 청크의 리스트입니다.
splits = text_splitter.split_documents(docs)

# 분할된 청크의 총 개수를 출력합니다.
len(splits)

print(f"len = {len(splits)}")

# Chroma.from_documents 메서드를 사용하여 문서 컬렉션으로부터 벡터 저장소를 생성합니다.
# documents 매개변수에는 분할된 문서들의 리스트인 splits를 전달합니다.
# embedding 모델은 OpenAIEmbeddings() 사용
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# vectorstore 객체의 as_retriever 메서드를 호출하여, 저장된 VectorDB를 기반으로 하는 검색기(retriever)를 생성합니다.
# 이 검색기는 저장된 문서 벡터들 중에서 쿼리에 가장 잘 맞는 문서를 검색하는 데 사용됩니다.
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template ("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
""")

# ChatOpenAI 인스턴스를 생성합니다. 여기서는 모델 이름으로 "gpt-4o"를 사용하고, temperature를 0으로 설정합니다.
# temperature가 0이면 모델의 출력이 더 결정론적이 됩니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# RAG(검색-생성) 체인을 구성합니다.
# 1. "context": 검색기(retriever)를 사용하여 쿼리에 가장 관련 있는 문서를 검색합니다.
# 2. "question": RunnablePassthrough()를 사용하여 입력된 질문을 변경 없이 다음 단계로 전달합니다.
# 3. prompt: 질문을 모델에 전달하기 전에 처리합니다. (prompt 변수의 내용은 이 코드에서 정의되지 않았습니다.)
# 4. llm: ChatOpenAI 인스턴스를 사용하여 질문에 대한 답변을 생성합니다.
# 5. StrOutputParser(): 생성된 답변을 문자열로 파싱합니다.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("SK텔레콤의 새로운 AI 강화 전화 서비스의 이름은 무엇인가요?")
print(answer)