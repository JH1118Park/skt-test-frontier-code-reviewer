# pip install langchain langchain_community langchain_chroma langchain-openai langchainhub
# pip install -U duckduckgo-search

import os

os.environ["OPENAI_API_KEY"] = ""

os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_PROJECT"] = "skt-frontier" # 원하는 프로젝트명

# csv의 사원데이터 ChromaDB에 임베딩하고 검색하는 기능 구현
# from google.colab import files
# uploaded = files.upload()
csv_file_path = "/Volumes/jhp/Study/AI_Frontier/5기/Employee_Data.csv"

# 1. CSV Retriever: 인덱싱 단계
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

loader = CSVLoader(csv_file_path, encoding='utf-8')
docs = loader.load()

chroma_db = Chroma.from_documents(docs, OpenAIEmbeddings())
csv_retriever = chroma_db.as_retriever()

# 2. csv_retriever를 tool로 전환
from langchain.tools.retriever import create_retriever_tool

# desc를 잘써야 Tool을 잘 고른다.
retriever_tool = create_retriever_tool(
    csv_retriever,
    name="employee_search",
    description="Search for information about a employee. Usually searches by name.",
)

# DuckDuckGo 검색 기능
# langchain에서 지원하기 때문에 쉽게 구현 가능
from langchain.tools import DuckDuckGoSearchRun

ddg_search = DuckDuckGoSearchRun()

# for test
result = ddg_search.invoke("서울에 어떤 도시야?")
print(result) 

# tool 리스트
# tools 리스트에 앞에서 만든 2개의 tool을 포함
tools = [retriever_tool, ddg_search]

# Prompt 생성
# OpenAI의 언어 모델을 활용하여 Agent 구현하는 Prompt

# langchain 허브에서 가져온 open ai의 기능을 잘 쓰기 위한
from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# 원래는 아래의 내용같은 Prompt를 넣는거지만, 이미 단들어진게 있으니 가져다 쓴 것.
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
# prompt = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template(
#         "You are a helpful assistant"
#     ),
#     MessagesPlaceholder(variable_name="chat_history", optional=True),
#     HumanMessagePromptTemplate.from_template("{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

# Agent 생성
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent

llm = ChatOpenAI(model='gpt-4', temperature=0)
agent = create_openai_functions_agent(llm= llm, tools = tools, prompt=prompt)

# Agent 실행을 위한 AgentExecutor
from langchain.agents import AgentExecutor

# AgentExecutor 클래스를 사용하여 agent와 tools를 설정하고, 상세한 로그를 출력하도록 verbose를 True로 설정합니다.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor에 질문 테스트
# response = agent_executor.invoke({"input": "김민지 사원이 사는 곳은 어디인가요?"})
# print(f'답변: {response["output"]}')

# response = agent_executor.invoke({"input": "오늘은 2024년 10월 16일 입니다. 서울에서 발생한 최신 뉴스 알려주세요."})
# print(f'답변: {response["output"]}')

# response = agent_executor.invoke({"input": "오늘은 2024년 10월 16일 입니다. 김민지 사원이 사는 곳에서 발생한 최신 뉴스 알려주세요."})
# print(f'답변: {response["output"]}')

# response = agent_executor.invoke({"input": "오늘은 2024년 10월 16일 입니다. 김민지 사원이 하는 업무에 관련된 뉴스를 알려주세요."})
# print(f'답변: {response["output"]}')

response = agent_executor.invoke({"input": "김민지 사원과 박성민 사원이 사는 곳의 거리는 얼마인가요?"})
print(f'답변: {response["output"]}')

