#pip3 install langgraph langsmith ipython graphviz mermaid

import os

os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_PROJECT"] = "skt-frontier" # 원하는 프로젝트명

#LangGraph 예제
# 1. GraphState
from typing import TypedDict

#아래꺼를 계속 노드로 전달함 (정의)
class GraphState(TypedDict):
    """
    GraphState 클래스는 그래프 상태를 나타내는 데이터 타입입니다.

    Attributes:
        UserName (str): 사용자 이름을 나타내는 문자열입니다.
        UserAge (int): 사용자 나이를 나타내는 정수입니다.
        UserInterest (list[str]): 사용자 관심사를 나타내는 문자열 리스트입니다.
    """
    UserName: str
    UserAge: int
    UserInterest: list[str]

# 2. Node
def get_user_name(state: GraphState) -> GraphState:
    # 사용자에게 이름을 입력받습니다.
    name = input("사용자 이름: ")
    # 입력받은 이름을 GraphState에 저장하여 반환합니다.
    return GraphState(UserName=name)

def get_user_age(state: GraphState) -> GraphState: 
    # 사용자에게 나이를 입력받습니다.
    age = input("사용자 나이: ")
    # 입력받은 나이를 정수형으로 변환하고 GraphState에 저장하여 반환합니다.
    return GraphState(UserAge=int(age))

def print_user_info(state: GraphState) -> GraphState:
    # GraphState에서 사용자 이름, 나이, 관심사를 가져와 출력합니다.
    print(f"이름: {state['UserName']}")
    print(f"나이: {state['UserAge']}")
    print(f"관심사: {state['UserInterest']}")
    # 상태를 그대로 반환합니다.
    return state

# Node 등록
# get_user_name 노드: 사용자 이름을 입력받는 작업을 수행합니다.
# get_user_age 노드: 사용자 나이를 입력받는 작업을 수행합니다.
# print_user_info 노드: 사용자 정보를 출력하는 작업을 수행합니다.
from langgraph.graph import START, END, StateGraph

# StateGraph 객체를 생성합니다. 그래프 상태는 GraphState 클래스로 정의합니다.
workflow = StateGraph(GraphState)

# workflow에 노드 들을 등록
workflow.add_node("get_user_name", get_user_name)
workflow.add_node("get_user_age", get_user_age)
workflow.add_node("print_user_info", print_user_info)

# 3. Edge
# 1. 분기가 없는 Edge 구성
# workflow.add_edge(START, "get_user_name")
# workflow.add_edge("get_user_name", "get_user_age")
# workflow.add_edge("get_user_age", "print_user_info")
# workflow.add_edge("print_user_info", END)

# 2. 분기가 있는 Edge 
def is_age_valid(state: GraphState) -> bool:
    return state["UserAge"] >= 19

workflow.add_edge(START, "get_user_name")
workflow.add_edge("get_user_name", "get_user_age")
# workflow.add_edge("get_user_age", "print_user_info")
workflow.add_conditional_edges(
    "get_user_age",
    is_age_valid,
    {
            True: "print_user_info", #이게 메인임, 조건을 걸 수 있다!!
            False: "get_user_name"
    }
)
workflow.add_edge("print_user_info", END)

# 4. Compiled StateGraph
app = workflow.compile()

# CompiledStateGraph 이미지로 출력
# app을 시각화
from IPython.display import Image, display

try:
    # 그래프를 이미지로 변환하여 출력합니다.
    display(
        Image(app.get_graph(xray=True).draw_mermaid_png())
    )
except:
    # 예외가 발생하면 아무 작업도 수행하지 않습니다.
    pass

# 입력을 받아 수행
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    recursion_limit=10,  # Node 기준 실행 제한 횟수 설정
    configurable={
        "thread_id": "SIMPLE_LANGGRAPH"  # 스레드 ID 설정
    }
)

start_input = GraphState(
    UserInterest=['Game', 'Music']
)

try:
    # 앱을 실행합니다.
    result = app.invoke(start_input, config=config)
    print('result:', result)
except Exception as e:
    # recursion_limit을 초과하면 RecursionError가 발생합니다.
    print('Exception:', e)

print('done')