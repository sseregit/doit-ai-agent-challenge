from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI()

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
  """
  State 클래스는 TypedDict를 상속받습니다.

  속성:
    messages (Annotated[list[str], add_messages]): 메시지들은 "list" 타입을 가집니다.
    'add_messages' 함수는 이 상태 키가 어떻게 업데이트되어야 하는지 정의한다.
    (이 경우, 메시지를 덮어쓰는 대신 리스트에 추가한다.)
  """
  messages: Annotated[list[str], add_messages]


graph_builder = StateGraph(State)


def generate(state: State):
  """
  주어진 상태를 기반으로 챗봇의 응답 메시지를 생성한다.

  매개변수:
    state (State): 현재 대화 상태를 나타내는 객체로, 이전 메시지들이 포함되어 있다.

  반환값:
    dict: 모델이 생성한 응답 메시지를 포함하는 딕셔너리.
          형식은 {"messages": [응답 메시지]}입니다.
  """
  return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", END)

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig

memory = MemorySaver()

config = RunnableConfig(
    {"configurable": {"thread_id": "abcd"}}
)


graph = graph_builder.compile(checkpointer=memory)


from langchain.schema import HumanMessage

while True:
  user_input = input("You\t:")

  if user_input in ["exit", "quit", "q"]:
    break


  for event in graph.stream({"messages": [HumanMessage(user_input)]},
                            config=config,
                            stream_mode="values"):
    event["messages"][-1].pretty_print()

  print(
      f'\n현재 메시지 개수: {len(event["messages"])}\n----------------------------\n')
