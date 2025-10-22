import os
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, \
  AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from models import Task
from utils import save_state, get_outline, save_outline

load_dotenv()

filename = os.path.basename(__file__)
absolute_path = os.path.abspath(__file__)
current_path = os.path.dirname(absolute_path)

llm = ChatOpenAI(model="gpt-4o")


class State(TypedDict):
  messages: List[AnyMessage | str]
  task_history: List[Task]


def supervisor(state: State):
  print("\n\n========== SUPERVISOR ===============")

  supervisor_system_prompt = PromptTemplate.from_template(
      """
      너는 AI 팀의 superviosr로서 AI 팀의 작업을 관리하고 지도한다.
      사용자가 원하는 책을 써야 한다는 최종 목표를 염두에 두고,
      사용자의 요구를 달성하기 위해 현재 해야 할 일이 무엇인지 결정한다.
      
      supervisor가 활용할 수 있는 agent는 다음과 같다.
      - content_strategist: 사용자의 요구 사항이 명확해졌을 때 사용한다. AI 팀의 콘텐츠 전략을 결정하고, 전체 책의 목차(outline)을 작성한다.
      - communicator: AI 팀에서 해야 할일을 스스로 판단할 수 없을 때 사용한다. 사용자에게 진행 상황을 보고하고, 다음 지시를 물어본다.
      
      아래 내용을 고려하여, 현재 해야할 일이 무엇인지, 사용할 수 있는 agent를 단답으로 말하라.
      
      ---
      previous_outline: {outline}
      ---
      messages: {messages}
      """
  )

  supervisor_chain = supervisor_system_prompt | llm.with_structured_output(Task)

  messages = state.get("messages", [])

  task = supervisor_chain.invoke({"messages": messages,
                                  "outline": get_outline(current_path)})
  task_history = state.get("task_history", [])
  task_history.append(task)

  supervisor_message = AIMessage(f"[Supervisor] {task}")
  messages.append(supervisor_message)
  print(supervisor_message.content)

  return {
    "messages": messages,
    "task_history": task_history
  }


def supervisor_router(state: State):
  task = state['task_history'][-1]
  return task.agent


def content_strategist(state: State):
  print("\n\n=========== CONTENT STRATEGIST ==============")
  content_strategist_system_prompt = PromptTemplate.from_template(
      """
      너는 책을 쓰는 AI 팀의 콘텐츠 전략가(Content Strategist)로서,
      이전 대화 내용을 바탕으로 사용자의 요구 사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.
      
      지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.
      
      --------
      - 지난 목차: {outline}
      --------
      - 이전 대화 내용: {messages}
      """
  )
  content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

  messages = state["messages"]
  outline = get_outline(current_path)

  inputs = {
    "messages": messages,
    "outline": outline
  }

  gathered = ''
  for chunk in content_strategist_chain.stream(inputs):
    gathered += chunk
    print(chunk, end='')

  print()

  save_outline(current_path, gathered)

  content_strategist_message = f"[Content Strategist] 목차 작성 완료"
  print(content_strategist_message)
  messages.append(AIMessage(content_strategist_message))

  task_history = state.get("task_history", [])

  if task_history[-1].agent != "content_strategist":
    raise ValueError(
        f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}")

  task_history[-1].done = True
  task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

  new_task = Task(
      agent="communicator",
      done=False,
      description="AI 팀의 진행 상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다",
      done_at=""
  )
  task_history.append(new_task)

  print(new_task)

  return {"messages": messages, "task_history": task_history}


def communicator(state: State):
  print("\n\n========== COMMUNICATOR ==============")

  communicator_system_prompt = PromptTemplate.from_template(
      """
      너는 책을 쓰는 AI 팀의 커뮤니케이터로서,
      AI 팀의 진행 상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다.
      
      사용자도 outline(목차)를 이미 보고 있으므로, 다시 출력할 필요는 없다.
      
      outline: {outline}
      ---
      messages: {messages}
      """
  )

  system_chain = communicator_system_prompt | llm

  messages = state["messages"]

  inputs = {"messages": messages, "outline": get_outline(current_path)}

  gathered = None

  print('\nAI\t: ', end='')
  for chunk in system_chain.stream(inputs):
    print(chunk.content, end='')

    if gathered is None:
      gathered = chunk
    else:
      gathered += chunk

  messages.append(gathered)

  task_history = state.get("task_history", [])

  if task_history[-1].agent != "communicator":
    raise ValueError(
        f"Communicator가 아닌 agent가 대화를 시도하고 있습니다.\n {task_history[-1]}")

  task_history[-1].done = True
  task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

  return {"messages": messages, "task_history": task_history}


graph_builder = StateGraph(State)

graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
      "content_strategist": "content_strategist",
      "communicator": "communicator"
    }
)
graph_builder.add_edge("content_strategist", "communicator")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

# graph.get_graph().draw_mermaid_png(output_file_path=absolute_path.replace('.py', '.png'))

state = State(
    messages=[
      SystemMessage(
          f"""
          너희 AI들은 사용자의 요구에 맞는 책을 쓰는 작가 팀이다.
          사용자가 사용하는 언어로 대화하라.
          
          현재 시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다.
          
          """
      )
    ],
    task_history=[]
)

while True:
  user_input = input("\nUser\t: ").strip()

  if user_input.lower() in ['exit', 'quit', 'q']:
    print("Goodbye!")
    break

  state["messages"].append(HumanMessage(user_input))
  state = graph.invoke(state)

  print('\n--------------------------- MESSAGE COUNT\t', len(state["messages"]))

  save_state(current_path, state)
