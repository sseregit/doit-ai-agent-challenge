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
from tools import retrieve, web_search, add_web_pages_json_to_chroma
from utils import save_state, get_outline, save_outline


load_dotenv()

filename = os.path.basename(__file__)
absolute_path = os.path.abspath(__file__)
current_path = os.path.dirname(absolute_path)

llm = ChatOpenAI(model="gpt-4o")


class State(TypedDict):
  messages: List[AnyMessage | str]
  task_history: List[Task]
  references: dict


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
      - web_search_agent: 웹 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.
      - vector_search_agent: 벡터 DB 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.
      
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


def vector_search_agent(state: State):
  print("\n\n============= VECTOR SEARCH AGENT =====================")

  tasks = state.get("task_history", [])
  task = tasks[-1]
  if task.agent != "vector_search_agent":
    raise ValueError(
      f"Vector Search Agent가 아닌 agent가 Vector Search Agent를 시도하고 있습니다.\n {task}")

  vector_search_system_prompt = PromptTemplate.from_template(
      """
      너는 다른 AI Agent 들이 수행한 작업을 바탕으로,
      목차(outline) 작성에 필요한 정보를 벡터 검색을 통해 찾아내는 Agent이다.
      
      현재 목차(outline)를 작성하는 데 필요한 정보를 확보하기 위해,
      다음 내용을 활용해 적절한 벡터 검색을 수행하라.
      
      - 검색 목적: {mission}
      ---
      - 과거 검색 내용: {references}
      ---
      - 이전 대화 내용: {messages}
      ---
      - 목차(outline): {outline}
      """
  )
  mission = task.description
  references = state.get("references", {"queries": [], "docs": []})
  messages = state["messages"]
  outline = get_outline(current_path)

  inputs = {
    "mission": mission,
    "references": references,
    "messages": messages,
    "outline": outline,
  }

  llm_with_retriever = llm.bind_tools([retrieve])
  vector_search_chain = vector_search_system_prompt | llm_with_retriever

  search_plans = vector_search_chain.invoke(inputs)

  for tool_call in search_plans.tool_calls:
    print('---------------------------', tool_call)
    args = tool_call["args"]

    query = args["query"]
    retrieved_docs = retrieve(args)

    references["queries"].append(query)
    references["docs"] += retrieved_docs

  unique_docs = []
  unique_page_contents = set()

  for doc in references["docs"]:
    if doc.page_content not in unique_page_contents:
      unique_docs.append(doc)
      unique_page_contents.add(doc.page_content)
  references["docs"] = unique_docs

  print('Queries:--------------------------')
  queries = references["docs"]
  for query in queries:
    print(query)

  print('References:--------------------------')
  for doc in references["docs"]:
    print(doc.page_content[:100])
    print('--------------------------')

  tasks[-1].done = True
  tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

  new_task = Task(
      agent="communicator",
      done=False,
      description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다.",
      done_at="",
  )

  tasks.append(new_task)

  msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
  message = AIMessage(msg_str)
  print(msg_str)

  messages.append(message)

  return {
    "messages": messages,
    "task_history": tasks,
    "references": references,
  }

  print(task)


def content_strategist(state: State):
  print("\n\n=========== CONTENT STRATEGIST ==============")
  content_strategist_system_prompt = PromptTemplate.from_template(
      """
      너는 책을 쓰는 AI 팀의 콘텐츠 전략가(Content Strategist)로서,
      이전 대화 내용을 바탕으로 사용자의 요구 사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.
      
      지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.
      목차를 작성하는 데 필요한 정보는 "참고 자료"에 있으므로 활용한다.
      --------
      - 지난 목차: {outline}
      --------
      - 이전 대화 내용: {messages}
      ---
      - 참고 자료: {references}
      """
  )
  content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

  messages = state["messages"]
  outline = get_outline(current_path)

  inputs = {
    "messages": messages,
    "outline": outline,
    "references": state.get("references", {"queries": [], "docs": []})
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

def web_search_agent(state: State):
  print("\n\n========= WEB SEARCH AGENT =============")

  tasks = state.get("task_history", [])
  task = tasks[-1]

  if task.agent != "web_search_agent":
    raise ValueError(f"Web Seach Agent가 아닌 agent가 Web Search Agent를 시도하고 있습니다.\n {task}")

  web_search_system_prompt = PromptTemplate.from_template(
      """
      너는 다른 AI Agent 들이 수행한 작업을 바탕으로,
      목차(outline) 작성에 필요한 정보를 웹 검색을 통해 찾아내는 Web Search Agent이다.
      
      현재 부족한 정보를 검색하고, 복합적인 질문으 나눠서 검색하라.
      
      - 검색 목적: {mission}
      ---
      - 과거 검색 내용: {references}
      ---
      - 이전 대화 내용: {messages}
      ---
      - 목차(outline): {outline}
      ---
      - 현재 시각: {current_time}
      """
  )

  messages = state.get("messages", [])

  inputs = {
    "mission": task.description,
    "references": state.get("references", {"queries": [], "docs": []}),
    "messages": messages,
    "outline": get_outline(current_path),
    "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
  }

  llm_with_web_search = llm.bind_tools([web_search])

  web_search_chain = web_search_system_prompt | llm_with_web_search

  search_plans = web_search_chain.invoke(inputs)

  queries = []

  for tool_call in search_plans.tool_calls:
    print('--- web search ---', tool_call)
    args = tool_call["args"]

    queries.append(args["query"])

    _, json_path = web_search.invoke(args)
    print('json_path:', json_path)

    add_web_pages_json_to_chroma(json_path)

  tasks[-1].done = True
  tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

  task_desc = "AI팀이 쓸 책의 세부 목차를 결정하기 위한 정보를 벡터 검색을 통해 찾아낸다."
  task_desc += f" 다음 항목이 새로 추가되었다\n: {queries}"

  new_task = Task(
      agent="vector_search_agent",
      done=False,
      description=task_desc,
      done_at="",
  )

  tasks.append(new_task)

  msg_str = f"[WEB SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
  message = AIMessage(msg_str)

  messages.append(message)

  return {
    "messages": messages,
    "task_history": tasks,
  }


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
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node("web_search_agent", web_search_agent)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
      "content_strategist": "content_strategist",
      "communicator": "communicator",
      "vector_search_agent": "vector_search_agent",
      "web_search_agent": "web_search_agent",
    }
)
graph_builder.add_edge("content_strategist", "communicator")
graph_builder.add_edge("web_search_agent", "vector_search_agent")
graph_builder.add_edge("vector_search_agent", "communicator")
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
