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
  user_request: str
  ai_recommendation: str
  supervisor_call_count: int


def business_analyst(state: State):
  print("\n\n--------- BUSINESS ANALYST --------")

  business_analyst_system_prompt = PromptTemplate.from_template(
      """
      너는 책을 쓰는 AI 팀의 비즈니스 애널리스트로서,
      AI 팀의 진행상황과 "시용자 요구 사항"을 토대로,
      현 시점에서 'ai_recommendation'과 최근 사용자의 발언을 바탕으로 요구 사항이 무엇인지 판단한다.
      지난 요구 사항이 달성되었는지 판단하고, 현 시점에서 어떤 작업을 해야 하는지 결정한다.
      
      다음과 같은 템플릿 형태로 반환한다.
      ```
      - 목표: 0000 \n 방법: 0000
      ```
      
      ---
      *AI 추천(ai_recommendation)* : {ai_recommendation}
      ---
      사용자 최근 발언: {user_last_comment}
      ---
      참고자료: {references}
      ---
      목차 (outline): {outline}
      ---
      "messages": {messages}
      """
  )

  ba_chain = business_analyst_system_prompt | llm | StrOutputParser()

  messages = state["messages"]

  user_last_comment = None
  for m in messages[::-1]:
    if isinstance(m, HumanMessage):
      user_last_comment = m.content
      break

  inputs = {
    "ai_recommendation": state.get("ai_recommendation", None),
    "previous_user_request": state.get("user_request", None),
    "user_last_comment": state.get("references", {"queries": [], "docs": []}),
    "references": get_outline(current_path),
    "outline": messages,
    "messages": user_last_comment,
  }

  user_request = ba_chain.invoke(inputs)

  business_analyst_message = f"[Business Analyst] {user_request}"
  print(business_analyst_message)
  messages.append(AIMessage(business_analyst_message))

  save_state(current_path, state)

  return {
    "messages": messages,
    "user_request": user_request,
    "ai_recommendation": ""
  }


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
      - web_search_agent: vector_search_agent를 시도하고, 검색 결과(references)에 필요한 정보가 부족한 경우 사용한다. 웹 검색을 통해 해당 정보를 벡터 DB에 보강한다.
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

  inputs = {"messages": messages,
            "outline": get_outline(current_path)}

  supervisor_call_count = state.get("supervisor_call_count", 0)

  if supervisor_call_count > 2:
    print("Supervisor 호출 횟수 초과: Communicator 호출")
    task = Task(
        agent="communicator",
        done=False,
        description="supervisor 호출 횟수를 초과했으므로, 현재까지의 진행 상황을 사용자에게 보고한다. ",
        dont_at="",
    )
  else:
    task = supervisor_chain.invoke(inputs)

  task = supervisor_chain.invoke(inputs)
  task_history = state.get("task_history", [])
  task_history.append(task)

  supervisor_message = AIMessage(f"[Supervisor] {task}")
  messages.append(supervisor_message)
  print(supervisor_message.content)

  return {
    "messages": messages,
    "task_history": task_history,
    "supervisor_call_count": supervisor_call_count + 1
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

  # new_task = Task(
  #     agent="communicator",
  #     done=False,
  #     description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다.",
  #     done_at="",
  # )
  #
  # tasks.append(new_task)

  msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
  message = AIMessage(msg_str)
  print(msg_str)

  messages.append(message)

  ai_recommendation = "현재 참고자료(references)가 목차(outline)를 개선하는 데 충분한지 확인하라. 충분하다면 content_strategist로 목차 작성을 하라."

  return {
    "messages": messages,
    "task_history": tasks,
    "references": references,
    "ai_recommendation": ai_recommendation,
  }


def content_strategist(state: State):
  print("\n\n=========== CONTENT STRATEGIST ==============")

  task_history = state.get("task_history", [])
  task = task_history[-1]
  if task.agent != "content_strategist":
    raise ValueError(
        f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task}")

  content_strategist_system_prompt = PromptTemplate.from_template(
      """
      너는 책을 쓰는 AI 팀의 콘텐츠 전략가(Content Strategist)로서,
      이전 대화 내용을 바탕으로 사용자의 요구 사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.
      
      지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.
      목차를 작성하는 데 필요한 정보는 "참고 자료"에 있으므로 활용한다.
      
다음 정보를 활용하여 목차를 작성하라. 
        - 사용자 요구사항(user_request)
        - 작업(task)
        - 검색 자료 (references)
        - 기존 목차 (previous_outline)
        - 이전 대화 내용(messages)

        너의 작업 목표는 다음과 같다:
        1. 만약 "기존 목차 구조 (previous_outline)"이 존재한다면, 사용자의 요구사항을 토대로 "기존 목차 구조"에서 어떤 부분을 수정하거나 추가할지 결정한다.
        - "이번 목차 작성의 주안점"에 사용자 요구사항(user_request)을 충족시키는 것을 명시해야 한다.
        2. 책의 전반적인 구조(chapter, section)를 설계하고, 각 chpater와 section의 제목을 정한다.
        3. 책의 전반적인 세부구조(chapter, section, sub-section)를 설계하고, sub-section 하부의 주요 내용을 리스트 형태로 정리한다.
        4. 목차의 논리적인 흐름이 사용자 요구를 충족시키는지 확인한다.
        5. 참고자료 (references)를 적극 활용하여 근거에 기반한 목차를 작성한다.
        6. 참고문헌은 반드시 참고자료(references) 자료를 근거로 작성해야 하며, 최대한 풍부하게 준비한다. URL은 전체 주소를 적어야 한다.
        7. 추가 자료나 리서치가 필요한 부분을 파악하여 supervisor에게 요청한다.

        사용자 요구사항(user_request)을 최우선으로 반영하는 목차로 만들어야 한다. 

        --------------------------------
        - 사용자 요구사항(user_request): 
        {user_request}
        --------------------------------
        - 작업(task): 
        {task}
        --------------------------------
        - 참고 자료 (references)
        {references}
        --------------------------------
        - 기존 목차 (previous_outline)
        {outline}
        --------------------------------
        - 이전 대화 내용(messages)
        {messages}
        --------------------------------

        작성 형식 아래 양식을 지키되 하부 항목으로 더 세분화해도 좋다. 목차(outline) 양식의 챕터, 섹션 등 항목의 갯수는 필요한만큼 추가하라. 
        섹션 갯수는 최소 2개 이상이어야 하며, 더 많으면 좋다. 

        outline_template은 예시로 앞부분만 제시한 것이다. 각 장은 ':---CHAPTER DIVIDER---:'로 구분한다.
        outline_template:
        {outline_template}

        사용자가 추가 피드백을 제공할 수 있도록 논리적인 흐름과 주요 목차 아이디어를 제안하라.      
      """
  )
  content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

  user_request = state.get("user_request", "")
  messages = state["messages"]
  outline = get_outline(current_path)

  with open(f"{current_path}/templates/outline_template.md", "r",
            encoding='utf-8') as f:
    outline_template = f.read()

  inputs = {
    "user_request": user_request,
    "task": task,
    "messages": messages,
    "outline": outline,
    "references": state.get("references", {"queries": [], "docs": []}),
    "outline_template": outline_template,
  }

  gathered = ''
  for chunk in content_strategist_chain.stream(inputs):
    gathered += chunk
    print(chunk, end='')

  print()

  save_outline(current_path, gathered)

  if '-----: DONE :-----' in gathered:
    review = gathered.split('-----: DONE :-----')[1]
  else:
    review = gathered[-200:]

  content_strategist_message = f"[Content Strategist] 목차 작성 완료: outline 작성 완료\n {review}"
  print(content_strategist_message)
  messages.append(AIMessage(content_strategist_message))

  # task_history = state.get("task_history", [])
  #
  # if task_history[-1].agent != "content_strategist":
  #   raise ValueError(
  #       f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}")

  task_history[-1].done = True
  task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

  # new_task = Task(
  #     agent="communicator",
  #     done=False,
  #     description="AI 팀의 진행 상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다",
  #     done_at=""
  # )
  # task_history.append(new_task)
  # print(new_task)

  return {"messages": messages, "task_history": task_history}


def outline_reviewer(state: State):
  print("\n\n====== OUTLINE REVIEWER =======")

  outline_reviewer_system_prompt = PromptTemplate.from_template(
      """
      너는 AI팀의 목차 리뷰어로서, AI팀이 작성한 목차(outline)를 검토하고 문제점을 지적한다.
      
      - outline이 사용자의 요구사항을 충족시키는지 여부
      - outline의 논리적인 흐름이 적절한지 여부
      - 근거에 기반하지 않은 내용이 있는지 여부
      - 주어진 참고자료(references)를 충분히 활용했는지 여부
      - 참고자료가 충분한지, 혹은 잘못된 참고자료가 있는지 여부
      - example.come 같은 더미 URL이 있는지 여부:
      - 실제 페이지 URL이 아닌 대표 URL로 되어 있는 경우 삭제 해야함: 어떤 URL이 삭제되어야 하는지 명시하라.
      - 기타 리뷰 사항
      
      그 분석 결과를 설명하고, 다음에 어떤 작업을 하면 좋을지 제안하라.
      
      - 분석 결과: outline이 사용자의 요구사항을 충족시키는지 여부
      - 제안 사항: (vector_search_agent, communicator 중 어떤 agent를 호출할지)
      
      ---
      user_request: {user_request}
      ---
      references: {references}
      ---
      outline: {outline}
      ---
      messages: {messages}
      """
  )

  user_request = state.get("user_request", None)
  outline = get_outline(current_path)
  references = state.get("references", {"queries": [], "docs": []})
  messages = state.get("messages", [])

  inputs = {
    "user_request": user_request,
    "outline": outline,
    "references": references,
    "messages": messages,
  }

  outline_reviewer_chain = outline_reviewer_system_prompt | llm

  review = outline_reviewer_chain.stream(inputs)

  gathered = None

  for chunk in review:
    print(chunk.content, end='')

    if gathered is None:
      gathered = chunk
    else:
      gathered += chunk
  if '[OUTLINE REVIEW AGENT]' not in gathered.content:
    gathered.content = f"[OUTLINE REVIEW AGENT] {gathered.content}"

  print(gathered.content)
  messages.append(gathered)

  ai_recommendation = gathered.content

  return {"messages": messages, "ai_recommendation": ai_recommendation}


def web_search_agent(state: State):
  print("\n\n========= WEB SEARCH AGENT =============")

  tasks = state.get("task_history", [])
  task = tasks[-1]

  if task.agent != "web_search_agent":
    raise ValueError(
        f"Web Seach Agent가 아닌 agent가 Web Search Agent를 시도하고 있습니다.\n {task}")

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

  return {"messages": messages, "task_history": task_history,
          "supervisor_call_count": 0}


graph_builder = StateGraph(State)

graph_builder.add_node("business_analyst", business_analyst)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)
graph_builder.add_node("outline_reviewer", outline_reviewer)
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node("web_search_agent", web_search_agent)

graph_builder.add_edge(START, "business_analyst")
graph_builder.add_edge("business_analyst", "supervisor")
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
graph_builder.add_edge("content_strategist", "outline_reviewer")
graph_builder.add_edge("outline_reviewer", "business_analyst")
graph_builder.add_edge("web_search_agent", "vector_search_agent")
graph_builder.add_edge("vector_search_agent", "business_analyst")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()


def get_mermaid_png():
  graph.get_graph().draw_mermaid_png(
      output_file_path=absolute_path.replace('.py', '.png'))


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
    task_history=[],
    references={"queries": [], "docs": []},
    user_request="",
)


def start():
  global state
  while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ['exit', 'quit', 'q']:
      print("Goodbye!")
      break

    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print('\n--------------------------- MESSAGE COUNT\t',
          len(state["messages"]))

    save_state(current_path, state)


# get_mermaid_png()
start()
