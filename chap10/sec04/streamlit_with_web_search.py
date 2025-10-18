from datetime import datetime
from typing import List

import pytz
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, \
  ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


@tool
def get_current_time(timezone: str, location: str) -> str:
  """ 현재 시각을 반환하는 함수"""
  try:
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    result = f'{timezone} ({location}) 현재 시각 {now}'
    return result
  except pytz.UnknownTimeZoneError:
    return f"알 수 없는 타임존: {timezone}"


@tool
def get_web_search(query: str, search_period: str) -> str:
  """
  웹 검색을 수행하는 함수

  Args:
      query (str): 검색어
      search_period (str): 검색 기간 (e.g, "w" for past week, "m" for past month, "y" for past year

  Return:
    str: 검색 결과
  """
  wrapper = DuckDuckGoSearchAPIWrapper(region='kr-kr', time=search_period)

  print('----------- WEB SEARCH -----------')
  print(query)
  print(search_period)

  search = DuckDuckGoSearchResults(
      api_wrapper=wrapper,
      results_separator=';\n'
  )

  docs = search.invoke(query)
  return docs


@tool
def get_youtube_search(query: str) -> List:
  """
  유투브 검색을 한 뒤, 영상들의 내용을 반환하는 함수,

   Args:
     query (str): 검색어

   Return:
     List: 검색 결과
  """
  print('----------- YOUTUBE SEARCH -----------')
  print(query)

  videos = YoutubeSearch(query, max_results=5).to_dict()

  videos = [video for video in videos if len(video['duration']) <= 5]

  api = YouTubeTranscriptApi()

  for video in videos:
    fetched = api.fetch(video['id'], languages=['ko', 'en'])

    fetched_datas = fetched.to_raw_data()

    video['video_url'] = 'https://youtube.com' + video['url_suffix']
    video['content'] = Document(
        page_content=' '.join(
            fetched_data.text for fetched_data in fetched_datas),
        metadata={"source": video['id']}
    )

  return videos


tools = [get_current_time, get_web_search, get_youtube_search]
tool_dict = {"get_current_time": get_current_time,
             "get_web_search": get_web_search,
             "get_youtube_search": get_youtube_search}

llm_with_tools = llm.bind_tools(tools)


def get_ai_response(messages):
  response = llm_with_tools.stream(messages)

  gathered = None
  for chunk in response:
    yield chunk  # 생성된 응답의 내용을 yield로 순차적으로 반환합니다.

    if gathered is None:
      gathered = chunk
    else:
      gathered += chunk

  if gathered.tool_calls:
    st.session_state.messages.append(gathered)

    for tool_call in gathered.tool_calls:
      selected_tool = tool_dict[tool_call['name']]
      tool_msg = selected_tool.invoke(tool_call)
      print(tool_msg, type(tool_msg))
      st.session_state.messages.append(tool_msg)

    for chunk in get_ai_response(st.session_state.messages):
      yield chunk


st.title("💬 GPT-4o Langchain Chat")

if "messages" not in st.session_state:
  st.session_state["messages"] = [
    SystemMessage("너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다."),
    AIMessage("How can I help you?")
  ]

for msg in st.session_state.messages:
  if msg:
    if isinstance(msg, SystemMessage):
      st.chat_message("system").write(msg.content)
    elif isinstance(msg, AIMessage):
      st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
      st.chat_message("user").write(msg.content)
    elif isinstance(msg, ToolMessage):
      st.chat_message("tool").write(msg.content)

if prompt := st.chat_input():
  st.chat_message("user").write(prompt)
  st.session_state.messages.append(HumanMessage(prompt))

  response = get_ai_response(st.session_state["messages"])

  result = st.chat_message("assistant").write_stream(response)
  st.session_state["messages"].append(AIMessage(result))
