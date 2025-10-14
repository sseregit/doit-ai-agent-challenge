import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory, \
  BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

st.title("💬 Chatbot")

if "messages" not in st.session_state:
  st.session_state["messages"] = [
    SystemMessage("너는 사용자의 질문에 친절이 답하는 AI 챗봇이다.")
  ]

if "store" not in st.session_state:
  st.session_state["store"] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
  if session_id not in st.session_state["store"]:
    st.session_state["store"][session_id] = InMemoryChatMessageHistory()
  return st.session_state["store"][session_id]


llm = ChatOpenAI(model="gpt-4o-mini")
with_message_history = RunnableWithMessageHistory(llm, get_session_history)

config = {"configurable": {"session_id": "abc2"}}

for msg in st.session_state.messages:
  if msg:
    if isinstance(msg, SystemMessage):
      st.chat_message("system").write(msg.content)
    elif isinstance(msg, AIMessage):
      st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
      st.chat_message("user").write(msg.content)

if prompt := st.chat_input():
  print('user:', prompt)
  st.session_state.messages.append(HumanMessage(prompt))
  st.chat_message("user").write(prompt)

  response = with_message_history.stream([HumanMessage(prompt)], config)

  ai_response_bucket = None

  with st.chat_message("assistant").empty():
    for r in response:
      if ai_response_bucket is None:
        ai_response_bucket = r
      else:
        ai_response_bucket += r
      print(r.content, end='')
      st.markdown(ai_response_bucket.content)

  msg = ai_response_bucket.content
  st.session_state.messages.append(response)
  print('assistant:', msg)
