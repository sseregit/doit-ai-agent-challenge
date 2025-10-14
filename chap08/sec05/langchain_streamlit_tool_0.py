import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


def get_ai_response(messages):
  response = llm.stream(messages)

  for chunk in response:
    yield chunk  # 생성된 응답의 내용을 yield로 순차적으로 반환합니다.


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

if prompt := st.chat_input():
  st.chat_message("user").write(prompt)
  st.session_state.messages.append(HumanMessage(prompt))

  response = get_ai_response(st.session_state["messages"])

  result = st.chat_message("assistant").write_stream(response)
  st.session_state["messages"].append(AIMessage(result))
