import json
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI  # 오픈AI 라이브러리를 가져오기

from gpt_functions import get_current_time, tools, get_yf_stock_info, \
  get_yf_stock_history, get_yf_stock_recommendations

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기

client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성


# ①
def get_ai_response(messages, tools=None, stream=True):
  response = client.chat.completions.create(
      model="gpt-5-nano-2025-08-07",  # 응답 생성에 사용할 모델 지정
      temperature=1,  # 응답 생성에 사용할 temperature 설정
      stream=stream,
      messages=messages,  # 대화 기록을 입력으로 전달
      tools=tools,
  )

  if stream:
    for chunk in response:
      yield chunk  # 생성된 응답의 내용을 yield로 순차적으로 반환합니다.
  else:
    return response  # 생성된 응답의 내용을 반환합니다.


st.title("💬 Chatbot")

if "messages" not in st.session_state:
  st.session_state["messages"] = [
    {"role": "system", "content": "너는 사용자를 도와주는 상담사야."}
  ]

for msg in st.session_state.messages:
  st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
  st.session_state.messages.append({"role": "user", "content": user_input})
  st.chat_message("user").write(user_input)

  ai_response = get_ai_response(st.session_state.messages, tools=tools)
  # print(ai_message)

  content = ''
  with st.chat_message("assistant").empty():
    for chunk in ai_response:
      content_chunk = chunk.choices[0].delta.content
      if content_chunk:
        print(content_chunk, end='')
        content += content_chunk
        st.markdown(content)

  print('\n===========')
  print(content)

  tool_calls = None
  if tool_calls:

    for tool_call in tool_calls:
      tool_name = tool_call.function.name
      tool_call_id = tool_call.id

      arguments = json.loads(tool_call.function.arguments)

      if tool_name == "get_current_time":
        func_result = get_current_time(timezone=arguments['timezone'])
      elif tool_name == "get_yf_stock_info":
        func_result = get_yf_stock_info(ticker=arguments['ticker'])
      elif tool_name == "get_yf_stock_history":
        func_result = get_yf_stock_history(ticker=arguments['ticker'],
                                           period=arguments['period'])
      elif tool_name == "get_yf_stock_recommendations":
        func_result = get_yf_stock_recommendations(ticker=arguments['ticker'])

      st.session_state.messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": func_result
      })

    ai_response = get_ai_response(st.session_state.messages)
    ai_message = ai_response.choices[0].message

  st.session_state.messages.append({
    "role": "assistant",
    "content": content
  })

  print("AI\t: " + content)
  st.chat_message("assistant").write(ai_message.content)
