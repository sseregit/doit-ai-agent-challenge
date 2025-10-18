import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

import retriever_deepseek

load_dotenv()

# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOllama(model="deepseek-r1:8b")


def get_ai_response(messages, docs):
  response = retriever_deepseek.document_chain.stream({
    "messages": messages,
    "context": docs
  })

  for chunk in response:
    yield chunk  # 생성된 응답의 내용을 yield로 순차적으로 반환합니다.


st.title("💬 DeepSeek-R1 Langchain Chat")

if "messages" not in st.session_state:
  st.session_state["messages"] = [
    SystemMessage("너는 문서에 기반해 답변하는 도시 정책 전문가야."),
    AIMessage("How can I help you?"),
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

  print("user\t:", prompt)

  augmented_query = retriever_deepseek.query_augmentation_chain.invoke({
    "messages": st.session_state["messages"],
    "query": prompt,
  })
  print("augmented_query\t", augmented_query)

  print("관련 문서 검색")
  docs = retriever_deepseek.retriever.invoke(f"{prompt}\n{augmented_query}")

  for doc in docs:
    print('---------------')
    print(doc)

    with st.expander(f"**문서:** {doc.metadata.get('source', '알 수 없음')}"):
      st.write(f"**page:**{doc.metadata.get('page', '')}")
      st.write(doc.page_content)
    print("==============")

  with st.spinner(f"AI가 답변을 준비 중입니다...'{augmented_query}'"):
    response = get_ai_response(st.session_state["messages"], docs)
    result = st.chat_message("assistant").write_stream(response)
  st.session_state["messages"].append(AIMessage(result))
