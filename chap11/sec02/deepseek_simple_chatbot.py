# from openai import OpenAI  # 오픈AI 라이브러리를 가져오기
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# import os
from langchain_ollama import ChatOllama

load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기

# client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성
# llm = ChatOpenAI(model="gpt-4o")
llm = ChatOllama(model="deepseek-r1:8b")

# ①
# def get_ai_response(messages):
#     response = client.chat.completions.create(
#         model="gpt-4o",  # 응답 생성에 사용할 모델 지정
#         temperature=0.9,  # 응답 생성에 사용할 temperature 설정
#         messages=messages,  # 대화 기록을 입력으로 전달
#     )
#     return response.choices[0].message.content  # 생성된 응답의 내용 반환

messages = [
  # {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},  # 초기 시스템 메시지
  SystemMessage("너는 사용자를 도와주는 상담사야.")
]

while True:
  user_input = input("사용자: ")  # 사용자 입력 받기

  if user_input == "exit":  # ② 사용자가 대화를 종료하려는지 확인인
    break

  messages.append(
      # {"role": "user", "content": user_input}
      HumanMessage(user_input)
  )  # 사용자 메시지를 대화 기록에 추가
  # ai_response = get_ai_response(messages)  # 대화 기록을 기반으로 AI 응답 가져오기

  response = llm.stream(messages)

  ai_message = None
  for chunk in response:
    print(chunk.content, end="")
    if ai_message is None:
      ai_message = chunk
    else:
      ai_message += chunk
  print('')

  message_only = ai_message.content.split("</think>")[1].strip()
  messages.append(AIMessage(message_only))
