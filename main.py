import os
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from dotenv import load_dotenv

file_path = "MyLife.txt"

# load_dotenv()

MY_LIFE_STORY = open(file_path, "r", encoding="utf-8").read()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个lxy的网络分身,请基于以下问题进行回答：\n{context}"),
    ("human", "{input}")
])

model = ChatOllama(
    model="qwen3:1.7b",
    temperature=0.7,
)

chain = prompt | model | StrOutputParser()

question = "你的项目经历是什么？"

for chunk in chain.stream({"context": MY_LIFE_STORY, "input": question}):
    print(chunk, end="", flush=True)
