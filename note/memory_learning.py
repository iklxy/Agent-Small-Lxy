import os
from langchain_community.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_community.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def demo_buffer_memory():
    """
    1. 基础缓冲记忆 (ConversationBufferMemory)
    作用：完整记录所有对话历史，直接作为 Prompt 的一部分发送给 LLM。
    优点：信息最完整。
    缺点：随着对话变长，Token 消耗巨大，容易超出上下文限制。
    """
    print("--- 1. ConversationBufferMemory Demo ---")
    
    # 初始化记忆对象
    # return_messages=True: 返回 Message 对象列表 (用于 ChatModel)
    # return_messages=False: 返回字符串 (用于普通 LLM)
    memory = ConversationBufferMemory(return_messages=True)
    
    # 模拟对话
    memory.save_context({"input": "你好，我是 lxy"}, {"output": "你好 lxy，很高兴认识你！"})
    memory.save_context({"input": "我是一名程序员"}, {"output": "程序员很棒！你需要我帮你写代码吗？"})
    
    # 查看记忆内容
    print("当前记忆内容:")
    print(memory.load_memory_variables({}))
    print("-" * 30 + "\n")

def demo_window_memory():
    """
    2. 窗口缓冲记忆 (ConversationBufferWindowMemory)
    作用：只保留最近 k 轮对话。
    优点：控制 Token 消耗，防止超出上限。
    缺点：会丢失较早的对话信息。
    """
    print("--- 2. ConversationBufferWindowMemory (k=1) Demo ---")
    
    # k=1 表示只保留最近 1 轮对话
    memory = ConversationBufferWindowMemory(k=1, return_messages=True)
    
    memory.save_context({"input": "第一句"}, {"output": "回答一"})
    memory.save_context({"input": "第二句"}, {"output": "回答二"})
    memory.save_context({"input": "第三句"}, {"output": "回答三"})
    
    # 查看记忆内容 (应该只有第三句)
    print("当前记忆内容 (k=1):")
    print(memory.load_memory_variables({}))
    print("-" * 30 + "\n")

def demo_summary_memory():
    """
    3. 摘要记忆 (ConversationSummaryMemory)
    作用：利用 LLM 对历史对话进行摘要，将摘要作为上下文。
    优点：可以记录很长的对话而不丢失核心信息，Token 占用相对稳定。
    缺点：需要额外的 LLM 调用来生成摘要，增加延迟和成本。
    """
    print("--- 3. ConversationSummaryMemory Demo ---")
    
    # 需要传入 llm 用于生成摘要
    memory = ConversationSummaryMemory(llm=llm, return_messages=True)
    
    long_conversation = [
        {"input": "我想写一个 AI 数字孪生项目", "output": "这是一个很好的想法。你打算用什么技术栈？"},
        {"input": "我打算用 LangChain 和 Python", "output": "LangChain 是构建 LLM 应用的最佳选择之一。"},
        {"input": "后端可能用 Go", "output": "Go 语言性能很高，适合做后端服务。"},
        {"input": "还需要一个向量数据库", "output": "你可以考虑使用 Chroma 或 Pinecone。"}
    ]
    
    for turn in long_conversation:
        memory.save_context({"input": turn["input"]}, {"output": turn["output"]})
    
    # 查看记忆内容 (应该是一段摘要)
    print("当前记忆内容 (Summary):")
    print(memory.load_memory_variables({}))
    print("-" * 30 + "\n")

# --- 新版 LCEL 记忆管理 (推荐) ---

# 用于存储不同 Session 的历史记录
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def demo_lcel_memory():
    """
    4. LCEL 记忆管理 (RunnableWithMessageHistory)
    作用：LangChain 新版推荐的记忆管理方式。
    核心：将 Chain 包装在 RunnableWithMessageHistory 中，自动处理历史记录的读取和保存。
    """
    print("--- 4. LCEL RunnableWithMessageHistory Demo ---")
    
    # 1. 定义 Prompt，包含历史记录的占位符
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手。"),
        MessagesPlaceholder(variable_name="history"), # 历史记录会自动填充到这里
        ("human", "{input}")
    ])
    
    # 2. 定义基础 Chain
    chain = prompt | llm | StrOutputParser()
    
    # 3. 包装 Chain，加入记忆功能
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history, # 获取历史记录的回调函数
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # 4. 调用 (同一个 session_id 会共享记忆)
    session_id = "user_123"
    
    print("User: 我叫 lxy")
    res1 = chain_with_history.invoke(
        {"input": "我叫 lxy"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"AI: {res1}")
    
    print("User: 我刚才说了什么？")
    res2 = chain_with_history.invoke(
        {"input": "我刚才说了什么？"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"AI: {res2}")
    
    # 打印实际存储的历史记录
    print(f"\nSession {session_id} 的历史记录:")
    print(store[session_id].messages)

if __name__ == "__main__":
    # demo_buffer_memory()
    # demo_window_memory()
    # demo_summary_memory()
    demo_lcel_memory()