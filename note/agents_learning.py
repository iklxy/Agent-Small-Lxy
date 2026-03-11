import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化模型
# 注意：Tool Calling Agent 需要支持函数调用的模型 (如 gpt-3.5-turbo, gpt-4)
# ReAct Agent 可以使用任何足够智能的模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 定义工具 (供 Agent 使用) ---

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [add, multiply, get_word_length]

# --- 1. Tool Calling Agent (Function Calling) ---

def demo_tool_calling_agent():
    """
    1. Tool Calling Agent (Function Calling)
    适用场景：使用 OpenAI 等支持 Function Calling 的模型。
    特点：更可靠，结构化输出，不易出错。
    """
    print("--- 1. Tool Calling Agent Demo ---")
    
    # 1. 获取 Prompt (可以直接从 LangSmith Hub 拉取，也可以自己写)
    # 这是一个标准的 tool calling prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # 关键占位符
    ])
    
    # 2. 创建 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 3. 创建 Executor (执行器)
    # AgentExecutor 负责运行 Agent，执行工具，处理错误，并将结果返回给 Agent
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 4. 执行 (多工具组合使用)
    query = "单词 'LangChain' 的长度是多少？将结果乘以 5，再加上 3。"
    print(f"User: {query}")
    
    res = agent_executor.invoke({"input": query})
    print(f"Final Answer: {res['output']}\n")

# --- 2. ReAct Agent (Reasoning + Acting) ---

def demo_react_agent():
    """
    2. ReAct Agent
    适用场景：适用于不支持 Function Calling 的通用 LLM，或者需要显式推理过程的场景。
    原理：基于 "Thought, Action, Observation" 的循环。
    """
    print("--- 2. ReAct Agent Demo ---")
    
    # 1. 获取 ReAct Prompt (标准模板)
    # 这个模板通常包含: Thought: ... Action: ... Action Input: ...
    # 你可以从 hub 拉取，或者复制标准模板
    prompt = hub.pull("hwchase17/react")
    
    # 2. 创建 Agent
    agent = create_react_agent(llm, tools, prompt)
    
    # 3. 创建 Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 4. 执行
    query = "计算 10 乘以 5，然后加上 20。"
    print(f"User: {query}")
    
    # 注意：ReAct Agent 有时对 Prompt 格式要求较严，gpt-4o-mini 通常表现良好
    try:
        res = agent_executor.invoke({"input": query})
        print(f"Final Answer: {res['output']}\n")
    except Exception as e:
        print(f"ReAct Agent Error: {e}")

# --- 3. 单工具 vs 多工具使用 ---
# 其实在 Agent 层面，单工具和多工具没有本质区别，只是 tools 列表的长度不同。
# Agent 会根据问题自动决定使用哪个工具，或者组合使用多个工具。

def demo_single_tool():
    """演示仅提供一个工具的情况"""
    print("--- 3. Single Tool Demo ---")
    
    single_tools = [get_word_length]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个只懂计算单词长度的助手。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, single_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=single_tools, verbose=True)
    
    res = agent_executor.invoke({"input": "Hello 的长度是多少？"})
    print(f"Result: {res['output']}\n")

if __name__ == "__main__":
    # 1. 推荐：OpenAI Function Calling Agent
    demo_tool_calling_agent()
    
    # 2. 通用：ReAct Agent
    # demo_react_agent()
    
    # 3. 单工具
    # demo_single_tool()