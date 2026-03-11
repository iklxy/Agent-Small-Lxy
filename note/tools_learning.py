import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化模型 (必须使用支持 Function Calling 的模型，如 gpt-3.5-turbo 或 gpt-4)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 1. 使用 @tool 装饰器定义工具 (最简单) ---

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a given location."""
    # 模拟 API 调用
    if "北京" in location:
        return "北京今天晴朗，气温 25 度。"
    elif "上海" in location:
        return "上海今天有雨，气温 22 度。"
    else:
        return f"{location} 的天气未知。"

def demo_tool_decorator():
    """
    1. 使用 @tool 装饰器
    这是最简单的定义工具方式。
    注意：函数必须有类型提示 (Type Hint) 和文档字符串 (Docstring)，因为 LLM 会读取这些信息来理解工具用途。
    """
    print("--- 1. @tool Decorator Demo ---")
    
    # 1. 绑定工具到模型
    tools = [multiply, get_current_weather]
    llm_with_tools = llm.bind_tools(tools)
    
    # 2. 直接调用 (模型会决定是否调用工具)
    query = "北京天气怎么样？"
    print(f"User: {query}")
    res = llm_with_tools.invoke(query)
    
    print(f"Model Response (Tool Call): {res.tool_calls}")
    # tool_calls 输出示例: [{'name': 'get_current_weather', 'args': {'location': '北京'}, 'id': '...'}]
    
    # 注意：这里模型只返回了“我要调用工具”的意图，并没有真正执行函数。
    # 真正的执行通常由 AgentExecutor 或手动代码完成。
    print("-" * 30 + "\n")

# --- 2. 使用 StructuredTool 定义工具 (更严谨，支持复杂参数) ---

class SearchInput(BaseModel):
    query: str = Field(description="The search query string")
    limit: int = Field(description="Max number of results to return", default=5)

def search_function(query: str, limit: int = 5):
    return f"Searching for '{query}' with limit {limit}. Found: [Result 1, Result 2...]"

def demo_structured_tool():
    """
    2. 使用 StructuredTool
    这种方式允许你显式定义参数的 Pydantic 模型，适合参数较多或需要严格验证的场景。
    """
    print("--- 2. StructuredTool Demo ---")
    
    search_tool = StructuredTool.from_function(
        func=search_function,
        name="custom_search",
        description="Search for information on the web",
        args_schema=SearchInput # 绑定 Pydantic 模型
    )
    
    # 测试工具本身
    print(search_tool.invoke({"query": "LangChain", "limit": 3}))
    print("-" * 30 + "\n")

# --- 3. Agent 调用工具 (完整流程) ---

def demo_agent_execution():
    """
    3. Agent 完整流程
    Model (决定调用) -> Executor (执行工具) -> Model (生成最终回答)
    """
    print("--- 3. Agent Execution Demo ---")
    
    tools = [multiply, get_current_weather]
    
    # 1. 创建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # 必须包含这个占位符，用于存放中间步骤
    ])
    
    # 2. 创建 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 3. 创建 Executor (负责循环执行)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 4. 执行
    query = "北京的天气怎么样？另外，50 乘以 3 是多少？"
    print(f"User: {query}")
    res = agent_executor.invoke({"input": query})
    
    print(f"\nFinal Answer: {res['output']}")

if __name__ == "__main__":
    # demo_tool_decorator()
    # demo_structured_tool()
    demo_agent_execution()