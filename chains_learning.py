import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# 1. 基础链和顺序链 (可能需要安装 langchain-community)
from langchain_community.chains import LLMChain, SimpleSequentialChain, SequentialChain

# 2. 数学链 (现在通常在 community 中)
from langchain_community.chains import LLMMathChain

# 3. 路由相关 (路径非常深，容易爆红)
from langchain_community.chains import MultiPromptChain
from langchain_community.chains import LLMRouterChain, RouterOutputParser
from langchain_community.chains import MULTI_PROMPT_ROUTER_TEMPLATE

# 4. SQL 链 (新版推荐位置)
from langchain_community.chains import create_sql_query_chain

# 5. 文档处理链 (核心位置)
from langchain_community.chains import create_stuff_documents_chain
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化模型 (可以使用 gpt-4o-mini 或 gpt-3.5-turbo)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def demo_basic_chain():
    """
    1. 基础链 (Basic Chain)
    旧版: LLMChain
    新版: LCEL (Prompt | LLM | Parser)
    """
    print("--- 1. 基础链 Demo ---")
    
    # 旧版写法 (不推荐，但了解一下)
    # prompt = PromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    # chain = LLMChain(llm=llm, prompt=prompt)
    # res = chain.run("程序员")
    
    # 新版 LCEL 写法 (推荐)
    prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    chain = prompt | llm | StrOutputParser()
    
    res = chain.invoke({"topic": "程序员"})
    print(f"结果: {res}\n")

def demo_sequential_chain():
    """
    2. 顺序链 (Sequential Chain)
    作用：将一个链的输出作为下一个链的输入。
    """
    print("--- 2. 顺序链 Demo ---")
    
    # 步骤 1: 生成大纲
    prompt1 = ChatPromptTemplate.from_template("为主题 {topic} 写一个简单的博客大纲")
    chain1 = prompt1 | llm | StrOutputParser()
    
    # 步骤 2: 根据大纲写内容
    prompt2 = ChatPromptTemplate.from_template("根据以下大纲写一篇简短的博客文章:\n{outline}")
    chain2 = prompt2 | llm | StrOutputParser()
    
    # 使用 LCEL 连接 (chain1 的输出传递给 chain2)
    # 这里的 {"outline": chain1} 意味着 chain1 的输出会被赋值给 outline 变量
    final_chain = ({"outline": chain1} | chain2)
    
    res = final_chain.invoke({"topic": "AI 的未来"})
    print(f"结果 (部分): {res[:100]}...\n")

def demo_math_chain():
    """
    3. 数学链 (LLMMathChain)
    作用：让 LLM 结合 Python REPL 进行精确的数学计算。
    注意：这是 Legacy Chain，但在处理复杂计算时依然有用。
    """
    print("--- 3. 数学链 Demo ---")
    try:
        # LLMMathChain 通常需要 llm 实例
        llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)
        query = "123456 乘以 654321 等于多少？"
        res = llm_math.run(query)
        print(f"问题: {query}")
        print(f"结果: {res}\n")
    except Exception as e:
        print(f"数学链运行失败 (可能缺少 numexpr 库): {e}\n")

def demo_router_chain():
    """
    4. 路由链 (Router Chain) - 使用 LCEL 实现分支逻辑
    作用：根据用户输入的内容，动态选择不同的处理链（例如：物理问题用物理链，数学问题用数学链）。
    """
    print("--- 4. 路由链 (LCEL 分支) Demo ---")
    
    # 定义两个不同的处理链
    math_chain = ChatPromptTemplate.from_template("你是一个数学家，请解决这个数学问题: {input}") | llm | StrOutputParser()
    physics_chain = ChatPromptTemplate.from_template("你是一个物理学家，请解释这个物理现象: {input}") | llm | StrOutputParser()
    general_chain = ChatPromptTemplate.from_template("请回答这个问题: {input}") | llm | StrOutputParser()

    # 简单的路由逻辑函数
    def route(info):
        question = info["input"].lower()
        if "计算" in question or "数学" in question or "加" in question:
            return math_chain
        elif "物理" in question or "力" in question or "速度" in question:
            return physics_chain
        else:
            return general_chain

    # 使用 RunnableLambda 构建路由
    from langchain_core.runnables import RunnableLambda
    
    full_chain = RunnableLambda(route)
    
    print("测试数学问题:")
    print(full_chain.invoke({"input": "计算 1+1 等于几"}))
    
    print("测试物理问题:")
    print(full_chain.invoke({"input": "解释牛顿第二定律"}))
    print("\n")

def demo_sql_chain():
    """
    5. SQL 查询链 (create_sql_query_chain)
    作用：将自然语言转换为 SQL 查询语句。
    """
    print("--- 5. SQL 查询链 Demo ---")
    # 这里只是演示 API 用法，因为没有实际的数据库连接
    # db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    # chain = create_sql_query_chain(llm, db)
    # response = chain.invoke({"question": "How many employees are there"})
    print("代码示例 (需连接真实 DB):")
    print("""
    chain = create_sql_query_chain(llm, db)
    response = chain.invoke({"question": "How many employees are there"})
    """)
    print("\n")

def demo_document_chain():
    """
    6. 文档处理链 (create_stuff_documents_chain)
    作用：将一堆文档 (Documents) "塞入" (Stuff) 到 Prompt 中发送给 LLM。
    这是 RAG 中最基础的合并文档方式。
    """
    print("--- 6. 文档处理链 Demo ---")
    
    prompt = ChatPromptTemplate.from_template("""
    请根据以下上下文回答问题:
    <context>
    {context}
    </context>
    
    问题: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    docs = [
        Document(page_content="LangChain 是一个用于构建 LLM 应用的框架。"),
        Document(page_content="LangChain 支持 Python 和 JavaScript。")
    ]
    
    res = document_chain.invoke({
        "input": "LangChain 支持哪些语言？",
        "context": docs
    })
    
    print(f"结果: {res}\n")

if __name__ == "__main__":
    # 可以根据需要取消注释运行特定的 demo
    demo_basic_chain()
    demo_sequential_chain()
    demo_math_chain() 
    demo_router_chain()
    demo_sql_chain()
    demo_document_chain()
