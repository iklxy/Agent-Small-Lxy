# LangChain 学习笔记

## 1. LangChain 调用 LLM 的核心接口

LangChain 提供了统一的 `Runnable` 协议，所有 LLM 对象（如 `ChatOpenAI`）都支持以下标准接口：

### (1) `invoke` (同步调用)
作用：最基础的调用方式，发送一个输入，等待并获取完整的输出。
场景：简单的问答，不需要流式输出，也不需要并发处理。
返回值：`AIMessage` 对象（包含 `content` 属性）。

### (2) `stream` (流式调用)
作用：以流的形式逐步返回结果，类似打字机效果。
场景：实时聊天应用，提升用户体验（不需要等整个答案生成完才显示）。
返回值：一个生成器（Iterator），每次迭代返回一个 `AIMessageChunk`。

### (3) `batch` (批量调用)
作用：同时处理多个输入列表。
场景：需要对一组数据进行相同的处理（如批量翻译、批量分类）。
特点：底层会自动并行化处理（如果支持），比循环调用 `invoke` 更快。

### (4) `ainvoke` / `astream` / `abatch` (异步版本)
作用：对应上述三个方法的异步版本，必须在 `async` 函数中使用 `await` 调用。
场景：高并发 Web 服务（如 FastAPI），需要同时处理大量请求而不阻塞主线程。

---

## 2. 同步调用 vs 异步调用

| 特性 | 同步调用 (Sync) | 异步调用 (Async) |
| :--- | :--- | :--- |
| 接口名称 | `invoke`, `stream`, `batch` | `ainvoke`, `astream`, `abatch` |
| 关键字 | 无 | `async`, `await` |
| 执行模式 | 阻塞式：必须等当前任务完成后，才能执行下一行代码。 | 非阻塞式：任务提交后立即返回控制权，可以在等待结果的同时处理其他任务。 |
| 耗时特征 | 串行执行，总耗时 = 任务1耗时 + 任务2耗时 + ... | 并发执行，总耗时 ≈ 最慢的那个任务耗时（取决于并发度）。 |
| 适用场景 | 脚本、简单工具、低并发应用、调试。 | 高并发 Web 服务、聊天机器人、需要同时调用多个外部 API 的 Agent。 |
| 代码复杂度 | 低，符合直觉。 | 中，需要理解 `asyncio`、事件循环、协程等概念。 |
| 示例 | `res = llm.invoke("hi")` | `res = await llm.ainvoke("hi")` |

### 核心区别总结
- 同步就像排队打饭，必须一个一个来。
- 异步就像餐厅点餐，点完餐后你可以玩手机（处理其他事），厨房做好后会叫号通知你（回调/await返回）。

---

## 3. LangChain 提示词模版 (Prompt Templates) 总结

提示词模版将“指令逻辑”与“动态数据”分离，是构建 LLM 应用的基础。

### (1) PromptTemplate (基础模版)
**适用场景**：生成单一字符串提示词，适用于非聊天模型（Completion）或简单任务。

- **定义模版**：使用 `from_template` 方法，用 `{variable}` 标记变量。
- **实例化**：调用 `invoke` 方法填入变量。
- **调用**：直接传入 LLM 或通过链式调用。

```python
from langchain_core.prompts import PromptTemplate

# 1. 定义模版
template = PromptTemplate.from_template("请把'{text}'翻译成{language}。")

# 2. 实例化 (填入变量)
prompt_value = template.invoke({"text": "Hello", "language": "中文"})
# 输出: "请把'Hello'翻译成中文。"

# 3. 调用
# res = llm.invoke(prompt_value)
```


### (2) ChatPromptTemplate (聊天模版)
**适用场景**：专为聊天模型（如 GPT-4）设计，包含 System、Human、AI 等多角色消息。

- **定义模版**：使用 `from_messages` 方法，传入 (角色, 内容) 的元组列表。
- **实例化**：调用 `invoke` 方法填入变量。
- **调用**：生成的是 `ChatPromptValue`（包含消息列表），更符合 Chat Model 的输入格式。

```python
from langchain_core.prompts import ChatPromptTemplate

# 1. 定义模版
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个精通{topic}的专家。"),
    ("human", "请简述{topic}的核心概念。"),
])

# 2. 实例化
messages = chat_template.invoke({"topic": "量子力学"})
# 输出: [SystemMessage(...), HumanMessage(...)]

# 3. 调用
# res = llm.invoke(messages)
```

### (3) FewShotPromptTemplate (少样本模版)
**适用场景**：通过提供示例（Few-Shot）让模型学习特定风格或逻辑（In-Context Learning）。

- **定义模版**：
    1.  定义单个示例的样式 (`example_prompt`)。
    2.  准备示例数据列表 (`examples`)。
    3.  组合成 `FewShotPromptTemplate`。
- **实例化**：填入用户的新输入。

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# 1. 准备示例数据
examples = [
    {"input": "高兴", "output": "悲伤"},
    {"input": "高", "output": "矮"},
]

# 2. 定义单个示例的格式
example_prompt = PromptTemplate.from_template("输入: {input}\n输出: {output}")

# 3. 组合 FewShot 模版
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="请给出以下词语的反义词：", # 前缀指令
    suffix="输入: {word}\n输出:",      # 后缀（用户输入位置）
    input_variables=["word"]
)

# 4. 实例化
final_prompt = few_shot_prompt.invoke({"word": "黑"})
# 输出包含前缀、所有示例和最后的"黑"
```

### (4) PipelinePromptTemplate (管道模版)
**适用场景**：复杂场景，将多个独立的模版（如角色定义、示例、具体任务）拼接成一个大模版。

- **定义模版**：准备多个子模版，定义最终的组合结构 (`final_prompt`)。
- **实例化**：一次性填入所有需要的变量。

```python
from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate

# 1. 定义子模版
full_template = PromptTemplate.from_template("{introduction}\n\n{example}\n\n{start}")
intro_template = PromptTemplate.from_template("你是一个{role}。")
example_template = PromptTemplate.from_template("例如：{input} -> {output}")
start_template = PromptTemplate.from_template("现在开始：{user_input}")

# 2. 定义管道映射
input_prompts = [
    ("introduction", intro_template),
    ("example", example_template),
    ("start", start_template)
]

# 3. 组合
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_template,
    pipeline_prompts=input_prompts
)

# 4. 实例化
res = pipeline_prompt.invoke({
    "role": "数学家",
    "input": "1+1",
    "output": "2",
    "user_input": "3+3"
})
### (5) MessagesPlaceholder (消息占位符)
**适用场景**：用于在 ChatPromptTemplate 中动态插入**消息列表**（如历史对话记录），而不是简单的字符串。这是实现**带记忆功能的聊天机器人**的关键组件。

- **定义模版**：在 `ChatPromptTemplate` 的消息列表中插入 `MessagesPlaceholder(variable_name="xxx")`。
- **实例化**：传入一个 `Message` 对象列表（如 `[HumanMessage(...), AIMessage(...)]`）。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# 1. 定义模版 (包含占位符)
chat_history_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的AI助手。"),
    MessagesPlaceholder(variable_name="history"), # <--- 占位符：此处将插入历史消息列表
    ("human", "{new_input}")
])

# 2. 准备历史消息数据
history_messages = [
    HumanMessage(content="我的名字叫小明。"),
    AIMessage(content="你好小明，很高兴认识你！")
]

# 3. 实例化 (填入历史记录列表)
prompt_with_history = chat_history_template.invoke({
    "history": history_messages, # 对应 variable_name="history"
    "new_input": "我叫什么名字？"
})

# 4. 调用
# res = llm.invoke(prompt_with_history)
```

---

## 4. LangChain 输出解析器 (Output Parsers) 总结

输出解析器负责将 LLM 返回的文本（通常是字符串）转换为结构化的数据（如 Python 列表、字典等），方便程序后续处理。

### (1) StrOutputParser (字符串解析器)
**作用**：最基础的解析器，直接提取 `AIMessage` 中的 `content` 内容，将其转换为纯字符串。
**场景**：大多数场景，特别是当你只需要 LLM 的文本回复时。

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
res = chain.invoke({...})
# res 类型: str
# res 值: "回答内容..."
```

### (2) CommaSeparatedListOutputParser (逗号分隔列表解析器)
**作用**：将 LLM 返回的“逗号分隔的字符串”自动解析为 Python 列表 (`list`)。
**场景**：需要 LLM 生成关键词列表、名字列表等。
**注意**：需要在 Prompt 中明确指示 LLM 使用“英文逗号”分隔。

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
prompt = ChatPromptTemplate.from_template("列出5个{topic}，用英文逗号分隔。")

chain = prompt | llm | output_parser
res = chain.invoke({"topic": "水果"})
# res 类型: list
# res 值: ['苹果', '香蕉', '橙子']
```

### (3) JsonOutputParser (JSON 解析器)
**作用**：将 LLM 返回的 JSON 格式字符串解析为 Python 字典 (`dict`)。
**场景**：需要从文本中提取结构化信息（如实体抽取）。
**配合 Pydantic**：可以传入 Pydantic 模型来生成 `format_instructions`，告诉 LLM 需要的 JSON 结构。

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 1. 定义期望的数据结构
class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")

parser = JsonOutputParser(pydantic_object=Person)

# 2. 将格式说明注入到 Prompt 中
prompt = ChatPromptTemplate.from_template(
    "生成一个人的信息。\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
res = chain.invoke({})
# res 类型: dict
# res 值: {'name': '张三', 'age': 25}
```

---

## 5. LCEL (LangChain Expression Language) 与链式结构 `|`

LangChain 使用 **管道符 (`|`)** 来构建链式调用，这种语法称为 **LCEL**。它让代码逻辑像流水线一样清晰，数据从左向右流动。

### (1) 核心公式
```python
chain = Prompt | LLM | OutputParser
```
1.  **Input** (字典) -> **Prompt** (格式化后的 PromptValue)
2.  **PromptValue** -> **LLM** (生成的 AIMessage)
3.  **AIMessage** -> **OutputParser** (最终的 Python 对象，如 str, list, dict)

### (2) 为什么使用 `|` 管道符？
*   **直观**：一眼就能看清数据处理的顺序。
*   **标准**：所有组件（Prompt, LLM, Parser）都实现了 `Runnable` 协议，可以无缝拼接。
*   **统一接口**：拼接后的 `chain` 对象依然支持 `invoke`, `stream`, `batch` 等标准方法。

### (3) 代码对比
**不使用 LCEL (旧式/繁琐写法):**
```python
prompt_value = prompt.invoke({"topic": "猫"})
message = llm.invoke(prompt_value)
result = output_parser.invoke(message)
```

**使用 LCEL (推荐写法):**
```python
chain = prompt | llm | output_parser
result = chain.invoke({"topic": "猫"})
```

---

## 6. LangChain Chain 模块详解

Chain 是 LangChain 中用于连接多个组件（Prompt, LLM, Tool 等）的核心概念。随着 LangChain 的发展，Chain 的实现方式也经历了从类（Class-based）到 LCEL（Function-based）的演变。

### (1) 基础链 (Basic Chain)
最简单的链，由 Prompt、LLM 和 OutputParser 组成。

*   **旧版写法**: `LLMChain` (已不推荐使用)
*   **新版写法 (LCEL)**: 使用管道符 `|`

```python
# 推荐写法 (LCEL)
prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
chain = prompt | llm | StrOutputParser()
res = chain.invoke({"topic": "程序员"})
```

### (2) 顺序链 (Sequential Chain)
将一个链的输出作为下一个链的输入，就像流水线作业。

*   **场景**: 先生成大纲，再根据大纲写文章。
*   **实现**: 使用 LCEL 的字典传递机制。

```python
# 链1: 生成大纲
chain1 = prompt1 | llm | StrOutputParser()

# 链2: 写文章 (输入需要 'outline')
chain2 = prompt2 | llm | StrOutputParser()

# 组合: chain1 的结果赋值给 'outline'，传递给 chain2
final_chain = ({"outline": chain1} | chain2)
res = final_chain.invoke({"topic": "AI"})
```

### (3) 数学链 (LLMMathChain)
**Legacy Chain**，但在处理精确计算时依然非常有用。它结合了 LLM 的理解能力和 Python 解释器（或 numexpr 库）的计算能力。

*   **场景**: 处理复杂的数学运算（LLM 经常算错数）。
*   **注意**: 需要安装 `numexpr` 库。

```python
from langchain.chains import LLMMathChain

llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)
res = llm_math.run("123456 乘以 654321 等于多少？")
```

### (4) 路由链 (Router Chain / Branching)
根据用户的输入内容，动态选择不同的处理路径（分支）。

*   **场景**: 用户问数学题走数学链，问物理题走物理链，闲聊走通用链。
*   **实现**: 使用 `RunnableLambda` 自定义路由逻辑。

```python
# 伪代码逻辑
def route(info):
    if "数学" in info["input"]:
        return math_chain
    elif "物理" in info["input"]:
        return physics_chain
    else:
        return general_chain

full_chain = RunnableLambda(route)
full_chain.invoke({"input": "牛顿第二定律是什么？"}) # 自动路由到 physics_chain
```

### (5) SQL 查询链 (create_sql_query_chain)
**新型链 (Factory Function)**，专门用于 Text-to-SQL 任务。它能将自然语言问题转化为 SQL 语句。

*   **场景**: 允许非技术人员查询数据库（"有多少员工？" -> "SELECT count(*) FROM employees"）。
*   **组件**: 结合 `SQLDatabase` 工具使用。

```python
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

# db = SQLDatabase.from_uri("sqlite:///Chinook.db")
chain = create_sql_query_chain(llm, db)
# response = chain.invoke({"question": "How many employees are there"})
```

### (6) 文档处理链 (create_stuff_documents_chain)
**新型链**，RAG (检索增强生成) 的核心组件之一。"Stuff" 意为将所有检索到的文档一次性“塞入” Prompt 的 Context 中。

*   **场景**: 回答基于文档的问题。
*   **原理**: `Prompt + Document List -> LLM`。

```python
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("基于以下内容回答: {context} 问题: {input}")
document_chain = create_stuff_documents_chain(llm, prompt)

# 传入文档列表
res = document_chain.invoke({
    "input": "LangChain 是什么？",
    "context": [Document(page_content="LangChain 是...")]
})
```

### (7) 代码示例参考
上述所有 Chain 的完整可运行代码（包含环境配置和调用示例）已整理在 `chains_learning.py` 文件中。
- **脚本路径**: [chains_learning.py](chains_learning.py)
- **运行方式**: `python chains_learning.py`

---

## 7. Memory 模块 (记忆)

Memory 是 LangChain 中用于让 LLM "记住" 之前对话内容的组件。由于 LLM 本身是无状态的（Stateless），每次请求都是独立的，因此需要额外的机制来存储和回传历史上下文。

### (1) 常见 Memory 类型

| 类型 | 类名 | 作用 | 优点 | 缺点 |
| :--- | :--- | :--- | :--- | :--- |
| **全量记忆** | `ConversationBufferMemory` | 完整记录所有对话历史。 | 信息最完整，逻辑连贯。 | 随着对话变长，Token 消耗巨大，容易超出上下文窗口限制。 |
| **窗口记忆** | `ConversationBufferWindowMemory` | 只保留最近 `k` 轮对话 (滑动窗口)。 | 有效控制 Token 消耗，防止溢出。 | 会丢失较早的对话细节，无法进行长跨度的指代。 |
| **摘要记忆** | `ConversationSummaryMemory` | 利用 LLM 对历史对话进行实时摘要，保留核心信息。 | 适合超长对话，Token 占用相对稳定且较小。 | 需要额外的 LLM 调用来生成摘要，增加延迟和成本；摘要可能丢失细节。 |

### (2) 两种使用方式

#### A. 传统方式 (配合 LLMChain)
将 Memory 对象直接传给 Chain。

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

memory = ConversationBufferMemory(memory_key="chat_history")
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
```

#### B. 现代方式 (LCEL + RunnableWithMessageHistory) 【推荐】
LangChain v0.1+ 推荐的做法。不将 Memory 绑定在 Chain 内部，而是作为外部工具挂载。

*   **核心组件**: `RunnableWithMessageHistory`
*   **原理**: 自动根据 `session_id` 读取历史记录 -> 注入 Prompt -> 执行 Chain -> 保存新的一轮对话。

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. 定义获取历史记录的方法 (通常对接 Redis/SQL)
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 2. 包装 Chain
chain_with_history = RunnableWithMessageHistory(
    runnable, # 原有的 Chain
    get_session_history,
    input_messages_key="input",
    history_messages_key="history" # 对应 Prompt 中的占位符
)

# 3. 调用
chain_with_history.invoke(
    {"input": "你好"},
    config={"configurable": {"session_id": "user_1"}}
)
```

---

## 8. Tool 模块 (工具)

Tool 是 Agent 的核心能力来源，允许 LLM 与外部世界（API、数据库、搜索引擎）进行交互。

### (1) 定义工具的三种方式

#### A. `@tool` 装饰器 (推荐用于简单函数)
最简单的方式，直接装饰一个 Python 函数。LLM 会自动读取函数的名称、参数类型提示 (Type Hint) 和文档字符串 (Docstring) 来理解工具。

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b

# 查看生成的 JSON Schema
# print(multiply.args) 
```

#### B. `StructuredTool` (推荐用于复杂参数)
当函数的参数比较复杂，或者需要更严格的参数验证时，可以使用 `StructuredTool` 结合 Pydantic 模型。

```python
from langchain_core.tools import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    limit: int = Field(default=5, description="Max results")

def search(query: str, limit: int = 5):
    return f"Searching {query}..."

search_tool = StructuredTool.from_function(
    func=search,
    name="custom_search",
    description="Search tool",
    args_schema=SearchInput
)
```

#### C. 继承 `BaseTool` 类
适用于需要维护状态或进行复杂初始化的工具（如数据库连接）。

### (2) 如何调用工具

#### A. 手动绑定与调用 (bind_tools)
让模型知道工具有哪些，但不自动执行。模型会返回一个 `tool_calls` 消息，告诉你它想调用哪个工具。

```python
tools = [multiply]
llm_with_tools = llm.bind_tools(tools)
res = llm_with_tools.invoke("5 乘以 8 是多少？")
# res.tool_calls -> [{'name': 'multiply', 'args': {'a': 5, 'b': 8}, ...}]
```

#### B. Agent 自动执行 (create_tool_calling_agent)
使用 AgentExecutor 来自动处理 "模型思考 -> 调用工具 -> 获取结果 -> 模型生成回答" 的循环。

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent

# 1. 创建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 2. 创建执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 3. 执行
agent_executor.invoke({"input": "北京天气如何？"})
```

---

## 9. Agent 模块 (智能体)

Agent 是 LangChain 的大脑，它使用 LLM 来决定采取什么行动（Action），执行该行动，观察结果（Observation），并重复此过程直到完成任务。

### (1) Agent 的类型

#### A. Tool Calling Agent (Function Calling Agent) 【推荐】
*   **适用场景**: 使用 OpenAI (gpt-3.5/4) 等支持 Function Calling 的模型。
*   **特点**: 结构化输出，极其稳定，不易出错。
*   **原理**: 利用模型原生的 `tool_calls` 能力。

#### B. ReAct Agent (Reasoning + Acting)
*   **适用场景**: 通用 LLM (如 Llama 2, Claude 2)，或者需要显式展示推理过程。
*   **特点**: 基于 "Thought -> Action -> Observation" 的循环。
*   **缺点**: 对 Prompt 格式要求严格，有时会解析失败。

### (2) Agent 的创建与使用

#### 通用步骤
1.  **定义工具 (Tools)**: 准备好 Agent 可以使用的工具列表。
2.  **选择模型 (LLM)**: 初始化 ChatModel。
3.  **构建 Prompt**: 包含必要的指令和占位符 (如 `{agent_scratchpad}`)。
4.  **创建 Agent**: 使用 `create_tool_calling_agent` 或 `create_react_agent`。
5.  **创建执行器 (AgentExecutor)**: 负责运行 Agent 的循环。

#### 代码示例 (Tool Calling Agent)

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub

# 1. 准备工具
tools = [multiply, search]

# 2. 获取 Prompt (标准模板)
prompt = hub.pull("hwchase17/openai-tools-agent")
# 或者自定义 Prompt，必须包含 MessagesPlaceholder(variable_name="agent_scratchpad")

# 3. 创建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 4. 创建 Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. 执行 (自动多步推理)
res = agent_executor.invoke({
    "input": "先计算 5 * 5，然后查询这个数字相关的历史事件"
})
print(res["output"])
```

### (3) 单工具 vs 多工具
*   **单工具**: 传入 `tools=[tool1]`。Agent 只能选择用或不用这个工具。
*   **多工具**: 传入 `tools=[tool1, tool2, ...]`。Agent 会根据问题自动决策：
    *   只用 tool1
    *   只用 tool2
    *   先用 tool1，拿到结果后再用 tool2 (链式调用)
    *   并行调用 (部分 Agent 支持)


---

## 10. Retrieval 模块 (检索)

Retrieval (检索) 是 RAG (检索增强生成) 的基石。它允许 LLM 访问训练数据之外的私有数据。

### (1) 文档加载器 (Document Loaders)
负责从不同数据源加载数据，并将其封装为 `Document` 对象。`Document` 包含 `page_content` (内容) 和 `metadata` (来源、页码等)。

*   `TextLoader`: 加载简单文本 (.txt)
*   `PyPDFLoader`: 加载 PDF
*   `WebBaseLoader`: 加载网页
*   `DirectoryLoader`: 加载整个文件夹

### (2) 文档转换器 (Text Splitters)
由于 LLM 有上下文窗口限制，需要将长文档切分为较小的 Chunks。

*   `CharacterTextSplitter`: 简单按字符/分隔符切分。
*   `RecursiveCharacterTextSplitter` (**推荐**): 智能递归切分。先按段落切，段落太长切句子，句子太长切单词。尽量保持语义完整性。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.create_documents([text])
```

### (3) 文本嵌入模型 (Embeddings)
将文本转换为向量 (Vector/Embedding)。语义相似的文本在向量空间中距离更近。

*   `OpenAIEmbeddings`: OpenAI 的嵌入模型 (text-embedding-3-small 等)。
*   `HuggingFaceEmbeddings`: 使用开源模型 (本地运行)。

### (4) 向量存储 (Vector Stores)
用于存储和检索 Embedding 向量的数据库。

*   `Chroma`: 轻量级，支持本地文件存储，适合开发测试。
*   `FAISS`: Facebook 开源的高效向量检索库，适合大规模数据。
*   `Pinecone/Milvus`: 生产级向量数据库服务。

### (5) 检索器 (Retrievers)
Retriever 是连接向量数据库和 Chain 的接口。它接收查询 (Query)，返回相关的 Document 列表。

*   **基础检索**: `vectorstore.as_retriever()`
*   **高级检索**:
    *   `MultiQueryRetriever`: 自动生成多个版本的 Query 进行检索。
    *   `ContextualCompressionRetriever`: 检索后对文档进行压缩/过滤，只返回相关片段。

### (6) 完整 RAG 流程代码

```python
# 1. 加载 -> 切分 -> 向量化 -> 存储
loader = TextLoader("data.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
splits = splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 2. 创建检索器
retriever = vectorstore.as_retriever()

# 3. 构建 RAG Chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("基于上下文回答: {context} 问题: {input}")
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 4. 提问
res = retrieval_chain.invoke({"input": "我的数据里说了什么？"})
```