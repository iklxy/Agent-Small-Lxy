# 导入节点函数
from text_agent import text_agent_node
from image_agent import image_agent_node
# 导入基本数据类型和类
from typing import Annotated,Literal,Sequence
from typing_extensions import TypeDict
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
# 导入状态图和结束常量
from langgraph.graph import StateGraph,END
from langgraph.graph.message import add_messages #add_messages确保消息正确添加到状态图中，且不会发生覆盖

#定义systemprompt
SUPERVISOR_AGENT_PROMPT = """"
    你是一个名为 lxy 的数字分身系统的路由总管.
    你的手下有两个专家:
    1. TextAgent:负责处理文字记忆、日记、技术经历、自我介绍等问题.
    2. ImageAgent:负责在相册中检索照片和视觉记录.
    请根据用户的最新对话,决定由谁来回答。如果问题已经解决或不需要他们处理,请回复FINISH.
"""

# 定义图状态类
class AgentState(TypeDict):
    """
    定义图状态，包含用户输入、模型响应等信息。
    """
    next : str
    message : Annotated[Sequence, add_messages]

# 定义Route
class Route(BaseModel):
    """
    定义路由，包含目标节点和消息。
    """
    next : Literal["text_agent","image_agent","FINISH"]

# 定义路由成员
members = ["text_agent","image_agent"]
options = ["FININSH"] + members

# LLM
llm = OllamaLLM(
    model="qwen3:1.7b"
    )

# supervisor_chain
supervisor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_AGENT_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "根据上面的对话,下一步该由谁来接手？"
            "只允许返回以下选项之一：{options}。"
        ),
    ])
    .partial(options=str(options), members=", ".join(members))
    | llm.with_structured_output(Route)
)

# 定义supervisor_node节点函数
def supervisor_node(state: AgentState) -> dict:
    """
    路由总管节点函数
    :param state: 输入状态，包含用户需求
    :return: 输出下一步应该由哪个agent来处理
    """
    route_decision = supervisor_chain.invoke(state)
    return {"next": route_decision.next}

# 构建状态图
graph = StateGraph(AgentState)

# 添加节点 添加边
graph.add_node("supervisor", supervisor_node)
graph.add_node("text_agent", text_agent_node)
graph.add_node("image_agent", image_agent_node)

# 添加边,text和image的Agent执行完任务之后都需要把结果返回给Supervisor
graph.add_edge("text_agent", "supervisor")
graph.add_edge("image_agent", "supervisor")

# 添加条件边,根据Supervisor的next字段来判断下一步该由谁来处理
graph.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "text_agent": "text_agent",
        "image_agent": "image_agent",
        "FINISH": END,
    },
)

# 添加入口节点
graph.add_edge("__start__", "supervisor")

# 运行图
real_graph = graph.compile()







