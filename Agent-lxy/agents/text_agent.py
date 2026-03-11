#定义的是文本处理Agent，用于处理文本相关的需求
from langchain_ollama import ChatOllama # 导入OllamaLLM
from langchain_core.prompts import ChatPromptTemplate # 导入ChatPromptTemplate
from langchain.agents import create_agent # 导入create_agent
from langchain.messages import SystemMessage # 导入SystemMessage
# 导入工具
from tools.text_retriever import text_retriever

#定义systemprompt
TEXT_AGENT_PROMPT = """你是一个专门负责处理 lxy 图像记忆的 AI 分身助理。
你的任务是准确、温和地回答关于 lxy 过往经历、日记、技术栈等文本类问题。

工作纪律：
1. 必须优先使用 `text_retriever` 工具来获取事实，不要胡编乱造。
2. 如果检索不到相关信息，请直接坦白“我的文字记忆中没有找到相关记录”。
3. 你的回答需要符合 lxy 的身份背景。
4. 不要返回任何图片链接，那不是你的工作范围。
"""
# LLM
llm = ChatOllama(
    model="qwen3:1.7b"
)

# 定义工具列表
TOOLS= [text_retriever]

# Agent
text_agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=SystemMessage(content=TEXT_AGENT_PROMPT), #将系统提示词加入到Agent的每一轮对话中，确保准确性
)

# 暴露给supervisor_agent的节点函数
def text_agent_node(state: dict)->dict:
    """
    文本处理Agent节点函数
    :param state: 输入状态，包含用户需求
    :return: 输出状态，包含处理结果
    """
    return 