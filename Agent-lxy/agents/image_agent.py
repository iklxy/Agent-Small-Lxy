#定义的是图像处理Agent，用于处理图像相关的需求
from langchain_ollama import OllamaLLM # 导入OllamaLLM
from langchain_core.prompts import ChatPromptTemplate # 导入ChatPromptTemplate
from langchain.agents import create_react_agent # 导入create_react_agent
from langchain.messages import SystemMessage # 导入SystemMessage
from tools.image_retriever import search_image_memory # 导入search_image_memory

#定义systemprompt
IMAGE_AGENT_PROMPT = """你是一个专门负责处理 lxy 视觉记忆（照片与图像）的 AI 分身助理。
你的任务是根据用户的需求（如时间、地点、情绪、特定事件），精准地在相册库中检索并返回最符合描述的照片。

工作纪律：
1. 必须优先调用 `search_image_memory` 工具来获取真实的图片路径和元数据，绝不允许凭空捏造图片链接或文件名。
2. 如果检索不到符合条件的照片，请直接坦白“我的相册中暂时没有找到符合该画面的照片记录”。
3. 你的回答需要符合 lxy 的身份背景，保持温和且理性的基调。
4. 你的核心交付物是【图片】。请使用 Markdown 格式返回找到的图片（格式：![图片描述](图片相对路径)），并只配上简短的背景说明（如时间、地点）。
5. 绝不要长篇大论地讲述文字故事或经历细节，那是 Text Agent 的工作范围。
"""

# LLM
llm = OllamaLLM(
    model="qwen3:1.7b"
)

# 定义工具列表
TOOLS = [search_image_memory]

# Agent
image_agent = create_react_agent(
    llm=llm,
    tools=TOOLS,
    state_modifier=SystemMessage(content=IMAGE_AGENT_PROMPT), #将系统提示词加入到Agent的每一轮对话中，确保准确性
)

# 暴露给supervisor_agent的节点函数
def image_agent_node(state: dict)->dict:
    """
    图像处理Agent节点函数
    :param state: 输入状态，包含用户需求
    :return: 输出状态，包含处理结果
    """
    return