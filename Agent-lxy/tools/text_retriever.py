import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tool

# -- 配置 
# 获取当前脚本所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 从 old_text 读取，使用绝对路径
DB_PATH = os.path.join(BASE_DIR, "../data/vector_db")
# 定义集合名称，用于存储向量
COLLECTION_NAME = "agent_memory"
# 嵌入模型名称，可以在此处换为自己本地的模型，前提是具有嵌入功能
EMBEDDING_MODEL_NAME = "qwen3-embedding:0.6b"
# ----------------

@tool
def text_retriever(query: str) -> str:
    """
    从向量数据库中检索与查询相关的文本。
    :param query: 用户查询
    :return: 相关文本
    """
    # 初始化嵌入模型
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    # 加载 Chroma 向量数据库
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    # 执行查询
    results = db.similarity_search(query, k=3)
    # 提取文本内容
    retrieved_texts = [doc.page_content for doc in results]
    return "\n".join(retrieved_texts)