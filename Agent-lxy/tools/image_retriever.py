import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tool

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "../data/vector_db")
COLLECTION_NAME = "agent_memory"
EMBEDDING_MODEL_NAME = "qwen3-embedding:0.6b" # 需与文本入库保持一致
DESCRIPTION_MODEL_NAME = "qwen3.5:0.8b" # 用于生成图片描述
# ----------------

@tool
def retrieve_images(query: str, n_results: int = 1) -> str:
    """
    根据用户的自然语言描述，在相册中检索最相关的照片。
    
    :param query: 描述想要查找的照片内容的关键词或句子（例如：“我和张宁远的小学合照”、“去海边玩的照片”）。
    :param n_results: 需要返回的照片数量，默认为 1 张。如果用户想看“一些”照片，可以设为 3。
    :return: 返回包含图片路径和简要描述的 Markdown 格式字符串。
    """
    
    # 1. 初始化数据库连接
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )
    
    # 2. 执行语义搜索
    # 只关心 type="image" 的记录
    results = vector_store.similarity_search(
        query,
        k=n_results,
        filter={"type": "image"} # 只搜图片，不搜纯文本记忆
    )
    
    if not results:
        return "抱歉，我的相册里好像没有找到符合描述的照片。"

    # 3. 格式化输出
    response_content = []
    for doc in results:
        # 获取元数据
        img_path = doc.metadata.get("source", "")
        filename = doc.metadata.get("filename", "未知图片")
        description = doc.page_content
    
        item = f"![{filename}]({img_path})\n> 🖼️ **回忆描述**: {description[:100]}..."
        response_content.append(item)
        
    return "\n\n---\n\n".join(response_content)

