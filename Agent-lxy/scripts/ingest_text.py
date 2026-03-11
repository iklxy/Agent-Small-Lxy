import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader # 用于加载文本文件和目录
from langchain_text_splitters import RecursiveCharacterTextSplitter # 用于递归切分文本
from langchain_ollama import OllamaEmbeddings # 用于生成文本嵌入向量
from langchain_chroma import Chroma # 用于存储和检索向量
from langchain_core.documents import Document # 用于表示文档，所有信息都被包装为一个Document对象


# --- 配置 ---
# 获取当前脚本所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 从 old_text 读取，使用绝对路径
DATA_PATH = os.path.join(BASE_DIR, "../data/old_text")
DB_PATH = os.path.join(BASE_DIR, "../data/vector_db")
# 定义集合名称，用于存储向量
COLLECTION_NAME = "agent_memory"
# 嵌入模型名称，可以在此处换为自己本地的模型，前提是具有嵌入功能
EMBEDDING_MODEL_NAME = "qwen3-embedding:0.6b"
# ----------------

def ingest_texts():
    print(f"开始处理文本数据，来源: {DATA_PATH}")
    
    # 1. 加载数据
    # 默认加载 .txt 和 .md 文件
    loaders = [
        DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
    ]
    
    docs = []
    for loader in loaders:
        try:
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            print(f"  - 加载了 {len(loaded_docs)} 个文档")
        except Exception as e:
            print(f"  - 加载出错: {e}")

    if not docs:
        print(f"没有找到任何文本文件,请添加对应的 .txt 或 .md 文件到 {DATA_PATH} 目录")
        return

    # 2. 文本切分
    # 将长文本切分为小块，每块 500 字符，重叠 50 字符
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "!", "?", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"共切分为 {len(splits)} 个片段")

    # 3. 添加元数据 (Metadata)
    # 我们给每个片段加一个 type: text 标签，方便后续区分是纯文本还是图片描述
    for split in splits:
        split.metadata["type"] = "text"

    # 4. 初始化 Embedding 模型
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

    # 5. 存入 ChromaDB
    print(f"存入向量数据库")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )
    
    # 分批插入，避免一次性过大
    batch_size = 100
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        vector_store.add_documents(documents=batch)
        print(f"  - 已存储批次 {i//batch_size + 1}/{(len(splits)-1)//batch_size + 1}")

    print("入库完成")

if __name__ == "__main__":
    ingest_texts()
