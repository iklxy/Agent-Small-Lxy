import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def demo_document_loader():
    """
    1. 文档加载器 (Document Loaders)
    作用：将不同格式的数据源加载为 Document 对象列表。
    Document 对象包含两个属性：page_content (内容) 和 metadata (元数据)。
    """
    print("--- 1. Document Loader Demo ---")
    
    # 1.1 加载文本文件
    loader = TextLoader("MyLife.txt", encoding="utf-8")
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from text file.")
    print(f"Metadata: {docs[0].metadata}")
    print(f"Content Preview: {docs[0].page_content[:50]}...\n")

    # 1.2 加载 PDF (需要安装 pypdf)
    # loader = PyPDFLoader("path/to/your.pdf")
    # docs = loader.load()
    
    # 1.3 加载网页 (需要安装 beautifulsoup4)
    # loader = WebBaseLoader("https://python.langchain.com")
    # docs = loader.load()

def demo_text_splitter():
    """
    2. 文档转换器 (Text Splitters)
    作用：将长文档切分为较小的块 (Chunks)，以便放入向量数据库和 LLM 的上下文窗口中。
    """
    print("--- 2. Text Splitter Demo ---")
    
    # 模拟一个长文档
    long_text = "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。\n\n它使应用程序能够：\n1. 具有上下文感知能力：将语言模型连接到上下文来源（提示指令，少量的示例，内容等）。\n2. 具有推理能力：依赖语言模型进行推理（根据提供的上下文如何回答，采取什么行动等）。"
    
    # 2.1 CharacterTextSplitter (按字符切分)
    # 简单粗暴，按分隔符切分。
    c_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=50,
        chunk_overlap=0
    )
    docs = c_splitter.create_documents([long_text])
    print(f"Character Splitter: Created {len(docs)} chunks.")
    
    # 2.2 RecursiveCharacterTextSplitter (递归字符切分 - 推荐)
    # 智能切分，尝试按段落 -> 句子 -> 单词 的顺序切分，尽量保持语义完整。
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10 # 重叠部分，防止切分点丢失语义
    )
    docs = r_splitter.create_documents([long_text])
    print(f"Recursive Splitter: Created {len(docs)} chunks.")
    print(f"Chunk 1: {docs[0].page_content}")
    print(f"Chunk 2: {docs[1].page_content}\n")

def demo_vector_store_retriever():
    """
    3. 嵌入模型 (Embeddings) & 4. 向量存储 (Vector Stores) & 5. 检索器 (Retrievers)
    这是 RAG 的核心流程：
    Text -> Embedding -> Vector Store -> Retriever -> LLM
    """
    print("--- 3,4,5. RAG Pipeline Demo ---")
    
    # 3. 初始化 Embedding 模型
    # 将文本转换为向量。常用的有 OpenAIEmbeddings, HuggingFaceEmbeddings 等。
    embeddings = OpenAIEmbeddings()
    
    # 准备数据
    texts = [
        "LangChain 支持 Python 和 JavaScript。",
        "LangChain 的核心组件包括 Chains, Agents, Retrieval 等。",
        "lxy 是一名热爱技术的程序员。",
        "lxy 正在构建一个 AI 数字孪生项目。"
    ]
    
    # 4. 创建向量存储 (Vector Store)
    # 这里使用 Chroma (轻量级，本地文件存储)。需要安装 chromadb。
    # from_texts 会自动调用 embeddings 将文本向量化并存储。
    db = Chroma.from_texts(texts, embeddings)
    
    # 5. 创建检索器 (Retriever)
    # 检索器是连接向量数据库和 Chain 的接口。
    retriever = db.as_retriever(search_kwargs={"k": 2}) # k=2 表示返回最相似的 2 个文档
    
    # 测试检索
    query = "lxy 是谁？"
    docs = retriever.invoke(query)
    print(f"Query: {query}")
    print(f"Retrieved {len(docs)} docs:")
    for doc in docs:
        print(f"- {doc.page_content}")
    print("\n")
    
    # --- 6. 构建完整的 RAG Chain ---
    print("--- 6. Full RAG Chain ---")
    
    # 6.1 创建 RAG Prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}
    """)
    
    # 6.2 创建文档处理链 (Stuff Documents Chain)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 6.3 创建检索链 (Retrieval Chain)
    # 它会自动：1. 拿 input 去 retriever 检索 -> 2. 把检索到的 docs 放入 context -> 3. 调用 document_chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # 6.4 执行
    res = retrieval_chain.invoke({"input": "lxy 在做什么项目？"})
    print(f"Answer: {res['answer']}")

if __name__ == "__main__":
    # demo_document_loader()
    # demo_text_splitter()
    demo_vector_store_retriever()