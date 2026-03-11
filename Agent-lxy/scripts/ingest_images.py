import os
import base64
import glob
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage



# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/old_image")
DB_PATH = os.path.join(BASE_DIR, "../data/vector_db")
COLLECTION_NAME = "agent_memory"
EMBEDDING_MODEL_NAME = "qwen3-embedding:0.6b" # 需与文本入库保持一致
DESCRIPTION_MODEL_NAME = "qwen3.5:0.8b" # 用于生成图片描述
# ----------------

def encode_image(image_path):
    """将图片转换为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_description(vision_model, image_path, sidecar_text=None):
    """调用模型获取图片描述"""
    base64_image = encode_image(image_path)
    
    # 构建提示词
    prompt = "请详细描述这张图片的内容。包括场景、人物（如果有）、动作、氛围、时间点以及任何显著的细节。请用第一人称‘我’的视角描述，假设这是我的相册回忆。不要加前缀，直接输出描述内容。"
    
    if sidecar_text:
        prompt = f"背景信息：{sidecar_text}\n\n基于以上背景信息{prompt}"

    # 构建多模态消息
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )
    try:
        response = vision_model.invoke([message])
        return response.content
    except Exception as e:
        print(f"调用失败 ({image_path}): {e}")
        return None

def ingest_images():

    print(f"开始处理图片数据，来源: {DATA_PATH}")
    
    # 初始化
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    vision_model = ChatOllama(model=DESCRIPTION_MODEL_NAME)
    
    # 1. 获取所有图片
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(DATA_PATH, ext)))
    
    print(f"找到 {len(image_paths)} 张图片")
    
    if not image_paths:
        return

    # 2. 逐张处理
    # docs 用于存储所有图片的描述文档
    docs = []

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        print(f"[{idx+1}/{len(image_paths)}] 正在分析: {img_name} ...")
        
        # 查找同名的 .txt 描述文件
        base_name = os.path.splitext(img_path)[0]
        sidecar_path = base_name + ".txt"
        sidecar_text = None
        
        if os.path.exists(sidecar_path):
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sidecar_text = f.read().strip()
                print(f"找到背景信息: {sidecar_text[:30]}...")
        # --------------------------------------
        
        description = get_image_description(vision_model, img_path, sidecar_text)
        
        if description:
            print(f"  -> 描述: {description[:50]}...")
            
            # 创建 Document 对象
            # Metadata 中存储了图片的相对路径，这样 Agent 检索到描述时，就知道是哪张图
            doc = Document(
                page_content=description,
                metadata={
                    "source": img_path,
                    "type": "image",  # 标记为图片类型
                    "filename": os.path.basename(img_path)
                }
            )
            docs.append(doc)
        else:
            print(f"  -> 跳过")

    # 3. 存入 ChromaDB
    if docs:
        print(f"正在存入向量数据库")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=DB_PATH,
        )
        
        vector_store.add_documents(documents=docs)
        print(f"图片入库完成！共存储 {len(docs)} 张图片的描述。")
    else:
        print("没有成功生成任何图片描述。")

if __name__ == "__main__":
    ingest_images()
