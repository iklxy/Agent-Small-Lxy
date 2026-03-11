my_ai_twin/
│
├── api/                    # 1. API 接口层 (FastAPI)
│   ├── main.py             # FastAPI 启动入口
│   ├── routes.py           # 定义 /chat 等接口路由
│   └── schemas.py          # Pydantic 数据模型 (定义请求和返回的数据格式)
│
├── core/                   # 2. 核心配置层
│   ├── config.py           # 环境变量 (API Key、数据库路径等)
│   └── state.py            # LangGraph 的全局状态定义 (State)
│
├── agents/                 # 3. Agent 逻辑层
│   ├── supervisor.py       # 负责路由决策的 Supervisor Agent
│   ├── text_agent.py       # 处理文字 RAG 的 Agent
│   └── image_agent.py      # 处理图片检索的 Agent
│
├── tools/                  # 4. 工具层 (被 Agent 调用)
│   ├── text_retriever.py   # 文本向量检索逻辑
│   └── image_retriever.py  # 图像元数据检索逻辑
│
├── data/                   # 5. 数据存储层
│   ├── raw_texts/          # 原始日记、文档 (.txt, .md)
│   ├── raw_images/         # 原始照片 (.jpg, .png)
│   └── vector_db/          # 向量数据库持久化文件 (FAISS/Chroma)
│
├── scripts/                # 6. 离线脚本 (极其重要)
│   ├── ingest_text.py      # 文本数据清洗、切分与入库脚本
│   └── ingest_images.py    # 调用视觉模型提取图片描述，并入库的脚本
│
├── requirements.txt        # 依赖包列表
└── README.md               # 项目说明文档