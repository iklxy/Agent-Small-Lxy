# 个人数字分身 (AI Digital Twin) 开发路线图

## 1. 项目愿景
打造一个“懂你人生经历”的智能体，基于 lxy 的个人数据（经历、生活、性格），实现真正的数字孪生。

## 2. 技术架构概览
- **核心模式**: Agentic RAG (代理式检索增强生成)
- **开发语言**: Python (Agent 逻辑), Go/C++ (后端/高性能模块)
- **框架**: LangChain

---

## 3. 开发阶段规划

### 阶段一：数据层（你的“数字人生”底座）
**目标**：将非结构化的个人记忆转化为机器可读的结构化数据。

*   **任务 1.1: 数据收集与清洗**
    *   **行动**: 整理照片(OCR)、日记、朋友圈备份、简历、个人作品集。
    *   **工具**: Python 脚本, OCR 工具。
*   **任务 1.2: 结构化提取 (你现在的能力范围)**
    *   **行动**: 使用 LangChain 的 `JsonOutputParser` 编写脚本，将纯文本日记提取为 JSON 格式。
    *   **示例**: `{"时间": "2010", "事件": "第一次编程", "情绪": "兴奋"}`
*   **任务 1.3: 知识库构建 (Standard RAG)**
    *   **知识点**: Text Splitters (文本切分), Embeddings (向量化)。
    *   **工具**: Vector Store (FAISS, Chroma 或 Pinecone)。

### 阶段二：检索增强层（从“死书”到“活查”）
**目标**：让 Agent 能精准回答跨度大、模糊的记忆问题。

*   **任务 2.1: 高级检索策略 (Advanced RAG)**
    *   **知识点**: `MultiQueryRetriever` (多角度提问), `Self-Querying` (元数据过滤)。
    *   **场景**: 解决“我初中三年最喜欢的老师是谁？”这类需要综合检索的问题。
*   **任务 2.2: 长短期记忆 (Memory)**
    *   **知识点**: `ConversationBufferMemory` (短期), 数据库存储 (长期)。
    *   **效果**: Agent 记得五分钟前聊过的话题。

### 阶段三：Agent 核心层（赋予它“思考”能力）
**目标**：从“搜索引擎”进化为“人生助理”。

*   **任务 3.1: 工具调用 (Tool Use)**
    *   **行动**: 开发工具函数 `Search_My_Life_Docs`, `Calculate_Age`, `Get_Current_Date`。
    *   **逻辑**: 闲聊 -> 直接回答; 问经历 -> 调用检索工具。
*   **任务 3.2: 推理框架 (ReAct)**
    *   **知识点**: `create_react_agent`。
    *   **行动**: 编写 System Prompt 设定人格（“你是一个温和的记录者...”）。

### 阶段四：进阶与工程化（发挥你的后端专长）
**目标**：构建生产级应用。

*   **任务 4.1: 复杂逻辑流 (LangGraph)**
    *   **场景**: 检索不到信息时主动反问 (Clarification Loop)。
*   **任务 4.2: 全栈部署**
    *   **后端**: Go (Gin/GRPC) 封装 Agent 接口。
    *   **前端**: Vue/React 或 C++ QT 客户端。

---

## 4. 当前可行性分析与下一步建议

### 你的现状
*   **已掌握**: Model I/O (Prompt, ChatModel), Output Parser (Json/Str)。
*   **优势**: 熟悉 Go/C++ 后端开发，具备工程化落地能力。

### 立即行动 (Action Items)
1.  **[代码] 数据提取脚本**: 利用你学过的 `JsonOutputParser`，写一个 Python 脚本，读取 `MyLife.txt`，将其中的“项目经历”部分提取为标准的 JSON 列表。
2.  **[学习] RAG 基础**: 接下来重点学习 LangChain 的 `Retrieval` 模块 (Loader -> Splitter -> Embedding -> VectorStore)。