# ScholarSearchX: 混合与知识图谱增强的学术搜索引擎

## 1. Introduction (项目背景)
随着学术文献数量的呈指数级增长，研究人员在海量数据中准确、快速地找到所需文献变得越来越困难。传统的基于关键字的检索系统（如简单的布尔检索或词频统计）往往无法理解查询的深层语义，也无法捕捉到作者、论文、研究概念之间错综复杂的网络关系。
本项目旨在开发一个名为 **ScholarSearchX** 的智能学术搜索引擎。它不仅巩固了传统信息检索（IR）的核心技术，还引入了前沿的语义向量检索、知识图谱（Knowledge Graph, KG）以及检索增强生成（RAG）技术，为用户提供精准的文献检索与智能问答体验。

## 2. Related Work (相关工作与文献研究)
在学术检索领域，技术演进经历了以下几个重要阶段：
1. **传统词法检索**：以 TF-IDF 和 BM25 为代表，基于词频和逆文档频率进行打分。虽然速度快、对精确匹配有效，但存在“词汇鸿沟”（Vocabulary Mismatch）问题。
2. **语义向量检索（Dense Retrieval）**：利用预训练语言模型（如 BERT, Nomic-Embed）将文本映射到高维稠密向量空间，通过余弦相似度等计算语义相关性，解决了同义词和表达多样性的问题。
3. **知识图谱增强（KG-Enhanced Search）**：学术实体（如作者、机构、论文、会议、概念）天然构成了一个庞大的图网络。引入图数据库可以实现基于关系的推理和推荐（如：“寻找与X学者合作过并且研究Y领域的作者的论文”）。
4. **大模型生成与检索增强（LLM & RAG）**：结合大型语言模型的总结与推理能力，对检索到的Top-K文献进行阅读理解并生成自然语言回答，极大提升了文献调研的效率。

本项目集成了上述所有演进阶段的技术，形成了一个多路召回与融合的混合学术搜索架构。

## 3. 课程相关技术应用 (与课程内容结合)
本项目紧密结合了《信息检索》课程的核心知识点，并将其落地为实际代码：
* **文本预处理与倒排索引 (对应早期课程)**：实现了基于词根提取、停用词过滤的文本处理管道，并从零构建了基于内存的倒排索引（Inverted Index）。
* **布尔检索 (Boolean Retrieval)**：支持 AND/OR/NOT 逻辑的精确学术文献过滤。
* **向量空间模型 (Vector Space Model & TF-IDF)**：实现了基于 TF-IDF 权重的文档向量化与余弦相似度打分机制。
* **检索评价指标 (Evaluation Metrics)**：集成了 MAP (Mean Average Precision) 和 NDCG@10 等课程强调的核心评价指标，用于量化对比不同检索策略的效果。
* **查询扩展与伪相关反馈 (Query Expansion / Rocchio)**：探讨了如何利用初步检索结果进行查询词权重的重构与扩展。

## 4. 超越课堂的前沿技术 (Advanced Technologies: KG & RAG)
为了让搜索引擎具备工业级的智能与可扩展性，本项目在课程基础之上引入了以下高级技术栈：
* **知识图谱存储与图查询 (Neo4j)**：
  - 构建了包含 `Paper`（论文）、`Author`（作者）、`Concept`（研究概念）节点的结构化学术图谱。
  - 使用 Cypher 语言支持复杂的关系跳跃查询。
* **稠密向量数据库 (Qdrant)**：
  - 引入了专用的向量搜索引擎 Qdrant，支持海量文本嵌入向量的毫秒级近似最近邻（ANN）检索。
* **本地大语言模型与 RAG (Ollama + Qwen2.5)**：
  - 实现了检索增强生成（RAG）管道。用户提问不仅会召回相关文献，还会将文献内容作为上下文喂给本地部署的 LLM（Qwen2.5:0.5b），直接生成结构化的文献综述或问题解答，保护数据隐私的同时提供智能交互。
* **容器化编排 (Docker & Docker Compose)**：
  - 使用 Docker Compose 将图数据库（Neo4j）、向量数据库（Qdrant）和大模型服务（Ollama）一键拉起，保证了复杂环境下的可移植性与易部署性。

## 5. 目前实现的功能与可视化展示
目前项目已经具备完整的端到端运行能力，可直接用于中期汇报展示：
1. **多路混合检索 Web UI (基于 Streamlit)**：
   - 提供了直观的网页端交互界面。
   - 支持单选/多选不同的检索算法（布尔、TF-IDF、Qdrant 向量检索、混合检索）。
   - 提供专属的 "RAG 问答" 模块，输入学术问题即可获取大模型的综合解答。
2. **知识图谱可视化展示 (基于 Neo4j Browser)**：
   - 通过 `http://localhost:7474`，可以直观地探索学术图谱。
   - 汇报时可展示的 Cypher 查询示例：
     ```cypher
     // 查找带有特定概念的所有论文及作者网络
     MATCH p=(c:Concept)-[:HAS_CONCEPT]-(paper:Paper)-[:AUTHORED_BY]-(a:Author) 
     RETURN p LIMIT 25
     ```

## 6. Future Work (未来工作计划)
在课程的后半段及期末验收前，本项目计划在以下方面进行深化：
1. **构建标准化的人工评测集 (Manual Evaluation Dataset)**：
   - 目前的评价多为定性观察。下一步将制定明确的“信息需求（Information Needs）”作为查询集，并通过人工标注构建标准的 `qrels`（Query Relevance Judgments）文件。
   - 在此标准数据集上全面运行 MAP 和 NDCG 评测，以量化证明 Hybrid + KG 相比纯 TF-IDF 的性能提升。
2. **学术数据的指数级扩展 (Data Scaling)**：
   - 目前系统内包含约 500 条 arXiv 样本数据。
   - 未来计划对接 arXiv API 或 Semantic Scholar API，将数据规模扩展至**近五年所有开源的 AI/IR 领域文献**，以验证系统的工业级大数据承载与检索能力。
3. **深度的图与向量融合 (Deep Graph-Vector Fusion)**：
   - 探索图神经网络（GNN）生成的图嵌入向量与文本语义向量的融合，进一步提高查询意图理解的准确性。

---

## 7. Code Structure (代码结构说明)
```text
ScholarSearchX/
├── data/                       # 存放原始论文数据及中间结果
├── src/hybrid_search/          # 核心源代码目录
│   ├── app/                    # Web 前端
│   │   └── streamlit_app.py    # Streamlit UI 入口页面
│   ├── index/                  # 传统 IR 模块
│   │   └── inverted_index.py   # 倒排索引与 TF-IDF 实现
│   ├── vector/                 # 向量检索模块
│   │   └── qdrant_store.py     # Qdrant 向量库对接
│   ├── kg/                     # 知识图谱模块
│   │   └── neo4j_store.py      # Neo4j 数据库连接与构建图谱
│   ├── rag/                    # 大语言模型模块
│   │   └── ollama_client.py    # 对接 Ollama 提供 Embedding 与 RAG 生成
│   └── cli.py                  # 命令行工具，用于一键构建索引/图谱/测试
├── docker-compose.yml          # Docker 容器编排文件
├── requirements.txt            # Python 依赖清单
└── README.md                   # 项目文档（本文件）
```

## 8. Run Instructions (运行指南)

### 步骤 1: 启动基础设施 (Docker)
确保系统已安装 Docker。在项目根目录下运行：
```bash
docker-compose up -d
```
等待容器启动后，拉取用于 RAG 和向量化的本地大语言模型（由于网络原因，建议使用轻量级模型）：
```bash
docker-compose exec ollama ollama pull qwen2.5:0.5b
docker-compose exec ollama ollama pull nomic-embed-text
```

### 步骤 2: 安装 Python 依赖
建议使用虚拟环境 (Conda / venv)：
```bash
pip install -r requirements.txt
```

### 步骤 3: 构建索引与知识图谱
解析 `data/arxiv_papers.json` 中的学术文献，并将其灌入传统索引、向量库和图数据库中：
```bash
# 构建 Qdrant 向量索引
python -m hybrid_search.cli vector-build data/arxiv_papers.json

# 构建 Neo4j 知识图谱
python -m hybrid_search.cli neo4j-load data/arxiv_papers.json
```

### 步骤 4: 启动前端可视化界面
运行 Streamlit 服务，启动学术搜索引擎的 Web 页面：
```bash
streamlit run src/hybrid_search/app/streamlit_app.py
```
此时可在浏览器中打开 `http://localhost:8501` 体验搜索和 RAG 问答功能。

### 步骤 5: 知识图谱可视化展示
在浏览器中打开 Neo4j Browser：
* **URL**: `http://localhost:7474`
* **Username**: `neo4j`
* **Password**: `neo4j_password`
登录后即可输入 Cypher 语句可视化探索学术图谱节点。
