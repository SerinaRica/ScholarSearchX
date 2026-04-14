# ScholarSearchX：基于混合检索与知识图谱增强的学术文献搜索系统

Hybrid & KG-Enhanced Academic Search Engine

## Introduction（Background）

学术检索系统的核心目标是：在海量文献中快速、准确地找到与用户问题相关的论文，并以可解释的方式呈现证据。传统 IR（倒排索引、TF‑IDF/BM25）擅长关键词匹配与高效召回，但对语义、概念层面的关联理解不足；而知识图谱（KG）能提供结构化关系与可解释扩展；RAG（Retrieval‑Augmented Generation）则能在检索到的证据基础上生成结构化回答与引用，提升“可用性”和“交互体验”。

本项目面向 IR 课程期末/中期展示，构建一个从零实现的混合检索系统，并在课堂内容之上集成 Neo4j（KG）+ Qdrant（向量库）+ Ollama（本地模型）实现 KG‑RAG。

当前数据规模：
- 已抓取并处理 ArXiv 元数据 500 篇（Title/Abstract/Authors/Categories/URL）
- 已构建倒排索引（支持可变字节压缩）
- 已构建 Qdrant 向量库（500 篇）
- 已导入 Neo4j KG（500 篇）

## Related Work（相关研究）

- 经典 IR：倒排索引、向量空间模型（VSM）、TF‑IDF、布尔检索、纠错与 tolerant retrieval。
- 查询扩展：Rocchio 伪相关反馈（Pseudo‑Relevance Feedback）。
- 知识图谱增强检索：利用实体/概念关系进行 query expansion 与可解释召回。
- RAG/Graph‑RAG/KG‑RAG：使用检索证据（文本/图结构）支撑生成式回答，并显式给出引用与可追溯证据链。

## Methods（与课程周次结合的技术路线）

### 课程内核心内容（对应 Week）

- Week 3：文本预处理（Tokenization/Stopwords/Stemming）
- Week 5/7：倒排索引构建与压缩（gap + Variable‑Byte）
- Week 2/4：布尔检索 + 宽容检索（编辑距离纠错）
- Week 8/9：TF‑IDF + 余弦相似度排序（Top‑K）
- Week 10：评测指标（MAP / NDCG@10）与对比实验
- Week 11：伪相关反馈（Rocchio）查询扩展

### 超越课堂内容（KG + RAG 集成）

- KG（图数据库）：Neo4j 存储 `Paper/Author/Category/Term` 与关系，并支持图查询（Term→Paper→Term 多跳扩展与可视化展示）。
- Dense Retrieval（向量库）：Qdrant 存储论文向量，实现语义相似检索。
- RAG：Ollama 提供 embedding 与生成能力，在“检索证据”基础上输出带引用的回答。
- Hybrid Fusion：将 sparse（TF‑IDF）、dense（向量检索）、KG（Neo4j/扩展）结果融合，形成可解释的最终候选集合。

## Current Features（当前已实现功能）

### 1) 检索与对比页面（Streamlit）

- Base：TF‑IDF 检索
- Enhanced：Rocchio + KG 扩展 + TF‑IDF
- 对比页：左右对比 Base vs Enhanced，并展示 Added terms 与 Top‑10 变化
- RAG 问答页：一键构建向量库、一键导入 Neo4j、直接问答并展示 sources 与 debug

启动：

```bash
source .venv/bin/activate
streamlit run src/hybrid_search/app/streamlit_app.py -- --index-dir data/index
```

### 2) KG 可视化展示（Neo4j Browser，中期/答辩强烈推荐）

Neo4j 自带 Browser，可把 Cypher 查询结果以图形式可视化，适合展示“KG 如何参与扩展/召回”。

- 打开： http://localhost:7474
- 连接：`bolt://localhost:7687`
- 默认账号密码：`neo4j / neo4j_password`

常用 Cypher（可直接复制粘贴）：

验证数据是否导入：

```cypher
MATCH (p:Paper) RETURN count(p) AS papers;
```

```cypher
MATCH (t:Term) RETURN count(t) AS terms;
```

```cypher
MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC;
```

抽样查看论文节点属性：

```cypher
MATCH (p:Paper)
RETURN p.doc_id AS doc_id, p.title AS title, p.url AS url
LIMIT 10;
```

展示 Paper–Term 结构（注意词项经过预处理/词干化，可能是 hallucin/knowledg 这类）：

```cypher
MATCH (t:Term {term: "hallucin"})<-[:MENTIONS]-(p:Paper)
RETURN p, t
LIMIT 30;
```

多跳扩展（Term → Paper → Term），用于解释 KG 扩展为什么能带来更高召回：

```cypher
MATCH (seed:Term {term:"knowledg"})<-[:MENTIONS]-(p:Paper)-[:MENTIONS]->(t:Term)
RETURN seed, p, t
LIMIT 80;
```

作者/类别统计（展示数据结构与分布）：

```cypher
MATCH (p:Paper)-[:HAS_AUTHOR]->(a:Author)
RETURN a.name AS author, count(p) AS papers
ORDER BY papers DESC
LIMIT 10;
```

```cypher
MATCH (p:Paper)-[:IN_CATEGORY]->(c:Category)
RETURN c.name AS category, count(p) AS papers
ORDER BY papers DESC
LIMIT 10;
```

## Future Work（后续提升方向）

- 人工评测集与评测任务：
  - 构建 20+ 复杂科研查询（qrels 标注 0/1/2）
  - 对比 Base vs +Expansion vs +KG‑RAG（MAP/NDCG@10）
- 数据扩展：
  - 当前 500 篇 ArXiv 元数据
  - 计划扩展到近五年相关领域的开源文献（更大规模、更稳定评测）
- 更强的混合融合策略：
  - BM25 代替/补充 TF‑IDF
  - RRF/学习融合权重（sparse/dense/kg）
- 更“像论文”的 KG‑RAG：
  - Neo4j 多跳路径检索（结构约束/路径评分）
  - 在 UI 中可视化展示命中的子图与证据链
- Reranking（可选）：
  - 对候选 Top‑N 使用轻量 reranker（本地或 API），提升最终 Top‑k 的语义精度

## Code Guide（每个模块做什么）

数据与预处理：
- [arxiv_downloader.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/datasets/arxiv_downloader.py)：抓取 ArXiv 元数据输出 `data/corpus.jsonl`
- [text.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/preprocess/text.py)：文本预处理（tokenize/stopwords/stem）与 NLTK 资源下载入口

索引与传统检索：
- [inverted_index.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/index/inverted_index.py)：倒排索引构建/保存/加载，doc_norms，文档 top terms
- [compress_vbyte.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/index/compress_vbyte.py)：可变字节编码（gap + tf）
- [boolean_query.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/search/boolean_query.py)：布尔检索解析与执行
- [tfidf_ranker.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/search/tfidf_ranker.py)：TF‑IDF + 余弦相似度排序
- [spell.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/search/spell.py)：编辑距离纠错建议

查询扩展与 KG：
- [query_expand.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/search/query_expand.py)：Rocchio + KG 扩展（词典 + 轻量图谱邻居）
- [graph.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/kg/graph.py)：从语料构建轻量“词项共现图” `data/kg/graph.pkl`
- [neo4j_store.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/kg/neo4j_store.py)：Neo4j schema/写入/基于词项检索
- [build_neo4j.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/kg/build_neo4j.py)：把 `data/corpus.jsonl` 导入 Neo4j（Paper/Author/Category/Term）

向量库与 RAG：
- [qdrant_store.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/vector/qdrant_store.py)：Qdrant collection/upsert/query
- [ollama_client.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/rag/ollama_client.py)：Ollama embeddings/generate HTTP 客户端
- [hybrid.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/rag/hybrid.py)：Hybrid 检索融合 + 生成式回答（带 sources）

评测：
- [metrics.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/eval/metrics.py)：AP/MAP/NDCG
- [evaluate.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/eval/evaluate.py)：评测入口（含 per_query 诊断信息）

系统入口：
- [cli.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/cli.py)：所有命令行任务入口（download/build-index/build-kg/vector-build/neo4j-load/rag-ask/search/evaluate/label）
- [streamlit_app.py](file:///Volumes/新加卷/课程/ir/project/src/hybrid_search/app/streamlit_app.py)：中期展示 UI（检索/对比/RAG 问答）

## How to Run（如何启动任务）

### 1) Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m hybrid_search.cli nltk-download
```

如果需要 KG+RAG（Neo4j/Qdrant/Ollama）：

```bash
python -m pip install -e ".[dev,kg_rag]"
```

### 2) 获取数据与构建索引（传统 IR）

抓取 ArXiv：

```bash
hybrid-search download --query '("knowledge graph" OR "kg-rag" OR "neuro-symbolic" OR "reasoning") AND cat:cs.AI' --max-results 500 --out data/corpus.jsonl
```

构建索引（可选压缩 vbyte）：

```bash
hybrid-search build-index --corpus data/corpus.jsonl --index-dir data/index --compress vbyte
```

可选：构建轻量共现 KG（用于扩展）

```bash
python -m hybrid_search.cli build-kg --index-dir data/index --out data/kg/graph.pkl
```

### 3) 启动 Docker（Neo4j + Qdrant + Ollama）

```bash
docker compose up -d
```

Ollama 拉模型（建议中期先用小模型，成功率更高）：

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen2.5:0.5b
```

### 4) 构建向量库（Qdrant）与导入 Neo4j KG

构建向量库（可用 --limit 先小规模验证）：

```bash
python -m hybrid_search.cli vector-build --index-dir data/index --collection arxiv_papers --batch 16 --limit 500
```

导入 Neo4j KG：

```bash
python -m hybrid_search.cli neo4j-load --corpus data/corpus.jsonl
```

### 5) 运行 RAG

```bash
python -m hybrid_search.cli rag-ask --index-dir data/index --question "What is KG-RAG and how does it mitigate hallucination?" --chat-model qwen2.5:0.5b --kg-expand --kg-neo4j
```

### 6) 评测（MAP / NDCG@10）

准备：
- `data/eval/queries.json`
- `data/eval/qrels.tsv`

辅助标注（输出候选 doc_id，方便做 qrels 池化标注）：

```bash
python -m hybrid_search.cli label --index-dir data/index --queries data/eval/queries.json --topk 30 --expand rocchio,kg
```

运行评测：

```bash
python -m hybrid_search.cli evaluate --index-dir data/index --queries data/eval/queries.json --qrels data/eval/qrels.tsv
```

## License

MIT License
