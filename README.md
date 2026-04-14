# 基于混合检索与知识图谱增强的学术文献搜索系统

Hybrid & KG-Enhanced Academic Search Engine

本项目面向 IR 课程期末大作业，覆盖传统信息检索核心（倒排索引、布尔检索、向量空间模型、TF-IDF、评测指标），并提供拔高路线（伪相关反馈 Rocchio、知识图谱增强扩展、可插拔重排接口、对比评测 + Streamlit 展示）。

## 1. 技术路线（可直接写进报告）

系统分为四条主线，对应课程大纲的“数据与索引构建 → 检索与打分 → 查询扩展与重排 → 系统评测与展示”。

### 1.1 数据底座与索引构建（Week 3/5/7）

- 数据集：抓取近三年 ArXiv 元数据（Title/Abstract/Authors/Categories），建议主题集中在 LLM Reasoning / Knowledge Graph / Neuro-Symbolic。
- 预处理：对 Abstract 做分词、大小写归一、停用词过滤、词干化（默认 NLTK）。
- 倒排索引：手写 `{term -> postings}`，postings 存 `[(doc_id, tf), ...]`。
- 压缩（加分项）：对 postings 的 doc_id 做 gap，再用可变字节编码（Variable-Byte）压缩；在报告中展示压缩前后体积差异。

产物：
- `data/corpus.jsonl`：文档库（doc_id + 元数据）
- `data/index/index.pkl`：索引与统计信息（含可选压缩 postings）

### 1.2 基础检索与打分（Week 2/4/8/9）

- 布尔检索：支持 `AND/OR/NOT` 与括号，返回命中文档集合。
- 宽容检索：对 OOV 词给出基于编辑距离的纠错建议。
- VSM + TF-IDF：基于倒排表动态计算 TF-IDF 权重，余弦相似度排序，输出 Top-K（默认 Top-100）。

### 1.3 高级进阶：查询扩展与重排（Week 11 + 拔高）

- 伪相关反馈（Rocchio）：取初排 Top-N 文档的高频词/高 TF-IDF 词做扩展，二次检索对比提升。
- KG 增强扩展：维护轻量概念-同义/子概念字典（`data/kg_dict.json`），命中实体时扩展到相关概念。
- 可插拔重排：提供重排接口占位（默认 no-op），可替换为本地小模型或 API 方案。

### 1.4 严谨评测与展示（Week 10）

- 测试集：人工构造查询（建议 20 个），并为语料中相关文档做 qrels 标注。
- 指标：MAP、NDCG@10，并对比三条系统线：
  - Base：TF-IDF
  - +Expansion：TF-IDF + Rocchio / KG 扩展
  - +Rerank：二阶段重排（可选）
- 展示：Streamlit 页面左右对比结果 + 指标报表（适合答辩演示）。

## 2. 代码结构

```
.
├── data/
│   ├── kg_dict.json
│   └── eval/
│       ├── queries.json
│       └── qrels.tsv
├── src/hybrid_search/
│   ├── cli.py
│   ├── datasets/arxiv_downloader.py
│   ├── preprocess/text.py
│   ├── index/inverted_index.py
│   ├── index/compress_vbyte.py
│   ├── search/boolean_query.py
│   ├── search/spell.py
│   ├── search/tfidf_ranker.py
│   ├── search/query_expand.py
│   ├── eval/metrics.py
│   ├── eval/evaluate.py
│   └── app/streamlit_app.py
└── tests/
```

## 3. 环境与安装（macOS + Python）

推荐 Python 3.11+。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

首次使用 NLTK 时需要下载资源：

```bash
python -m hybrid_search.cli nltk-download
```

### 常见问题：zsh: command not found: hybrid-search

`hybrid-search` 是通过 `pip install -e ".[dev]"` 安装到虚拟环境里的命令行入口。遇到这个错误通常是：

- 没有激活虚拟环境：`source .venv/bin/activate`
- 没有安装项目依赖：`python -m pip install -e ".[dev]"`

如果你不想依赖命令行入口，也可以用模块方式运行（等价）：

```bash
python -m hybrid_search.cli build-index --corpus data/corpus.jsonl --index-dir data/index --compress vbyte
```

## 4. 快速开始

### 4.1 抓取 ArXiv 元数据

```bash
hybrid-search download \
  --query '("knowledge graph" OR "kg-rag" OR "neuro-symbolic" OR "reasoning") AND cat:cs.AI' \
  --max-results 2000 \
  --out data/corpus.jsonl
```

### 4.2 构建索引（含可选压缩）

```bash
hybrid-search build-index \
  --corpus data/corpus.jsonl \
  --index-dir data/index \
  --compress vbyte
```

### 4.2.1 构建轻量知识图谱（可选但推荐，KG-Enhanced 的核心）

这一步会从你的语料里自动构建一个“词项共现图”（不需要手工画 KG），用于查询扩展的“邻居概念”召回。

```bash
python -m hybrid_search.cli build-kg --index-dir data/index --out data/kg/graph.pkl
```

## 4.2.2 真正集成 KG + RAG（Docker 方案）

本项目提供一个可复现的“Neo4j KG + Qdrant 向量库 + Ollama 本地模型”的集成方案：
- Neo4j：存储 Paper/Author/Category/Term 与关系（真正图数据库）
- Qdrant：存储论文摘要向量（向量库）
- Ollama：同时提供 embeddings 与回答生成（RAG 的 G）

### 启动服务

```bash
docker compose up -d
```

可选：在 Ollama 里拉模型（首次需要）

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2:3b
```

### 安装额外依赖

```bash
python -m pip install -e ".[dev,kg_rag]"
```

### 构建向量库（Qdrant）

```bash
python -m hybrid_search.cli vector-build --index-dir data/index
```

### 导入 Neo4j KG

```bash
python -m hybrid_search.cli neo4j-load --corpus data/corpus.jsonl
```

### Neo4j 可视化展示（中期/答辩强烈推荐）

Neo4j 自带 Browser，可直接把 Cypher 查询结果以图的形式可视化，非常适合中期展示“KG 如何参与检索与扩展”。

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

作者/类别统计（中期展示很直观）：

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

### 运行 RAG

```bash
python -m hybrid_search.cli rag-ask --index-dir data/index --question "What is KG-RAG and how does it mitigate hallucination?" --kg-expand --kg-neo4j
```

### 4.3 检索

布尔检索：

```bash
hybrid-search search \
  --index-dir data/index \
  --mode boolean \
  --query 'LLM AND (Reasoning OR Graph) NOT Vision' \
  --topk 20
```

TF-IDF 排序检索（默认含宽容检索建议）：

```bash
hybrid-search search \
  --index-dir data/index \
  --mode tfidf \
  --query 'kg rag hallucination mitigation' \
  --topk 20
```

启用扩展（二阶段）：

```bash
hybrid-search search \
  --index-dir data/index \
  --mode tfidf \
  --query 'kg rag' \
  --expand rocchio,kg \
  --topk 20
```

### 4.4 评测（MAP / NDCG@10）

准备：
- `data/eval/queries.json`：`[{ "qid": "...", "query": "..." }, ...]`
- `data/eval/qrels.tsv`：`qid<TAB>doc_id<TAB>relevance(0/1/2)`

标注建议（让评测“有意义”且方便写报告）：
- 每条 query 先用检索跑出候选集合（例如 Base Top-30 + Enhanced Top-30 的并集），再从中挑选相关文档做 qrels
- 至少给每条 query 标 5–15 篇（包含 0/1/2 三档更容易拉开 NDCG）

辅助标注命令（输出可直接复制 doc_id 进 qrels）：

```bash
python -m hybrid_search.cli label --index-dir data/index --queries data/eval/queries.json --topk 30 --expand rocchio,kg
```

运行：

```bash
hybrid-search evaluate \
  --index-dir data/index \
  --queries data/eval/queries.json \
  --qrels data/eval/qrels.tsv
```

### 4.5 Streamlit 展示

```bash
streamlit run src/hybrid_search/app/streamlit_app.py -- \
  --index-dir data/index
```

## 5. 报告建议（拿高分用）

- 指数构建：展示 postings 数量、词表大小、构建耗时
- 压缩：展示压缩前后索引文件大小、解压检索耗时
- 检索：展示布尔检索解析示例、纠错示例
- 评测：用同一套 qrels 对比 Base vs +Expansion vs +Rerank（至少 Base vs +Expansion）
- 可解释性：展示扩展出的词（Rocchio/ KG）与它们的贡献

## 6. 许可

课程项目用途。若使用第三方 API/模型，请自行遵守对应服务条款。
