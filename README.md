# MedAgent：口腔多 Agent 后端

MedAgent 将病历文本、病历图片和患者口语描述整理为结构化因素，调用专病 Agent 与独立规则引擎，并由 Leader 关联知识资料、整理分析结果。当前包含根管治疗、牙隐裂、牙髓与根尖周、牙周四个专病 Agent。

## 版本与迭代

`main` 保存完整的清理后代码历史。以下 tag 固定到对应版本提交，可通过 `git checkout <tag>` 查看。早期版本的启动方式不同，本文的配置与 API 示例对应当前 `main`。

| Tag | 提交 | 版本 | 主要变化 |
| --- | --- | --- | --- |
| [v0.0.0](https://github.com/auince/MedAgent/tree/v0.0.0) | `d144949` | 初始基线 | 建立 Git 时的项目代码基线。 |
| [v1.0.0](https://github.com/auince/MedAgent/tree/v1.0.0) | `4aca917` | 根管原型 | 病历文本 → 领域模型因素提取 → 根管治疗难度评估。 |
| [v2.0.0](https://github.com/auince/MedAgent/tree/v2.0.0) | `98f2729` | 牙隐裂与手动路由 | 新增牙隐裂；医生显式选择专病任务。 |
| [v3.0.0](https://github.com/auince/MedAgent/tree/v3.0.0) | `5cee917` | AgentHarness | Leader 整理输入、调用两个专病 Agent、整理输出，并记录执行过程。 |
| [v4.0.0](https://github.com/auince/MedAgent/tree/v4.0.0) | `3dbf266` | 独立规则引擎 | 模型提取患者级、患牙级因素；规则引擎执行根管难度合成及牙隐裂计分。 |
| [v5.0.0](https://github.com/auince/MedAgent/tree/v5.0.0) | `11c3c85` | 并行审校与工具 | 图像裁剪、放大、OCR；审校与专病分析并行，文本修正后重新分析。 |
| [v6.0.0](https://github.com/auince/MedAgent/tree/v6.0.0) | `222611c` | LangGraph | Leader 主图、专病子图、结构校验、有限纠错、版本检查及条件回流。 |
| [v7.0.0](https://github.com/auince/MedAgent/tree/v7.0.0) | `0fb8453` | 患者对话与记忆 | 口语症状整理、追问、辅助预诊、短期会话和需授权保存的长期记忆。 |
| [v8.0.0](https://github.com/auince/MedAgent/tree/v8.0.0) | `17b38de` | 四个专病 Agent | 新增牙髓与根尖周、牙周，统一专病注册表。 |
| [v8.1.0](https://github.com/auince/MedAgent/tree/v8.1.0) | `64e79cf` | 后端服务 | FastAPI、异步任务、会话及记忆 API，统一配置管理。 |
| [v9.0.0](https://github.com/auince/MedAgent/tree/v9.0.0) | `d064035` | 混合 RAG | BM25 + 向量召回、加权 RRF 融合、模型重排序和版本化索引 API。 |
| [v9.1.0](https://github.com/auince/MedAgent/tree/v9.1.0) | `d6799f6` | Agent 职责调整 | Leader 统一检索与汇总；专病 Agent 提取因素并生成关联证据的风险说明。 |
| [v9.2.0](https://github.com/auince/MedAgent/tree/v9.2.0) | `57217ea` | 参考规则 | 加入牙髓与根尖周 AAE、牙周 AAP/EFP 参考规则与证据约束。 |
| [v9.3.0](https://github.com/auince/MedAgent/tree/v9.3.0) | `d50c72e` | 研究仓库拆分记录 | 记录数据生成、训练、评测独立仓库的拆分；此提交未改变发布的应用代码。 |
| [v9.3.1](https://github.com/auince/MedAgent/tree/v9.3.1) | 本 README 提交 | 使用说明 | 新增配置、API 与调用示例；应用代码不变。 |

清理发布历史后，提交哈希与原开发仓库不同，迭代顺序保留。`v9.x.x` 用于标记第九版之后的增量；它们是源码发布 tag，配置中的应用版本当前仍为 `9`。

## 当前功能与职责

- **LeaderAgent**：引导患者对话与追问；审校病历图片和初步文本；调度专病 Agent；基于专病结果调用 RAG，整理证据、相似病症文档与最终说明。
- **专病 Agent**：提取结构化症状、患者级和患牙级因素，保留否定、未知、冲突与原文证据；解释独立规则引擎给出的风险等级及其依据。
- **规则引擎**：执行根管治疗难度合成；牙隐裂加分、扣分、模块封顶和等级映射；牙髓与根尖周、牙周参考规则。
- **LangGraph**：编排解析、路由、并行审校与专病分析、输入版本检查、条件回流和汇总。修正输入后重新分析，避免沿用旧文本结果。
- **RAG**：BM25 与向量召回、加权 RRF 融合、可选模型重排序；保留检索阶段、来源片段、文本版本和降级信息。
- **后端服务**：HTTP API、异步任务、会话、显式授权的长期记忆、索引管理及执行记录。启动后端不会启动模型服务或训练任务。

四种评估输出采用不同含义：根管为治疗难度；牙隐裂为项目规则的支持等级；牙髓与根尖周为就诊紧急程度及候选方向；牙周区分严重程度 Stage 与进展 Grade。相似文档匹配、原文定位和规则输出不等于临床确诊。信息不足时保留未知与待补充项。

## 目录

```text
medagent/           Agent、规则、RAG、HTTP API 与会话实现
schemas/            业务因素词表和评分规则（代码依赖）
scripts/server.py   启动后端
scripts/run.py      直接执行一次病历评估
scripts/chat.py     会话与记忆命令行入口
scripts/build_dense_index.py  构建向量索引
web/index.html      原型界面
```

仓库不包含实际部署配置、病例数据、模型权重、教材/知识库文件、原始提示词资料及 skill 文档。README 中的病历是人工编写的接口演示，不是患者数据或实测结果。

## 安装与本地准备

需要 Python **3.10+**。会话与任务锁使用 SQLite 和 `fcntl`，后端适合运行在 Linux 或 macOS；Windows 可使用 WSL。

```bash
git clone https://github.com/auince/MedAgent.git
cd MedAgent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
mkdir -p configs knowledge prompts skills/information_review
```

使用本地文件 `configs/service.json` 统一管理地址、模型标识、路径和运行参数。相对路径以**配置文件所在目录**为基准，而非当前工作目录。示例占位端点 `*.example.invalid` 不可连接，启动评估前应替换成已部署的模型接口。

本地还需准备：

| 资源 | 用途 |
| --- | --- |
| `configs/specialties.json` | 专病注册表，示例如下 |
| `skills/information_review/SKILL.md` | 病历审校策略；评估及独立审校入口读取此文件，即使请求仅有文本也需要准备 |
| `prompts/crack_legacy.txt` | 牙隐裂因素提取格式契约；应与 `schemas/crack_rules.json` 和因素校验器一致 |
| `prompts/root_canal_legacy.txt` | 选择 `explicit_fields` 或 `training_aligned` 根管提取策略时需要；默认 `original` 根管策略使用代码内模板与词表 |
| `knowledge/<专业ID>/*.md` 或 `*.txt` | 有使用权限的知识材料；按四个专业 ID 分目录，文档可用 `Source: https://...` 标记来源 |

这些资源沿用已有本地资源，或按源码契约自行准备；克隆仓库后不会自动下载它们。缺少 skill、提示词或模型服务时，业务调用可能失败；知识库为空时不会产生可引用的病症文档。

### 配置示例

将以下**示例模板**保存到本地 `configs/service.json`，替换所有 `<...>` 模型标识和模型服务地址。密钥只通过对应环境变量提供，不能填入文件。

```json
{
  "version": 9,
  "service_name": "MedAgent",
  "http": {
    "host": "127.0.0.1",
    "port": 8000,
    "api_prefix": "/api/v1",
    "docs_path": "/docs",
    "openapi_path": "/openapi.json",
    "public_base_url": null,
    "cors_origins": [],
    "api_key_env": "MEDAGENT_API_TOKEN",
    "allow_unauthenticated_loopback": false,
    "max_request_bytes": 20000000,
    "workers": 2,
    "max_pending_runs": 8
  },
  "paths": {
    "knowledge_dir": "../knowledge",
    "schemas_dir": "../schemas",
    "prompts_dir": "../prompts",
    "skills_dir": "../skills",
    "specialties_file": "specialties.json",
    "runtime_dir": "../runtime",
    "jobs_dir": "../runtime/jobs",
    "assessment_runs_dir": "../runtime/assessments",
    "graph_dir": "../runtime/graphs",
    "conversation_runs_dir": "../runtime/conversations",
    "sessions_db": "../runtime/sessions.sqlite3",
    "dense_index": "../runtime/knowledge/dense.json",
    "index_runs_dir": "../runtime/index-runs",
    "review_runs_dir": "../runtime/reviews"
  },
  "model_services": {
    "dental": {
      "base_url": "https://dental.example.invalid",
      "chat_path": "/v1/chat/completions",
      "api_key_env": "DENTAL_API_KEY"
    },
    "general": {
      "base_url": "https://general.example.invalid",
      "chat_path": "/v1/chat/completions",
      "api_key_env": "GENERAL_API_KEY"
    },
    "embedding": {
      "base_url": "https://embedding.example.invalid",
      "api_key_env": "EMBEDDING_API_KEY"
    },
    "reranker": {
      "base_url": "https://reranker.example.invalid",
      "api_key_env": "RERANKER_API_KEY"
    }
  },
  "models": {
    "root_canal": {
      "service": "dental",
      "model": "<SPECIALTY_MODEL_ID>",
      "max_tokens": 4096,
      "timeout": 180,
      "retries": 1
    },
    "crack": {
      "service": "dental",
      "model": "<SPECIALTY_MODEL_ID>",
      "max_tokens": 4096,
      "timeout": 180,
      "retries": 1
    },
    "pulp_periapical": {
      "service": "dental",
      "model": "<SPECIALTY_MODEL_ID>",
      "max_tokens": 4096,
      "timeout": 180,
      "retries": 1
    },
    "periodontal": {
      "service": "dental",
      "model": "<SPECIALTY_MODEL_ID>",
      "max_tokens": 4096,
      "timeout": 180,
      "retries": 1
    },
    "leader": {
      "service": "general",
      "model": "<GENERAL_MODEL_ID>",
      "max_tokens": 4096,
      "timeout": 180,
      "retries": 1
    },
    "vision": {
      "service": "general",
      "model": "<VISION_MODEL_ID>",
      "max_tokens": 4096,
      "timeout": 180,
      "retries": 1
    },
    "review": {
      "service": "general",
      "model": "<GENERAL_MODEL_ID>",
      "max_tokens": 4096,
      "timeout": 180,
      "retries": 1
    },
    "general": {
      "service": "general",
      "model": "<GENERAL_MODEL_ID>",
      "max_tokens": 4096,
      "timeout": 180,
      "retries": 1
    }
  },
  "agent": {
    "review_policy": "original",
    "extraction_policy": "original",
    "graph_version": "service-v9-reference-rule-analysis"
  },
  "consultation": {
    "version": "consultation-v9",
    "max_actions": 12,
    "max_tools": 8,
    "max_questions": 4,
    "max_experts": 2,
    "max_experts_independent": 4,
    "models": {
      "frontend": "general",
      "leader": "general",
      "needs": "general",
      "answer": "general",
      "specialist": "general",
      "vision": "vision",
      "pulp_periapical": "pulp_periapical",
      "periodontal": "periodontal",
      "root_canal": "root_canal",
      "crack": "crack",
      "review": "review"
    }
  },
  "limits": {
    "record_chars": 30000,
    "chat_chars": 12000,
    "images": 4,
    "search_query_chars": 1024,
    "search_limit": 10
  },
  "retrieval": {
    "mode": "bm25",
    "top_k": 4,
    "candidate_k": 20,
    "chunk_size": 800,
    "chunk_stride": 600,
    "failure_policy": "bm25_with_notice",
    "index_policy": "require_current",
    "max_index_bytes": 100000000,
    "fusion": {
      "rrf_k": 60,
      "bm25_weight": 1.0,
      "dense_weight": 1.0
    },
    "embedding": {
      "service": "embedding",
      "request_path": "/v1/embeddings",
      "model": "<EMBEDDING_MODEL_ID>",
      "revision": "<MODEL_REVISION>",
      "query_prefix": "",
      "document_prefix": "",
      "dimensions": null,
      "batch_size": 8,
      "max_input_chars": 4096,
      "timeout": 60,
      "retries": 1,
      "retry_delay_seconds": 1,
      "max_response_bytes": 4000000
    },
    "reranker": {
      "enabled": false,
      "service": "reranker",
      "request_path": "/rerank",
      "protocol": "tei",
      "model": "<RERANK_MODEL_ID>",
      "revision": "<MODEL_REVISION>",
      "max_query_chars": 1024,
      "max_document_chars": 4096,
      "timeout": 60,
      "retries": 1,
      "retry_delay_seconds": 1,
      "max_response_bytes": 1000000
    }
  },
  "references": {
    "urgent_referral": "https://www.nhs.uk/conditions/dental-abscess/"
  }
}
```

将以下注册表示例保存到本地 `configs/specialties.json`。模型 ID 可按专业分别配置，不要求四个专业共用模型。

```json
{
  "version": "specialties-v9",
  "agents": [
    {
      "id": "root_canal",
      "name": "根管治疗",
      "scope": "已提交病历中的根管治疗难度因素",
      "adapter": "root_canal",
      "tools": [
        "rule_assess"
      ],
      "enabled": true,
      "role": "structured_symptom_and_risk_analysis",
      "leader_tools": [
        "knowledge_search"
      ],
      "legacy_task": "root_canal"
    },
    {
      "id": "crack",
      "name": "牙隐裂",
      "scope": "已提交病历中的牙隐裂相关因素",
      "adapter": "crack",
      "tools": [
        "rule_assess"
      ],
      "enabled": true,
      "role": "structured_symptom_and_risk_analysis",
      "leader_tools": [
        "knowledge_search"
      ],
      "legacy_task": "crack"
    },
    {
      "id": "pulp_periapical",
      "name": "牙髓与根尖周",
      "scope": "牙髓与根尖周症状及有依据的规则因素",
      "adapter": "pulp_periapical",
      "tools": [
        "rule_assess"
      ],
      "enabled": true,
      "role": "structured_symptom_and_risk_analysis",
      "leader_tools": [
        "knowledge_search"
      ]
    },
    {
      "id": "periodontal",
      "name": "牙周",
      "scope": "牙周症状及有依据的规则因素",
      "adapter": "periodontal",
      "tools": [
        "rule_assess"
      ],
      "enabled": true,
      "role": "structured_symptom_and_risk_analysis",
      "leader_tools": [
        "knowledge_search"
      ]
    }
  ]
}
```

### 启动

```bash
# 创建仅在当前环境使用的服务访问令牌；不要提交或打印令牌。
export MEDAGENT_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
# 如模型服务需要认证，通过 DENTAL_API_KEY、GENERAL_API_KEY 等环境变量设置。
python scripts/server.py --config configs/service.json
```

也可设置 `MEDAGENT_CONFIG` 指向配置文件。API 前缀、监听端口、文档路径均可配置。默认示例为 `http://localhost:8000/docs` 和 `/api/v1`。服务使用单个 Uvicorn 进程；`http.workers` 控制进程内任务线程数。多个服务进程必须使用独立的 runtime 目录。

## API 概览

下表路径相对于配置的 `http.api_prefix`。业务 API 使用 `Authorization: Bearer <令牌>`；具体请求约束可查看服务生成的 OpenAPI。

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| GET | `/health`、`/capabilities` | 健康状态、功能和专病列表 |
| POST | `/assessments` | 提交病历评估，支持医生手动路由或自动路由 |
| POST | `/records/review` | 单独审校病历图片与文本 |
| GET | `/runs/{run_id}` | 查询异步任务状态及结果 |
| POST / GET | `/sessions` / `/sessions/{conversation_id}` | 创建、查看会话 |
| POST | `/sessions/{conversation_id}/messages` | 发送患者口语描述或病历材料 |
| GET / POST / DELETE | `/sessions/{conversation_id}/memory` | 查询、授权保存、删除长期记忆 |
| POST | `/knowledge/search` | 检索知识库，返回片段和检索过程 |
| GET / POST | `/knowledge/status` / `/knowledge/index` | 查询索引状态、提交索引构建任务 |
| GET | `/graphs` | 获取 Leader、专病及对话图的 Mermaid 描述 |

评估、审校、会话消息和索引构建返回 HTTP `202`。拿到 `run_id` 后通过 `/runs/{run_id}` 查询结果。提交成功不表示分析完成；任务 `status` 与 `result.status` 分别表示执行状态和业务结果，例如任务完成后业务仍可能是 `needs_review`。

## 调用示例

以下 `curl` 示例需要 Python 3，用于从响应中提取 ID。访问令牌须与启动后端时一致。

```bash
export API_BASE="http://localhost:8000/api/v1"
curl --fail-with-body -sS "$API_BASE/health" \
  -H "Authorization: Bearer $MEDAGENT_API_TOKEN"
```

健康接口中 `model_readiness: "not_probed"` 只说明后端没有探测模型，不能据此判断模型可用。

### 1. 医生手动选择根管治疗评估

```bash
RUN_ID=$(curl --fail-with-body -sS "$API_BASE/assessments" \
  -H "Authorization: Bearer $MEDAGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: demo-root-001" \
  -d '{"text":"36牙拟行首次根管治疗。患者张口受限。尚未提供根管影像资料。","tasks":["root_canal"],"routing_mode":"manual","instruction":"整理已有因素并评估治疗难度；未记载因素保持未知。"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')

curl --fail-with-body -sS "$API_BASE/runs/$RUN_ID" \
  -H "Authorization: Bearer $MEDAGENT_API_TOKEN"
```

任务未结束时可再次查询同一 ID。完成后主要关注 `result.specialty_results`（提取与风险说明）、`result.results`（Leader 分析）、`result.closest_disease_documents`（检索文档）、`result.review`（审校）和 `result.graph_trace`（执行记录）。资料不完整时不应预期一个确定的难度等级。

牙隐裂可以将 `tasks` 改为 `["crack"]`，例如提交“36牙咬硬物时出现局部疼痛，松口时明显；检查资料尚未提供”。`tasks` 支持四个专业 ID 的非重复列表。

### 2. 自动路由到合适的专病 Agent

```bash
curl --fail-with-body -sS "$API_BASE/assessments" \
  -H "Authorization: Bearer $MEDAGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"患者描述左下后牙冷热刺激后疼痛，同时刷牙时牙龈出血；尚无检查报告。","routing_mode":"auto","instruction":"整理症状、判断分析需求并保留缺失信息。"}'
```

自动路由时不传 `tasks`；医生手动路由时必须传非空 `tasks`。不能同时指定自动路由和专病列表。

### 3. 患者口语对话与追问

```bash
CONVERSATION_ID=$(curl --fail-with-body -sS "$API_BASE/sessions" \
  -H "Authorization: Bearer $MEDAGENT_API_TOKEN" \
  -H "Content-Type: application/json" -d '{}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["conversation_id"])')

curl --fail-with-body -sS "$API_BASE/sessions/$CONVERSATION_ID/messages" \
  -H "Authorization: Bearer $MEDAGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"我下面左边的大牙最近疼，喝凉水更明显，但我说不清是哪一颗。"}'
```

该接口也返回 `run_id`。查询该任务，阅读追问并在**当前消息任务结束后**向同一会话发送下一条回答。模型应保留含糊牙位，不应自行猜测 FDI 编号。长期记忆不会自动保存：用 `GET /sessions/{id}/memory` 查询，用户确认后用 `POST` 提交实际返回的 `fact_ids`，用 `DELETE` 删除。

### 4. 病历图片与初步文本审校

图片只接受 PNG/JPEG/WebP 的 base64 data URL，不接受本地文件路径或任意外部图片 URL。以下 Python 示例读取自己本地的 `record.png` 并提交；图片不会写入 Git 仓库。

```python
import base64
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

image = base64.b64encode(Path("record.png").read_bytes()).decode()
payload = {
    "text": "初步识别文本：36牙叩痛（+）。",
    "images": ["data:image/png;base64," + image],
}
request = Request(
    os.environ["API_BASE"] + "/records/review",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["MEDAGENT_API_TOKEN"],
    },
    method="POST",
)
with urlopen(request, timeout=30) as response:
    accepted = json.load(response)
print(accepted["run_id"])  # 再通过 /runs/{run_id} 查询审校结果
```

输出中 `changed` 表示文本是否修正，`review.applied_corrections` 记录修正，`review.tools` 记录裁剪、放大或 OCR 调用。无法辨认或缺乏 OCR 支持的内容不能被自动补写。评估入口 `/assessments` 同样支持 `text` 与 `images`，用于并行审校和专病分析。

### 5. 混合检索与重排序

默认模板使用 BM25，便于先接入本地材料。启用第九版完整检索路径时，将配置中的 `retrieval.mode` 改为 `"hybrid"`、`retrieval.reranker.enabled` 改为 `true`，并填写真实 embedding/reranker 地址、模型标识及 revision。

- Embedding：兼容 `/v1/embeddings` 的向量接口。
- Reranker：支持 `tei`（`query`、`texts` → 含 `index`、`score` 的数组）或 `cohere`（`query`、`documents` → 含 `index`、`relevance_score` 的 `results`）；切换协议时同步修改 `request_path`。
- `require_current` 要求已有且匹配的索引；先调用构建接口。`build_if_missing` 可在需要时构建。语料和模型身份改变后应重新构建索引。
- `bm25_with_notice` 在混合检索失败时返回显式降级标记；`fail` 则使该检索失败。模型 revision 是配置声明，不能视为独立校验了模型权重。

```bash
# 构建任务返回 run_id；查询 /runs/{run_id} 确认完成后再检索。
curl --fail-with-body -sS "$API_BASE/knowledge/index" \
  -H "Authorization: Bearer $MEDAGENT_API_TOKEN" \
  -H "Content-Type: application/json" -d '{"force":false}'

curl --fail-with-body -sS "$API_BASE/knowledge/search" \
  -H "Authorization: Bearer $MEDAGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"牙龈出血和牙齿松动的评估依据","task":"periodontal","limit":3}'
```

返回 `results` 和 `retrieval`。完整混合路径的 `retrieval.stages` 包含 `bm25`、`dense`、`weighted_rrf`、`reranker`；检查 `degraded`、`reranked` 和阶段状态，不能仅凭配置判断重排序已经执行。检索分数不等于疾病概率。

### 返回结构示例

以下只展示异步任务**受理**的形状，ID 与时间均为演示值，未声称模型已完成分析：

```json
{
  "run_id": "00000000000000000000000000000001",
  "kind": "assessment",
  "status": "queued",
  "created_at": 0
}
```

实际 `run_id` 是服务生成的 32 位十六进制 ID。查询任务可能得到 `queued`、`running`、`completed`、`failed` 或服务中断后的 `interrupted`；业务结果仍需结合 `issues`、未知项和证据阅读。相同幂等键与相同请求可复用任务，改请求体时应更换幂等键。

## 发布与本地资源

仅发布应用代码和业务规则。`.gitignore` 排除配置、数据、病例、文档资料、模型权重、运行输出、密钥及环境文件，仅为本 README 设置例外。请勿强制提交本地资源或把凭据放入源码/README。

本 README 根据源码接口编写；其中模型、图像与检索请求没有在本次文档更新中执行，不代表已完成模型服务或临床验证。
