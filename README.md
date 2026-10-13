# MedAgent

MedAgent 是一个口腔多 Agent 后端，支持病历分析、图片审校和患者症状对话。当前包含根管治疗、牙隐裂、牙髓与根尖周、牙周四个专病 Agent。

## 功能

- **症状对话**：Leader 整理患者的口语描述，通过追问补充信息，并管理会话与用户授权保存的长期记忆。
- **病历审校**：结合病历图片和初步文本，调用裁剪、放大、OCR 工具校对；修正文本后重新进行专病分析。
- **专病评估**：专病 Agent 提取患者级、患牙级因素，关联原文证据并生成风险说明；独立规则引擎负责评分和等级判定。
- **知识检索**：Leader 根据专病结果检索知识库，支持 BM25、向量召回、RRF 融合与重排序，返回相似病症文档和分析依据。

流程由 LangGraph 主图与专病子图编排，主要功能通过 FastAPI 提供。评估结果用于辅助分析，信息不足时保留未知项，不替代临床诊断。

## 版本

| 版本 | 主要变化 |
| --- | --- |
| V1 | 根管治疗难度评估原型 |
| V2 | 牙隐裂评估与医生手动路由 |
| V3 | AgentHarness、Leader 与两个专病 Agent |
| V4 | 因素提取与规则评分分离 |
| V5 | 图像工具、并行审校与修正后重评 |
| V6 | LangGraph 主图与专病子图 |
| V7 | 患者对话、症状追问与记忆 |
| V8 | 新增牙髓与根尖周、牙周 Agent；后续加入后端 API 与统一配置 |
| V9 | 混合检索与重排序；后续完善 Agent 职责、风险说明和参考规则 |

各版代码可在 [Tags](https://github.com/auince/MedAgent/tags) 中查看，例如 `git checkout v8.0.0`。以下使用方式对应当前 `main`。

## 启动

需要 Python 3.10+，运行环境为 Linux 或 macOS。

```bash
git clone https://github.com/auince/MedAgent.git
cd MedAgent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/server.py --config configs/service.json
```

启动前需准备本地的 `configs/service.json`、`configs/specialties.json`、提示词、审校 skill 和知识资料，并部署模型服务。地址、模型标识和资源路径统一由配置文件管理，凭据通过环境变量传入。仓库不包含本地配置、病例数据、模型权重及知识资料。

服务访问令牌由配置中的 `http.api_key_env` 指定；以下示例使用 `MEDAGENT_API_TOKEN`。完整接口与请求格式可在服务的 OpenAPI 页面查看，文档路径由 `http.docs_path` 配置。

## 使用示例

以下病历均为虚构示例。将 `API_BASE` 改为自己的服务地址，并设置对应访问令牌。

```bash
export API_BASE="http://localhost:8000/api/v1"
```

### 病历评估

医生手动选择根管治疗任务：

```bash
RUN_ID=$(curl --fail-with-body -sS "$API_BASE/assessments" \
  -H "Authorization: Bearer $MEDAGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"36牙拟行首次根管治疗。患者张口受限。尚未提供根管影像资料。","tasks":["root_canal"],"routing_mode":"manual","instruction":"整理已有因素并评估治疗难度；未记载因素保持未知。"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')

curl --fail-with-body -sS "$API_BASE/runs/$RUN_ID" \
  -H "Authorization: Bearer $MEDAGENT_API_TOKEN"
```

评估为异步任务，需查询 `run_id` 获取结果。结果包含结构化因素、风险等级、原文证据、审校记录和相似病症文档。`tasks` 可选择 `root_canal`、`crack`、`pulp_periapical`、`periodontal`；使用自动路由时设为 `routing_mode: "auto"`，并省略 `tasks`。

### 患者对话

创建会话，再发送患者的口语描述：

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

消息接口同样返回 `run_id`。查询结果、阅读追问后，在任务结束时向同一会话发送下一条回答。

### 知识检索

```bash
curl --fail-with-body -sS "$API_BASE/knowledge/search" \
  -H "Authorization: Bearer $MEDAGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"牙龈出血和牙齿松动的评估依据","task":"periodontal","limit":3}'
```

返回知识片段及来源。启用混合检索时，配置 `retrieval.mode: "hybrid"` 与 `retrieval.reranker.enabled: true`，接入 embedding 和 reranker 服务，并通过 `/knowledge/index` 构建索引。

## 主要接口

路径相对于配置的 API 前缀。

| 接口 | 功能 |
| --- | --- |
| `POST /assessments` | 病历评估与专病路由 |
| `POST /records/review` | 图片与文本审校，图片采用 base64 data URL |
| `POST /sessions` | 创建会话 |
| `POST /sessions/{id}/messages` | 症状对话与追问 |
| `GET /runs/{id}` | 查询异步任务和结果 |
| `GET / POST / DELETE /sessions/{id}/memory` | 查询、授权保存或删除长期记忆 |
| `POST /knowledge/search` | 知识检索 |
| `POST /knowledge/index` | 构建混合检索索引 |
