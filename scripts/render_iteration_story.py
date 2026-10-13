"""Generate a factual interview draft from completed measured reports."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json
cmp=read_json(ROOT/'runtime/evaluation/comparison.json');old=read_json(ROOT/'runtime/ablation/results.json');new=read_json(ROOT/'runtime/ablation/followup_results.json')
def pct(v):return '尚无可计分结果' if v is None else f'{100*v:.2f}%'
def metrics(m):return f"症状字段准确率 {pct(m['symptom_field_exact']['accuracy'])}、症状 micro-F1 {pct(m['symptom_extraction']['micro_f1'])}、分级准确率 {pct(m['grade']['accuracy'])}"
def change(a,b):return ('提升' if b>=a else '下降')+f'{abs(100*(b-a)):.2f} 个百分点'
adapter=read_json(ROOT/'runtime/ablation/schema_adapter_results.json');paired=read_json(ROOT/'runtime/ablation/paired_schema_results.json')
selected=read_json(ROOT/'runtime/ablation/selected_pipeline_results.json');finalr=selected['tasks']['root_canal'];finalc=selected['tasks']['crack']
live=read_json(ROOT/'runtime/ablation/live_graph_status.json');assert live['status']=='completed';durations='、'.join(str(round(x['elapsed_seconds']))+' 秒' for x in live['results'])
root=cmp['results']['root_canal'];crack=cmp['results']['crack'];r=new['tasks']['root_canal'];c=new['tasks']['crack'];rv=r['variants'];cv=c['variants']
text=f'''# 口腔临床辅助评估 Agent：开发历程草案

> 这是依据现有项目代码和 2026-10-02 的实际对比、消融重建的技术迭代顺序，并非从历史日志还原的开发时间线。下文可用作面试讲述草案，第一人称职责应按实际参与范围保留。完整模型测试与开发消融使用不同集合，分别列数。

## 一、先明确业务，再做基座模型的直接评估

我参与的是面向口腔病历的辅助评估系统。业务主要有两个方向：从根管治疗病历中抽取患者级和患牙级因素，评估治疗难度；从牙隐裂相关病历中抽取风险因素，按项目规则给出风险等级。系统需要保留同一患者不同牙位的关系，正确理解否定描述，并把判断依据定位到病历原文。

最初的可复现起点是 Qwen3-8B 基座加原项目提示词，直接输出结构化因素和评估结果。我先建立了统一测试口径：根管使用 101 例测试病例，屏蔽原文已有的难度结论；牙隐裂原测试集有 20 例，其中 2 例与训练候选集文本重复，所以主结果采用剩余 18 例。漏报、多报患牙和无法解析的输出都保留在分母中，同时把字段准确率与多标签 F1 分开。

根管基座最初使用了 toothNumber、difficultyLevel 等字段别名，在原有严格集成协议下全部无法对齐，因此不能直接把零分当作临床能力。对两模型应用同一组明确字段别名转换、仅接受唯一精确词表匹配后，根管基座的辅助对照是{metrics(adapter['base']['metrics'])}；牙隐裂基座是{metrics(crack['base']['decontaminated'])}。牙隐裂分级是因素标签经过项目规则得到的等级一致性，不是对未来实际发病概率的验证。

## 二、接入领域微调模型，用同口径确认收益

接下来我接入了两个历史 Qwen3-8B 文本微调模型，保持同样的提示词、病历和生成设置。根管微调后是{metrics(adapter['finetuned']['metrics'])}；牙隐裂微调后是{metrics(crack['finetuned']['decontaminated'])}。相对于基座，根管分级准确率{change(adapter['base']['metrics']['grade']['accuracy'],adapter['finetuned']['metrics']['grade']['accuracy'])}，牙隐裂分级准确率{change(crack['base']['decontaminated']['grade']['accuracy'],crack['finetuned']['decontaminated']['grade']['accuracy'])}。

字段适配后的症状分数只衡量可对齐的结构化因素，不能覆盖未能唯一映射的自由描述。我还补做了同一显式 Schema 下的 20 例配对对照，逐例核验 system/user 提示完全相同；两模型使用同样的语法修复和规则处理，详见 paired_schema_results.md，避免把不公平的协议差异归因于训练。

这一步仍然存在工程问题。例如牙隐裂模型会把“25 牙”和“35 牙”合为一条记录，也会把组合记录拆成多个牙位，造成标签归属不一致。根管输出则受到历史训练格式影响，会复制病例元数据、生成 Python 字典形式，甚至遗漏末尾括号。因此，领域训练的效果和结构化输出的可靠性需要分开验证。

## 三、结构化契约也会失败，需要用消融检查提示词

为了让模型输出能够被后续模块直接使用，我增加了固定字段契约、枚举验证和最多一次纠错重试。消融使用固定的 20 例根管病例和 18 例去重牙隐裂病例，选择清单在生成模块结果前冻结。

但第一版根管契约出现了一个具体错误：示例中的“各指标”被模型理解成真实字段名，实际症状字段没有放在正确层级。该版本在 20 例开发消融集上的症状字段准确率为 {pct(old['tasks']['root_canal']['variants']['finetuned_contract']['symptom_field_exact']['accuracy'])}。这说明加了约束不一定会改进输出；示例本身必须与训练格式和业务 Schema 一致。

随后我把患者因素和患牙因素的真实字段名完整列出，禁止额外增加中间层，并区分“字段没有返回”和“病例未记载、字段值为空”。同时增加保守的语法修复：只在字符串完整、容器层级明确时补齐缺少的末尾括号，不填写临床字段，不创造标签。显式契约的首轮输出为{metrics(rv['explicit_contract_raw'])}；只增加语法闭合后为{metrics(rv['explicit_contract_format_only'])}。仍无法通过结构检查的输出才进入有次数上限的模型纠错。

不过，显式新契约加修复后，根管开发集的症状字段准确率为 {pct(rv['explicit_contract_repair']['symptom_field_exact']['accuracy'])}，仍低于保留训练协议时的 {pct(finalr['model_direct']['symptom_field_exact']['accuracy'])}。因此最终版本没有强行统一两个专病的提示词：根管保留训练时的任务模板，在输出侧做结构检查；牙隐裂采用消融中更有效的结构契约。统一的是 Skill 接口与执行约束。这个选择也有取舍：保留训练协议的 20 例规则分级准确率为 {pct(finalr['rules']['grade']['accuracy'])}，低于候选契约的 {pct(rv['rules_and_conservative_review']['grade']['accuracy'])}，只相差 1 例。因此当前选型优先考虑因素提取与协议兼容，还需要独立病例确认，不宣称它在所有指标上最优。

## 四、把因素判读和规则计算分开

模型需要判断病历对应哪些因素，但根管因素等级如何合成为治疗难度、牙隐裂如何加分扣分和封顶，可以交给确定性规则执行。我因此保留了两份结果：模型直接给出的等级，以及按模型抽取因素重新计算的等级。两者不一致时记录冲突，保留原文证据供核查。

在最终选择的同一开发集上，根管使用训练协议与输出检查后，直接等级表现为{metrics(finalr['model_direct'])}；同一批因素通过规则计算后为{metrics(finalr['rules'])}。这里能归因的是等级计算模块：因素没有因为重算规则而改变。此外，在完整 101 例原提示词测试中，同一份微调输出从直接等级 74.31% 到规则重算 82.57%，这组对照也完整保留。牙隐裂原模型只输出因素，没有可直接对照的模型等级，因此不能编造一个“关闭规则”的牙隐裂准确率。

修正后的牙隐裂链路在 18 例开发集上为{metrics(cv['rules_and_conservative_review'])}。这些开发集结果不能与前面的完整根管测试结果拼成一条连续提升曲线。

## 五、信息审校曾经过度纠正，因此收紧职责

我随后接入 Qwen3-VL-8B-Instruct 做病历解析、信息审校和证据说明。第一次文本审校实验并不理想：模型修改了全部 38 例输入文本，并把 18 例牙隐裂病例全部判为未通过。检查原因后发现，它会把“未做某项检查”当成矛盾，也会将不同牙位、不同时间或不同检查方式的结果当成互相冲突。直接把这类改写送回因素模型，会改变牙位和标签关系，反而降低效果。

因此，我把纯文本审校的职责收紧为“提示原文内部可定位的显式冲突”。无图时保持原文不变；未记载不等于矛盾；每条冲突建议必须给出原文中的两处不同引文。修正后，根管审校通过 {r['review_approved']}/{r['review_cases']}，牙隐裂通过 {c['review_approved']}/{c['review_cases']}。这只是减少过度拦截的工程结果，剩余冲突建议仍需专业判断，不能把有两处引文等同于医学判断正确。

对真正带图的输入，系统支持图像转写后再带图审校；只有文本版本被改变时才触发后续回流，旧文本上的因素和证据不能直接复用。本次找回的 4 例原图均完成转写，其中 3 例审校输出结构有效。但 2 例与训练候选重复，原图又保留了文本测试删去的诊断栏，所以我没有把这些结果包装成 OCR 准确率提升或独立视觉诊断性能。

## 六、增加原文证据和 RAG，解决可解释、可追溯的问题

仅有因素和等级，还不足以说明系统为什么这样判断。我增加了原文证据定位，让每个因素附上病历中的逐字片段、起止位置和说明；找不到依据的因素进入待复核流程。在修正后的开发集上，根管证据定位通过 {finalr['gate_and_evidence']['supported_factors']}/{finalr['gate_and_evidence']['factors']}，牙隐裂通过 {finalc['gate_and_evidence']['supported_factors']}/{finalc['gate_and_evidence']['factors']}。片段存在只能证明可定位，不能代替医学语义审核。

RAG 当前使用本地 BM25，知识源是两个项目的原始规则文本，按专病划分，片段保留来源、版本哈希和文本偏移。它参与结果说明，不参与修改症状标签或最终分级。因此，我的消融固定前面的因素和等级，比较有检索、无检索、有原文证据和无原文证据的说明，而不会声称 RAG 提高了当前链路的分级准确率。

在实际因素查询上的专病过滤对照中，不过滤时有 {new['retrieval']['summary']['no_task_filter']['outside_task']} 个返回片段来自另一专病，加入任务过滤后为 {new['retrieval']['summary']['task_filter']['outside_task']} 个。该实验验证知识范围约束，不是临床检索准确率。知识库目前也不能称为已经覆盖教材与指南的完整医学知识库。

## 七、用 LangGraph 统一 Agent 编排

在模块相对明确后，我把 Agent 层统一为 LangGraph。Leader 主图负责病历解析、任务路由、并行审校和专病任务、等待分支完成、核对文本版本，以及条件回流和汇总。根管和牙隐裂分别使用专病子图，内部按提取、有限纠错、检索、证据定位、说明生成执行。Skills 保持独立，主图只管理任务状态与执行顺序。

并行分支通过 reducer 合并各自专病结果；一个专病失败时保留另一个专病的结果；每个病例使用独立 thread_id，SQLite 检查点本地保存执行状态。这样可以核查一次结果经过了哪些节点、使用了哪个版本的文本，而不会把不同病例的上下文混在一起。

路由消融还发现关键词捷径会误处理否定指令。例如“只做根管难度评估，不评估牙隐裂”会错误选择两个专病。修正后，含排除表达的指令交由模型按用户限制选择，10 条人工构造路由指令的匹配数由 {new['routing_original']['correct']}/10 变为 {new['routing_revised']['correct']}/10。剩下的模糊请求采用要求明确任务的策略；这一小组指令不能代表真实用户路由准确率。

当前 LangGraph 版本通过 35 项工程测试，覆盖真实并行到达、分支结果合并、有限重试、文本回流、失败隔离、检查点和病例隔离。模型网关目前仍串行且按请求加载。通过真实 HTTP 入口提交的两例病例已跑通 LangGraph、模型调用和结果保存，耗时约 {durations}，均被标为需要复核。这是两请求同时提交时的工程实测，包含排队和模型加载，不能称为平均或 P95 时延，也不支持简历中的 15—30 秒响应说法。

## 八、当前成果与下一轮重点

这一轮工作的核心收获是建立了可复现的模型对比、逐模块对照和失败追踪，并且把两个历史模型接入了统一的 LangGraph 系统。实验也说明，模块越多并不一定越好：提示词占位符、审校过度纠正和不受控的文本改写都可能抵消微调收益。

证据与质量门控之后，根管可自动通过 {finalr['gate_and_evidence']['accepted']}/{finalr['gate_and_evidence']['total']}，牙隐裂可自动通过 {finalc['gate_and_evidence']['accepted']}/{finalc['gate_and_evidence']['total']}。其余结果明确进入人工复核，不能把被拦截的病例从准确率分母中悄悄删除。当前版本应定位为可追溯的开发验证系统，进一步提高自动处理覆盖率还需要新的独立病例、图像配对和医学专家对证据的评审。

强化学习方面，目前已经准备了训练入口和格式、因素 F1、等级准确率组合奖励，但本轮没有执行新的 CPT、VLM SFT 或 GRPO 训练，也没有取得可归因的 RL 对照结果。因此，当前讲述中我不会把 RL 写成已验证的性能贡献。下一步可以把已发现的多牙归属、否定条件、结构错误和无依据因素做成独立验证集，再决定是否用数据修订、SFT 或 RL 解决。

---

## 证据与报告

- 完整模型对比：`comparison_results.md`。
- 根管字段协议适配审计：`schema_adapter_results.md`。
- 相同显式 Schema 的配对对照：`paired_schema_results.md`。
- 完整测试集的确定性模块消融：`full_postprocess_results.md`。
- 原始模块消融，包括负向结果：`ablation_results.md`。
- 基于失败分析的候选修正：`followup_results.md`。
- 当前选择的各专病配置：`selected_pipeline_results.md`。
- 对照口径与不可归因项：`ablation_protocol.md`。
- LangGraph 实现与验证：`langgraph_migration.md`。

数字来源均为服务器保存的预测与计分结果。开发集上的后续修正属于同一集合上的迭代优化，仍需独立集合确认泛化能力。
'''
(ROOT/'docs/development_journey.md').write_text(text)
print('development_journey.md written')
