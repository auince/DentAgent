import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json

r=read_json(ROOT/'runtime/evaluation/comparison.json')
def pct(x):return 'N/A' if x is None else f'{100*x:.2f}%'
def change(a,b):return 'N/A' if a is None or b is None else f'{100*(b-a):+.2f} 个百分点'
lines=['# 基座与微调模型业务指标对比','',
    '基座：Qwen3-8B。根管微调：2026-05-05 checkpoint-800；牙隐裂微调：2026-09-11 merged-300。',
    '两侧使用相同输入、提示词、非思考模式、贪心解码和 BF16，batch size=4。此为模型输出比较，不是完整 Agent 或视觉链路评测。',
    '**根管基座的零分是严格字段协议下的集成失败，不能解读为临床判读能力为零。基座使用 toothNumber、difficultyLevel 等别名；确定性适配后的辅助结果见 schema_adapter_results.md，同一完整 Schema 下的开发集对照另报。**','']
for task,title in [('root_canal','根管治疗'),('crack','牙隐裂')]:
    for subset,label in [('all','全量测试集'),('decontaminated','剔除训练候选文本/ID重叠与测试集重复')]:
        if task=='root_canal' and subset=='decontaminated':continue
        a=r['results'][task]['base'][subset];b=r['results'][task]['finetuned'][subset]
        lines += [f'## {title}：{label}', '',f"病例数：{a['samples']}；金标牙位记录：{a['grade']['gold_teeth']}。",'',
            '| 指标 | 基座 | 微调后 | 变化 |','|---|---:|---:|---:|']
        pairs=[('症状因素严格字段匹配率',a['symptom_field_exact']['accuracy'],b['symptom_field_exact']['accuracy']),
            ('症状因素 micro-F1',a['symptom_extraction']['micro_f1'],b['symptom_extraction']['micro_f1']),
            ('等级准确率',a['grade']['accuracy'],b['grade']['accuracy']),
            ('整例等级完全一致率',a['grade']['case_exact_accuracy'],b['grade']['case_exact_accuracy'])]
        for name,x,y in pairs:lines.append(f'| {name} | {pct(x)} | {pct(y)} | {change(x,y)} |')
        lines += ['',f"结构与牙位字段符合评分器协议：{a['parse_ok']}/{a['samples']} → {b['parse_ok']}/{b['samples']}；可进入规则处理：{a['parse_schema_ok']}/{a['samples']} → {b['parse_schema_ok']}/{b['samples']}。前者不是 JSON 语法成功率，后者不代表所有临床因素完整。",'',
            f"等级命中：{a['grade']['correct']}/{a['grade']['total']} → {b['grade']['correct']}/{b['grade']['total']}；多报牙位记录：{a['grade']['extra_teeth']} → {b['grade']['extra_teeth']}，漏报：{a['grade']['missing_teeth']} → {b['grade']['missing_teeth']}。",'']
lines += ['## 口径与限制','',
    '- 提取字段：根管为诊断、牙周病损、根管感染程度；牙隐裂为咬合受力、临床激发体征、裂纹根部波及征象。',
    '- 严格字段匹配率要求该字段所有选项完全一致；micro-F1 同时计算误提和漏提，不将未出现的选项当作大量正确负例。',
    '- 根管比较模型直接输出的 difficulty 与原金标；牙隐裂从预测/金标因素使用相同项目规则重算等级。',
    '- 等级分母为金标与预测牙位记录的并集，多报、漏报和无效等级均计错；组合牙位保留原数据记录粒度。',
    '- 根管 74 例已有旧难度文本统一屏蔽；有 1 例金标等级与规则重算冲突，保留原标注并披露。',
    '- 牙隐裂 20 例中有 2 例与训练候选文本重复，18 例去重子集作为补充；不能证明患者级独立。',
    '- 结构化解析失败计错；其他非目标字段的 schema 错误不抹掉本任务正确结果。整份 schema 严格门控的结果另外保存在 JSON 的 strict_all / strict_decontaminated 中。',
    '- 样本规模小且类别不均衡，数字只能支持当前测试集；未测 OCR、VLM、RAG 或完整 Agent 的新增收益。','']
(ROOT/'docs/comparison_results.md').write_text('\n'.join(lines))
print('\n'.join(lines))
