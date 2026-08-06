#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示 v2.0 两大新功能：
  1. 拦截后回退合规替答（不再甩错误原话）
  2. 判例库短路（第二次同样问题直接命中，跳过 LLM）

场景：空调拆装跌落，DeepSeek 幻觉说"不保、自费"
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from nmp_integrated import NMPEngine

# 清空旧判例库，保证演示干净
import json
from pathlib import Path
case_file = Path("cases/case_log.jsonl")
case_file.parent.mkdir(parents=True, exist_ok=True)
case_file.write_text("", encoding='utf-8')

# 清日志
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
for f in log_dir.glob("*.log"):
    f.unlink()

engine = NMPEngine("config.yaml")

QUESTION = "我买的空调在我拆装的时候掉了，请问保修吗，运费也包吗？"

# ============================================================
# 第一次：走全流程（LLM → 校验 → 拦截 → 生成合规替答 → 写入判例库）
# ============================================================
print("\n" + "█" * 65)
print("▶ 第一次提问（冷启动：LLM调用 + 四层校验 + 写入判例库）")
print("█" * 65)
print(f"\n👤 用户: {QUESTION}\n")

t0 = time.time()
result1 = engine.query(QUESTION)
elapsed1 = int((time.time() - t0) * 1000)

print(f"{'─'*65}")
print(f"📤 最终输出给用户:\n")
print(result1['final'])
print(f"\n{'─'*65}")
print(f"  📊 状态:     {result1['status']}")
print(f"  📊 来源:     {result1.get('source', '?')}")
print(f"  📊 冲突数:   {len(result1.get('conflicts', []))}")
print(f"  📊 耗时:     {result1.get('latency_ms', elapsed1)}ms")
if result1.get('conflicts'):
    print(f"\n  🔍 拦截详情:")
    for i, c in enumerate(result1['conflicts'], 1):
        print(f"     [{i}] 第{c['layer']}层 | {c['type']} | 置信度:{c['similarity']}")

# ============================================================
# 第二次：同样问题，命中判例库，直接返回（跳过 LLM + 检索 + 校验）
# ============================================================
print("\n\n" + "█" * 65)
print("▶ 第二次提问（同样问题 → 判例库命中 → 直接返回，零 LLM 调用）")
print("█" * 65)
print(f"\n👤 用户: {QUESTION}\n")

t0 = time.time()
result2 = engine.query(QUESTION)
elapsed2 = int((time.time() - t0) * 1000)

print(f"{'─'*65}")
print(f"📤 最终输出给用户:\n")
print(result2['final'])
print(f"\n{'─'*65}")
print(f"  📊 状态:     {result2['status']}")
print(f"  📊 来源:     {result2.get('source', '?')}")
print(f"  📊 判例相似度: {result2.get('case_similarity', 0)}")
print(f"  📊 累计命中: 第 {result2.get('hit_count', 0)} 次")
print(f"  📊 耗时:     {result2.get('latency_ms', elapsed2)}ms")

# ============================================================
# 第三次：轻微改写，仍然命中判例库
# ============================================================
QUESTION3 = "空调自己拆的时候摔了，还保不保？运费谁出？"
print("\n\n" + "█" * 65)
print("▶ 第三次提问（改写措辞 → 判例库语义命中 → 仍然短路）")
print("█" * 65)
print(f"\n👤 用户: {QUESTION3}\n")

t0 = time.time()
result3 = engine.query(QUESTION3)
elapsed3 = int((time.time() - t0) * 1000)

print(f"{'─'*65}")
print(f"📤 最终输出给用户:\n")
print(result3['final'])
print(f"\n{'─'*65}")
print(f"  📊 状态:     {result3['status']}")
print(f"  📊 来源:     {result3.get('source', '?')}")
print(f"  📊 判例相似度: {result3.get('case_similarity', 0)}")
print(f"  📊 累计命中: 第 {result3.get('hit_count', 0)} 次")
print(f"  📊 耗时:     {result3.get('latency_ms', elapsed3)}ms")

# ============================================================
# 总结对比
# ============================================================
print("\n\n" + "█" * 65)
print("📊 效果对比总结")
print("█" * 65)
print(f"""
┌─────────────┬────────────┬────────────┬──────────────────────────┐
│ 轮次        │ 耗时       │ 是否调LLM │ 输出内容                 │
├─────────────┼────────────┼────────────┼──────────────────────────┤
│ 第1次(冷启)│ {elapsed1:>4}ms    │ ✅ 是      │ 合规替答(事实库生成)  │
│ 第2次(相同)│ {elapsed2:>4}ms    │ ❌ 否      │ 判例库直接返回        │
│ 第3次(改写)│ {elapsed3:>4}ms    │ ❌ 否      │ 判例库语义命中返回    │
└─────────────┴────────────┴────────────┴──────────────────────────┘

关键改进:
  ✅ 拦截后不再甩错误原话 → 用事实库生成确定性的合规回答
  ✅ 判例库从"只写不读"→ 可读可查可短路
  ✅ 同样/相似问题第二次起 → 跳过 LLM + 跳过检索 → 毫秒级响应
  ✅ 审计日志记录每次调用的来源(case_library / safe_fallback / llm_direct)
""")
