#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMP 空模型引擎 v2.0 —— 企业规则政策约束版
新增：
  1. 拦截后回退合规替答（不再只甩错误原话）
  2. 判例库反查短路（命中历史直接返回，跳过 LLM + 检索）

用法:
  python nmp_integrated.py -q "我空调拆装时掉了，保修吗？"
  python nmp_integrated.py          # 交互模式
"""

import re
import json
import yaml
import logging
import argparse
import numpy as np
import requests
import jieba
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# 全局：jieba 分词
# ============================================================
def tokenize(text: str) -> str:
    return " ".join(jieba.cut(text))

# ============================================================
# Vault 加载
# ============================================================
class VaultLoader:
    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.facts = []
        self.constraints = []
        self._load_all()

    def _load_all(self):
        if not self.vault_path.exists():
            print(f"  ⚠️ 事实库目录不存在: {self.vault_path}")
            return
        for f in sorted(self.vault_path.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                self.facts.extend(data.get('facts', []))
                self.constraints.extend(data.get('constraints', []))
                print(f"  📄 已加载: {f.name} ({len(data.get('facts',[]))} 事实, {len(data.get('constraints',[]))} 约束)")
            except Exception as e:
                logging.warning(f"加载 vault {f} 失败: {e}")

    def get_fact_texts(self) -> List[str]:
        texts = []
        for fact in self.facts:
            content = fact.get('content', {})
            if isinstance(content, dict):
                entity = content.get('entity', '')
                attrs = content.get('attributes', {})
                attr_text = ', '.join([f"{k}={v}" for k, v in attrs.items()])
                texts.append(f"{entity}: {attr_text}")
            else:
                texts.append(str(content))
        return texts

    def get_constraint_texts(self) -> List[str]:
        texts = []
        for c in self.constraints:
            content = c.get('content', {})
            name = content.get('name', '')
            forbidden = content.get('forbidden', [])
            texts.append(f"{name}: 禁止 {json.dumps(forbidden, ensure_ascii=False)}")
        return texts

# ============================================================
# 向量检索
# ============================================================
class VectorRetriever:
    def __init__(self, texts: List[str]):
        self.texts = texts
        if texts:
            tokenized = [tokenize(t) for t in texts]
            self.vectorizer = TfidfVectorizer()
            self.embeddings = self.vectorizer.fit_transform(tokenized).toarray()
        else:
            self.vectorizer = None
            self.embeddings = None

    def search(self, query: str, top_k: int = 8, threshold: float = 0.05) -> List[Dict]:
        if not self.texts or self.embeddings is None:
            return []
        q_emb = self.vectorizer.transform([tokenize(query)]).toarray()[0]
        if np.all(q_emb == 0):
            return []
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
        norms = np.where(norms == 0, 1e-10, norms)
        sims = np.dot(self.embeddings, q_emb) / norms
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_idx:
            score = float(sims[idx])
            if score < threshold:
                break
            results.append({"text": self.texts[idx], "score": round(score, 4)})
        return results

# ============================================================
# 四层校验器
# ============================================================
class EnhancedNullModelChecker:
    """四层校验：精确关键词 → 正则模式 → 否定事实 → 语义向量"""

    FORBIDDEN_EXACT = [
        "收取费用", "用户自费", "不在保修范围", "三包规定排除",
        "自行拆装不保", "无法保修", "需要付费维修",
        "运费自理", "转人工处理", "不属于免费保修",
        "需要您自行承担", "需要由您自行承担",
    ]

    FORBIDDEN_PATTERNS = [
        r'需要.{0,5}(自费|付费|承担费用|自行承担)',
        r'(运费|费用|维修费).{0,10}(自理|自行|用户|自己)',
        r'不属于.{0,10}(保修|免费|包修)',
        r'不在.{0,10}(保修|包修).{0,10}范围',
        r'无法.{0,5}保修',
    ]

    NEGATION_PATTERNS = [
        r'不(在|属于|享受|包).{0,15}(保修|包修|免费|8年)',
        r'(不能|不可|无法).{0,10}(保修|包修|免费|维修)',
    ]

    OPPOSITE_PAIRS = {
        "免费": "收费", "包修": "不保", "承担": "自理",
        "全免": "自费", "保修": "不保修",
    }

    def __init__(self, facts: list, constraints: list):
        self.facts = facts
        self.constraints = constraints
        all_texts = facts + constraints
        self.all_texts = all_texts
        if all_texts:
            tokenized = [tokenize(t) for t in all_texts]
            self.vectorizer = TfidfVectorizer()
            self.embeddings = self.vectorizer.fit_transform(tokenized).toarray()
        else:
            self.vectorizer = None
            self.embeddings = None

    def detect_conflict(self, text: str, threshold: float = 0.15) -> Tuple[bool, List[Dict]]:
        sentences = [s.strip() for s in re.split(r"[。！？;；\n]", text) if len(s.strip()) >= 5]
        conflicts = []

        for sent in sentences:
            sent_conflicts = []

            # 第1层：精确关键词
            for kw in self.FORBIDDEN_EXACT:
                if kw in sent:
                    sent_conflicts.append({
                        "type": "keyword_violation",
                        "sentence": sent,
                        "against": f"禁止关键词:「{kw}」",
                        "similarity": 1.0,
                        "layer": 1
                    })
                    break

            # 第2层：正则模式
            if not sent_conflicts:
                for pattern in self.FORBIDDEN_PATTERNS:
                    if re.search(pattern, sent):
                        sent_conflicts.append({
                            "type": "pattern_violation",
                            "sentence": sent,
                            "against": f"匹配禁止模式",
                            "similarity": 0.95,
                            "layer": 2
                        })
                        break

            # 第3层：否定事实
            if not sent_conflicts:
                for pattern in self.NEGATION_PATTERNS:
                    if re.search(pattern, sent):
                        sent_conflicts.append({
                            "type": "negation_violation",
                            "sentence": sent,
                            "against": f"否定保修事实",
                            "similarity": 0.9,
                            "layer": 3
                        })
                        break

            # 第4层：语义向量 + 反义词
            if not sent_conflicts and self.vectorizer is not None:
                sent_emb = self.vectorizer.transform([tokenize(sent)]).toarray()[0]
                if not np.all(sent_emb == 0):
                    norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(sent_emb)
                    norms = np.where(norms == 0, 1e-10, norms)
                    sims = np.dot(self.embeddings, sent_emb) / norms

                    n_facts = len(self.facts)
                    for idx in range(min(n_facts, len(sims))):
                        if sims[idx] > threshold and self._has_opposite(self.facts[idx], sent):
                            sent_conflicts.append({
                                "type": "semantic_conflict",
                                "sentence": sent,
                                "against": self.facts[idx],
                                "similarity": round(float(sims[idx]), 4),
                                "layer": 4
                            })
                            break

                    if not sent_conflicts:
                        for idx in range(n_facts, min(len(self.all_texts), len(sims))):
                            if sims[idx] > threshold:
                                cons_text = self.all_texts[idx]
                                forbidden = re.findall(r'"([^"]+)"', cons_text)
                                for fb in forbidden:
                                    if fb in sent:
                                        sent_conflicts.append({
                                            "type": "constraint_violation",
                                            "sentence": sent,
                                            "against": cons_text,
                                            "similarity": round(float(sims[idx]), 4),
                                            "layer": 4
                                        })
                                        break
                                if sent_conflicts:
                                    break

            conflicts.extend(sent_conflicts)

        # 去重
        seen = set()
        deduped = []
        for c in conflicts:
            key = (c['type'], c['sentence'][:40])
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        return len(deduped) > 0, deduped

    def _has_opposite(self, fact: str, text: str) -> bool:
        for k, v in self.OPPOSITE_PAIRS.items():
            if (k in text and v in fact) or (v in text and k in fact):
                return True
        return False

# ============================================================
# LLM 客户端
# ============================================================
class LLMClient:
    def __init__(self, config: Dict):
        self.backend = config.get('backend', 'deepseek')
        self.model = config.get('model', 'deepseek-chat')
        self.api_key = config.get('api_key', '')
        self.api_base = config.get('api_base', 'https://api.deepseek.com/v1')
        self.temperature = config.get('temperature', 0.7)

    def chat(self, prompt: str) -> str:
        if self.backend in ('openai', 'deepseek'):
            return self._call_api(prompt)
        elif self.backend == 'ollama':
            return self._call_ollama(prompt)
        return "[错误] 未配置 LLM"

    def _call_api(self, prompt: str) -> str:
        if not self.api_key:
            return self._simulate(prompt)
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature
            }
            base = self.api_base.rstrip('/')
            url = f"{base}/chat/completions"
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            return f"[API 错误] {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            return f"[API 调用失败] {e}"

    def _call_ollama(self, prompt: str) -> str:
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": self.temperature}
            }
            resp = requests.post(self.api_base, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["message"]["content"]
            return f"[Ollama 错误] {resp.status_code}"
        except Exception as e:
            return f"[Ollama 连接失败] {e}"

    def _simulate(self, prompt: str) -> str:
        """无 API Key 时模拟 LLM 回答（复现 DeepSeek 幻觉场景）"""
        return (
            "您好，感谢您的来电。关于您反映的自行拆装空调时掉落损坏的情况，"
            "我需要向您说明：\n\n"
            "第一，根据国家《部分商品修理更换退货责任规定》以及我司的保修条款，"
            "自行拆装或非官方授权人员拆动造成的损坏，不属于免费保修范围。"
            "虽然您的空调在8年保修期内，但这种情况属于人为因素导致的故障，需要自费维修。\n\n"
            "第二，关于运费，由于这不属于产品质量问题导致的保修，"
            "往返运输费用也需要由您自行承担。"
        )

# ============================================================
# 审计日志 + 判例库（可读可写）
# ============================================================
class CaseLibrary:
    """
    判例库：写入 + 读取 + 语义检索
    - 写入：每次拦截后记录 question + safe_answer + conflicts
    - 读取：新 query 进来时先查判例库，命中则直接返回 safe_answer
    """

    def __init__(self, case_path: Path):
        self.case_path = case_path
        self.case_path.mkdir(parents=True, exist_ok=True)
        self.case_file = self.case_path / "case_log.jsonl"
        self.vectorizer = None
        self.case_embeddings = None
        self.cases = []
        self._load_cases()

    def _load_cases(self):
        """启动时加载所有历史判例到内存"""
        if not self.case_file.exists():
            return
        try:
            for line in self.case_file.read_text(encoding='utf-8').strip().split('\n'):
                if line.strip():
                    self.cases.append(json.loads(line))
            if self.cases:
                print(f"  📚 已加载判例库: {len(self.cases)} 条历史判例")
                # 建向量索引
                questions = [c['question'] for c in self.cases]
                tokenized = [tokenize(q) for q in questions]
                self.vectorizer = TfidfVectorizer()
                self.case_embeddings = self.vectorizer.fit_transform(tokenized).toarray()
        except Exception as e:
            logging.warning(f"加载判例库失败: {e}")

    def search(self, query: str, threshold: float = 0.6) -> Optional[Dict]:
        """
        语义检索判例库，命中返回该判例，否则返回 None
        threshold=0.6 对 jieba+TF-IDF 来说是比较严格的要求
        """
        if not self.cases or self.vectorizer is None:
            return None

        q_emb = self.vectorizer.transform([tokenize(query)]).toarray()[0]
        if np.all(q_emb == 0):
            return None

        norms = np.linalg.norm(self.case_embeddings, axis=1) * np.linalg.norm(q_emb)
        norms = np.where(norms == 0, 1e-10, norms)
        sims = np.dot(self.case_embeddings, q_emb) / norms

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            case = self.cases[best_idx]
            return {
                "question": case['question'],
                "safe_answer": case.get('safe_answer', ''),
                "conflicts": case.get('conflicts', []),
                "similarity": round(best_sim, 4),
                "hit_count": case.get('hit_count', 0) + 1
            }
        return None

    def save(self, question: str, safe_answer: str, conflicts: List[Dict], facts: List[Dict]):
        """保存新判例"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "safe_answer": safe_answer,
            "conflicts": conflicts,
            "facts_used": [f.get('text', '') for f in facts],
            "hit_count": 0
        }
        self.cases.append(entry)
        # 更新向量索引
        questions = [c['question'] for c in self.cases]
        tokenized = [tokenize(q) for q in questions]
        self.vectorizer = TfidfVectorizer()
        self.case_embeddings = self.vectorizer.fit_transform(tokenized).toarray()

        # 追加写入文件
        with open(self.case_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# ============================================================
# 审计日志
# ============================================================
class AuditLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_path / f"nmp_{datetime.now():%Y%m%d}.log", encoding='utf-8'),
            ]
        )
        self.logger = logging.getLogger("NMP")

    def log(self, question, status, source, conflicts_count, latency_ms):
        self.logger.info(
            f"Q: {question[:60]} | Status: {status} | Source: {source} | "
            f"Conflicts: {conflicts_count} | Latency: {latency_ms}ms"
        )

# ============================================================
# 合规替答生成器
# ============================================================
class SafeAnswerGenerator:
    """
    拦截后不用 LLM 原话，而是用事实库 + 用户问题
    直接拼出确定性的合规回答
    """

    @staticmethod
    def generate(question: str, facts: List[Dict], conflicts: List[Dict]) -> str:
        """用事实库拼合规回答"""
        fact_texts = [f['text'] for f in facts]

        # 简单模板：把事实逐条转成肯定句
        lines = ["您好！根据我们的政策：", ""]
        for t in fact_texts:
            # 把 "key=value" 转成自然语言
            if '=' in t:
                entity_part, attrs_part = t.split(': ', 1) if ': ' in t else (t, '')
                lines.append(f"  ✓ {attrs_part}")
            else:
                lines.append(f"  ✓ {t}")
        lines.append("")
        lines.append("综合上述政策，您的情况处理如下：")

        # 根据冲突类型给出针对性回答
        violated_keywords = ' '.join([c.get('against', '') for c in conflicts])
        answer_detail = SafeAnswerGenerator._infer_answer(question, violated_keywords, fact_texts)
        lines.append(f"  {answer_detail}")

        lines.append("")
        lines.append("请问还有其他可以帮您的吗？")
        return "\n".join(lines)

    @staticmethod
    def _infer_answer(question: str, violated: str, facts: List[str]) -> str:
        """根据问题和冲突，从事实库推断正确回答"""
        q = question

        # 保修相关
        if any(k in q for k in ['保修', '包修', '保吗', '保不保']):
            if any('人为损坏' in f and '包修' in f for f in facts):
                return "您的情况是人为损坏，但在我们的保修范围内——8年整机免费包修，含配件费和人工费，无需您支付任何费用。"
            elif any('保修期限' in f for f in facts):
                return "您的空调在8年保修期内，可以免费维修。"

        # 运费相关
        if any(k in q for k in ['运费', '快递', '邮寄', '寄']):
            if any('运费' in f and ('承担' in f or '免' in f) for f in facts):
                return "保修期内的往返运费由我们企业承担，您不需要支付。"

        # 拆装/跌落相关
        if any(k in q for k in ['拆', '掉', '摔', '碰']):
            if any('自行拆装' in f or '跌落' in f for f in facts):
                return "自行拆装或跌落造成的损坏属于人为损坏，仍在我们的8年包修范围内，免费维修，运费也由我们承担。"

        # 通用兜底
        return "根据企业政策，您的情况属于免费包修范围，无需支付任何费用。"

# ============================================================
# 主引擎
# ============================================================
class NMPEngine:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        cfg = self.config
        print("📂 加载事实库...")
        self.vault = VaultLoader(Path(cfg['nmp']['vault_path']))

        facts_texts = self.vault.get_fact_texts()
        constraints_texts = self.vault.get_constraint_texts()

        print(f"\n✅ 引擎初始化完成")
        print(f"   事实条目: {len(facts_texts)} | 约束条目: {len(constraints_texts)}")

        self.retriever = VectorRetriever(facts_texts) if facts_texts else None
        self.checker = EnhancedNullModelChecker(facts_texts, constraints_texts)
        self.llm = LLMClient(cfg['llm'])

        # 判例库（带索引）
        self.case_lib = CaseLibrary(Path(cfg['nmp']['case_path']))

        # 审计日志
        self.audit = AuditLogger(Path(cfg['audit']['log_path']))

    def query(self, question: str) -> Dict:
        start = datetime.now()

        # ========== 第0步：判例库短路 ==========
        case_hit = self.case_lib.search(question, threshold=0.6)
        if case_hit:
            latency = int((datetime.now() - start).total_seconds() * 1000)
            self.audit.log(question, "case_hit", "case_library", 0, latency)
            return {
                "status": "case_hit",
                "source": "case_library",
                "final": case_hit['safe_answer'],
                "conflicts": case_hit['conflicts'],
                "latency_ms": latency,
                "case_similarity": case_hit['similarity'],
                "hit_count": case_hit['hit_count'],
                "facts": [],
                "llm_raw": ""
            }

        # ========== 第1步：检索事实 ==========
        facts = self.retriever.search(question, top_k=8, threshold=0.05) if self.retriever else []

        if not facts:
            latency = int((datetime.now() - start).total_seconds() * 1000)
            msg = "⚠️ 未检索到相关事实，无法约束。请补充事实库。"
            self.audit.log(question, "no_facts", "-", 0, latency)
            return {"status": "no_facts", "message": msg, "final": msg,
                    "facts": [], "llm_raw": "", "conflicts": [], "latency_ms": latency}

        facts_text = "\n".join([f"- {f['text']}" for f in facts])

        # ========== 第2步：构造 Prompt ==========
        prompt = f"""你可以自由调用自身全部行业知识，仅遵守一条底层规则：
你输出的所有内容，不能与下方客观事实产生矛盾。

【客观确定事实】
{facts_text}

【用户问题/需求】
{question}

请输出你的回答。"""

        # ========== 第3步：调用 LLM ==========
        llm_raw = self.llm.chat(prompt)

        # ========== 第4步：四层校验 ==========
        has_conflict, conflicts = self.checker.detect_conflict(llm_raw, threshold=0.15)

        # ========== 第5步：组装输出 ==========
        if has_conflict:
            # ★ 关键改动：拦截后回退合规替答，不再甩错误原话
            safe_answer = SafeAnswerGenerator.generate(question, facts, conflicts)
            final = safe_answer
            status = "intercepted"
            # 写入判例库
            self.case_lib.save(question, safe_answer, conflicts, facts)
        else:
            final = llm_raw
            status = "passed"

        latency = int((datetime.now() - start).total_seconds() * 1000)
        source = "case_library" if status == "case_hit" else ("safe_fallback" if status == "intercepted" else "llm_direct")
        self.audit.log(question, status, source, len(conflicts), latency)

        return {
            "status": status,
            "source": source,
            "final": final,
            "llm_raw": llm_raw if has_conflict else "",
            "conflicts": conflicts,
            "latency_ms": latency,
            "facts": facts
        }

# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="NMP 空模型事实约束引擎 v2.0")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--question", "-q", help="单次查询")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"❌ 配置文件不存在: {args.config}")
        return

    engine = NMPEngine(args.config)

    if args.question:
        result = engine.query(args.question)
        print("\n" + "="*60)
        print(result['final'])
        print("="*60)
        print(f"  [状态: {result['status']} | 耗时: {result.get('latency_ms',0)}ms]")
        return

    # 交互模式
    print("\n🧠 NMP 空模型事实约束引擎 v2.0 - 交互模式")
    print("💡 试试: 我拆空调时掉了，保修吗？运费包吗？")
    print("   输入 'quit' 退出\n")

    while True:
        try:
            q = input("💬 你: ").strip()
            if q.lower() in ('quit', 'exit', 'q'):
                break
            if not q:
                continue
            result = engine.query(q)
            print(f"\n{'─'*60}")
            print(result['final'])
            src = result.get('source', '?')
            print(f"{'─'*60}  [{result['status']} | src={src} | {result.get('latency_ms',0)}ms]\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
    print("👋 再见")

if __name__ == "__main__":
    main()
