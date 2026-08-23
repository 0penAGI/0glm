"""
0GLM-Q — query-conditioned grain navigation.
Учим навигатор отвечать на запросы, а не просто генерировать текст.

Команды:
  --build              # 0agi corpora (universe2/corpus/universe) → corpus_qa/*.txt + qa_map.json
  --ask "вопрос"       # retrieval (TF-IDF по гранулам пула) + cond запроса → генерация ответа

Механика:
  build: каждый QA-док пишется как "Q: <prompt>\n\nA: <response>", стилометрия промпта
         складывается в qa_map.json; при extract_all траектории этих доков получают
         cond = вектор запроса → train_multi учит cond_proj «отвечать» на класс запроса.
  ask:   TF-IDF cosine top-k релевантных гранул → их фичи сеют контекст, их кластеры
         получают буст при выборке; cond = стилометрия запроса на каждом шаге.
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from pathlib import Path

import granular_text_field as g

GLM = Path(__file__).resolve().parent
QA_DIR = GLM / "corpus_qa"
AGI_CORPUS = GLM.parent / "0agi" / "corpus"

CODE_RE = re.compile(r"^\s*(import |from |def |class |#include|const |var |let |function )")
DIALOGUE_RE = re.compile(r"USER:\s*(.*?)\s*ASSISTANT:\s*(.*)", re.S)


def looks_like_code(t):
    return bool(CODE_RE.match(t)) or t.count("def ") > 2 or t.count("{") > 5


def split_qa_universe2(text):
    """universe2: 'инструкция\n\nответ'."""
    parts = text.split("\n\n", 1)
    if len(parts) != 2: return None
    q, a = parts[0].strip(), parts[1].strip()
    if len(q) < 20 or len(a) < 100: return None
    if looks_like_code(q) or looks_like_code(a[:200]): return None
    return q, a


def split_qa_dialogue(text):
    """corpus/universe: 'USER:\n...\n\nASSISTANT:\n...'."""
    m = re.search(r"USER:\s*(.+?)\n+ASSISTANT:\s*(.+)", text, re.S)
    if not m: return None
    q, a = m.group(1).strip(), m.group(2).strip()
    # отрезаем вложенные диалоги из промпта
    q = re.split(r"\nAssistant:|\nUser:", q)[0].strip()
    if len(q) < 10 or len(a) < 100: return None
    return q, a


def cmd_build(args):
    QA_DIR.mkdir(exist_ok=True)
    qa_map_path = g.POOL_DIR / "qa_map.json"
    sources = [
        (AGI_CORPUS / "universe2.jsonl", split_qa_universe2),
        (AGI_CORPUS / "corpus.jsonl", split_qa_dialogue),
        (AGI_CORPUS / "universe.jsonl", split_qa_dialogue),
    ]
    n = 0
    qa_map = {}
    for path, splitter in sources:
        if n >= args.max_pairs: break
        if not path.exists(): continue
        print(f"📖 {path.name}")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if n >= args.max_pairs: break
                try: doc = json.loads(line)
                except Exception: continue
                if not isinstance(doc, str): doc = doc.get("text", "")
                pair = splitter(doc)
                if pair is None: continue
                q, a = pair
                a = a[:args.max_resp_chars]
                fn = f"qa_{n:06d}.txt"
                (QA_DIR / fn).write_text(f"Q: {q}\n\nA: {a}", encoding="utf-8")
                qf = g.extract_feat_from_text(q)
                if qf is not None:
                    qa_map[fn] = [round(float(x), 4) for x in qf]
                n += 1
        print(f"   total: {n}")
    with open(qa_map_path, "w") as f:
        json.dump(qa_map, f)
    print(f"✅ {n} QA docs → {QA_DIR}, map → {qa_map_path}")


def load_engine_and_model(model_path=None):
    pool = g.load_pool()
    with open(g.CLUSTERS_CACHE) as f: clusters = json.load(f)
    engine = g.TextGrainEngine(pool, clusters)
    model = g.MultiNavigator().to(g.DEVICE)
    mp = model_path or g.MODEL_MULTI_CACHE
    if not os.path.exists(mp):   # относительный путь → от каталога 0glm/
        alt = Path(__file__).resolve().parent / mp
        if alt.exists(): mp = str(alt)
    g.safe_load(model, mp)
    model.eval()
    return model, engine, pool


def torch_load(p):
    import torch
    return torch.load(p, map_location=g.DEVICE, weights_only=False)


class Retriever:
    """TF-IDF по текстам гранул пула; cosine top-k."""

    def __init__(self, pool, max_texts=60000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        all_texts = []
        for ln in ("micro", "meso", "macro"):
            all_texts.extend(pool[f"{ln}_texts"])
        if len(all_texts) > max_texts:
            self.idx_map = np.random.RandomState(42).choice(
                len(all_texts), max_texts, replace=False)
        else:
            self.idx_map = np.arange(len(all_texts))
        texts = [all_texts[i] for i in self.idx_map]
        self.vec = TfidfVectorizer(max_features=30000, stop_words="english", sublinear_tf=True)
        self.mat = self.vec.fit_transform(texts)
        self.texts = texts
        print(f"🔎 Retriever: {self.mat.shape[0]} grains × {self.mat.shape[1]} tfidf dims")

    def topk(self, query, k=8):
        qv = self.vec.transform([query])
        sims = (self.mat @ qv.T).toarray().ravel()
        idx = np.argsort(-sims)[:k]
        return [(self.texts[i], float(sims[i]), int(self.idx_map[i]))
                for i in idx if sims[i] > 0.01]


def cmd_ask(args):
    import torch
    model, engine, pool = load_engine_and_model(args.model)
    if getattr(args, "sem", False):
        engine.attach_sems(g.build_semantics(pool))
    retr = Retriever(pool)

    hits = retr.topk(args.ask, k=args.k)
    print(f"\n🔍 Retrieved {len(hits)} grains:")
    for t, s, _c in hits[:3]:
        print(f"   [{s:.3f}] {t[:90]}...")

    q_feat = g.extract_feat_from_text(args.ask)
    if q_feat is None: q_feat = np.zeros(g.FEAT_DIM, dtype=np.float32)

    # контекст сеем релевантными гранулами: их фичи → ctx, кластеры → буст
    all_f = np.concatenate([pool["micro_feats"], pool["meso_feats"], pool["macro_feats"]])
    hit_feats = []
    for t, s, _c in hits:
        f = g.extract_feat_from_text(t)
        if f is not None: hit_feats.append(f)
    # кластеры хитов через cluster_map по ближайшим фичам
    inv = {}
    for cid, members in engine.cluster_map.items():
        for ln, gi in members:
            inv[(ln, gi)] = cid
    boosted_clusters = set()
    for t, s, _c in hits:
        f = g.extract_feat_from_text(t)
        if f is None: continue
        best_cid, best_d = None, 1e18
        for ln_off, ln_name in ((0, "micro"), (1, "meso"), (2, "macro")):
            arr = pool[f"{ln_name}_feats"]
            d = np.linalg.norm(arr - f, axis=1)
            j = int(np.argmin(d))
            if d[j] < best_d:
                best_d = d[j]
                best_cid = inv.get((ln_off, j))
        if best_cid is not None: boosted_clusters.add(best_cid)

    ctx_seed = np.array(hit_feats[:g.CONTEXT_LEN], dtype=np.float32)
    if len(ctx_seed) < g.CONTEXT_LEN:
        pad = all_f[np.random.choice(len(all_f), g.CONTEXT_LEN - len(ctx_seed), replace=False)]
        ctx_seed = np.concatenate([ctx_seed, pad], axis=0)

    text, z_arr, stream_steps = g.generate_multi(
        model, engine, pool, n_steps=args.steps, seed=args.seed, temp=args.temperature,
        target_stats=q_feat, noise_inject=0.0,
        ctx_init=ctx_seed, boost_clusters=boosted_clusters)

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_md = os.path.join(g.OUT, f"glmq_{ts}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# 0GLM-Q answer\n\n**Q:** {args.ask}\n\n---\n\n{text}\n")
    print(f"\n✅ {out_md}\n\n{'─'*60}\nQ: {args.ask}\nA: {text[:900]}\n{'─'*60}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--max-pairs", type=int, default=2000)
    p.add_argument("--max-resp-chars", type=int, default=4000)
    p.add_argument("--ask", type=str, default=None)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--sem", action="store_true", help="LSA semantic rerank (A/B winner)")
    args = p.parse_args()

    if args.build:
        cmd_build(args)
    elif args.ask:
        cmd_ask(args)
    else:
        print("use --build or --ask \"...\"")


if __name__ == "__main__":
    main()
