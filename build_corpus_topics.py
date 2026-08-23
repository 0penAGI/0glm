#!/usr/bin/env python
"""build_corpus_topics (v18c): таргет-харвест плотности.
Полисемия лечится объёмом тонких кластеров, не патчами. Два потока:
  1) TOPIC-доки (python-programming, resume/career) — гигиена та же,
     лимит отдельный;
  2) GENERAL-доки — обычная гигиена build_corpus_big, добор плотности.
md5-дедуп против всех корпусов. Гигиена импортируется из build_corpus_big."""
import json, hashlib, re, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import granular_text_field as g
import build_corpus_big as base

CZ = str(g.ROOT / "0agi" / "corpus-zero")
TOK = str(g.ROOT / "0agi" / "tokenizer" / "0agi-univ2.model")
OUT = str(g.GLM_DIR / "corpus_0agi_big")

# --- таргет-семейства (регистр важен: 'resume'-глагол отсекается формами) ---
PROG = re.compile(
    r"(\bPython\b(?! 3\.\d))|\b(import\s+\w+\s*$)|(\bdef\s+\w+\()|"
    r"\brecursion\b|\brecursive\b|\bdictionary\b|\blist comprehension\b|"
    r"\bprogramming language\b|\bdata structure\b|\bobject-oriented\b",
    re.I)
RESUME = re.compile(
    r"\br[ée]sum[ée]\b|\bcover letter\b|\bcurriculum vit[aæ]e\b|"
    r"\bhiring manager\b|\bjob interview\b|\bapplicant tracking\b", re.I)


def topic_of(text):
    p = len(PROG.findall(text))
    r = len(RESUME.findall(text))
    if p >= 4:
        return "prog"
    if r >= 3:
        return "resume"
    return None


def main(general_target=2500, topic_target=400):
    import sentencepiece as spm
    from sentence_transformers import SentenceTransformer
    sp = spm.SentencePieceProcessor(model_file=TOK)
    eos = sp.eos_id()
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    os.makedirs(OUT, exist_ok=True)
    n_written = len([f for f in os.listdir(OUT) if f.endswith(".txt")])
    kept_gen = {"prog": 0, "resume": 0}
    kept_g = dirty = lowc = dup = 0
    BATCH = 2000          # кандидатов в батче consec (MPS любит крупные прогоны)
    CH = 24               # чанков на док (хватает для оценки связности)

    def flush(cands):
        """Батч- consec по накопленным кандидатам; пишет прошедших.
        cands: список (body, topic|None) — topic для таргет-доков."""
        nonlocal kept_g, lowc, dup, n_written
        if not cands:
            return
        chunks, spans = [], []
        for b, _t in cands:
            cs = base.sentences(b)[:CH]
            pairs = [" ".join(cs[i:i + 2]) for i in range(0, len(cs) - 1, 2)]
            spans.append((len(chunks), len(pairs)))
            chunks.extend(pairs)
        E = st.encode(chunks or ["."], normalize_embeddings=True,
                      batch_size=256, show_progress_bar=False)
        for ci, (b, t) in enumerate(cands):
            s0, ln = spans[ci]
            if ln < 3:
                ok_cs = 1.0
            else:
                e = E[s0:s0 + ln]
                ok_cs = float(np.mean([float(np.dot(e[i], e[i + 1]))
                                       for i in range(ln - 1)]))
            if ok_cs < 0.45:
                lowc += 1
                continue
            h = hashlib.md5(b.encode()).hexdigest()
            if h in base.SEEN_MD5:
                dup += 1
                continue
            base.SEEN_MD5.add(h)
            n_written += 1
            with open(os.path.join(OUT, f"big_{n_written:06d}.txt"), "w") as fo:
                fo.write(b)
            if t:
                kept_gen[t] += 1
            else:
                kept_g += 1

    for si in range(1, 6):
        fp = os.path.join(CZ, f"shard_{si:05d}.bin")
        if not os.path.exists(fp):
            continue
        mm = np.memmap(fp, dtype=np.uint16, mode="r")
        eos_idx = np.where(mm == eos)[0]
        print(f"shard {si}: {len(eos_idx):,} доков "
              f"[gen {kept_g}/{general_target} prog {kept_gen['prog']} "
              f"resume {kept_gen['resume']}]", flush=True)
        start = 0
        cands = []
        for end in eos_idx:
            if kept_g >= general_target and \
               all(v >= topic_target for v in kept_gen.values()):
                break
            if end - start < 250:
                start = end + 1
                continue
            ids = mm[start:end].astype(np.int32).tolist()
            start = end + 1
            text = sp.decode(ids)
            if len(text) > 12000:
                text = text[:12000]
            top = topic_of(text)
            need_topic = top and kept_gen[top] < topic_target
            if not need_topic and kept_g >= general_target:
                continue
            body, _f = base.clean_doc(text)
            if body is None:
                dirty += 1
                continue
            cands.append((body, top if need_topic else None))
            if len(cands) >= BATCH:
                flush(cands)
                cands = []
                print(f"  прогон: +{kept_g + sum(kept_gen.values())} "
                      f"(мусор {dirty}, несвязн {lowc}, дублей {dup})", flush=True)
        flush(cands)
        del mm
        print(f"  shard {si} закрыт: gen {kept_g}, prog {kept_gen['prog']}, "
              f"resume {kept_gen['resume']}", flush=True)
        if kept_g >= general_target and \
           all(v >= topic_target for v in kept_gen.values()):
            break
    print(f"\n✅ general +{kept_g}, prog +{kept_gen['prog']}, "
          f"resume +{kept_gen['resume']} → {OUT}")
    print(f"   отклонено: мусор {dirty}, несвязных {lowc}, дублей {dup}")
    json.dump({"general": kept_g, **kept_gen},
              open(os.path.join(str(g.GLM_DIR), "harvest_stats.json"), "w"))


if __name__ == "__main__":
    gt = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    tt = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    main(gt, tt)
