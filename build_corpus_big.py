#!/usr/bin/env python
"""build_corpus_big (v15, шаги 1-3 большой пересборки поля):
0agi/corpus-zero шарды 1-5 (шард 0 уже декодирован в corpus/) → corpus_0agi_big/.
Гигиена на КОРПУСНОМ уровне: мусорные предложения (цитаты, оглавления,
инструкции) удаляются ДО грануляции; док отклоняется если после чистки:
 - осталось <55% предложений, <700 знаков или <6 предложений;
 - локальная связность consec<0.45 (2-предложенные куски, MiniLM).
Диалоговые транскрипты и SFT-пары пропускаются. Дубликаты — по md5."""
import json, hashlib, re, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import granular_text_field as g

CZ = str(g.ROOT / "0agi" / "corpus-zero")
TOK = str(g.ROOT / "0agi" / "tokenizer" / "0agi-univ2.model")
OUT = str(g.GLM_DIR / "corpus_0agi_big")

DOC_JUNK = re.compile(r"^\s*(instruction|response)\b\s*:?", re.I)
SEEN_MD5 = set()
for d in ("corpus", "corpus_0agi", "corpus_0agi_big"):
    p = os.path.join(str(g.GLM_DIR), d)
    if os.path.isdir(p):
        for fn in os.listdir(p):
            if fn.endswith(".txt"):
                h = hashlib.md5(open(os.path.join(p, fn), "rb").read()).hexdigest()
                SEEN_MD5.add(h)


def sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ«\"'])", text)
    return [p.strip() for p in parts if len(p.split()) >= 4]


def clean_doc(text):
    """Чистка предложений → (тело, доля выброшенного). None если док мусор."""
    if len(re.findall(r"^\s*(?:User|Assistant|USER|ASSISTANT)\s*:", text, re.M)) >= 3 \
            or re.match(r"^\s*USER\s*:", text):
        return None, 1.0                       # диалоговый транскрипт
    sts = sentences(text)
    if len(sts) < 6:
        return None, 1.0
    clean = [s for s in sts
             if not g.CITE_RE.search(s) and not g.JUNK_RE.search(s)
             and not DOC_JUNK.match(s)]
    if len(clean) < max(6, 0.55 * len(sts)):
        return None, 1.0
    body = "\n".join(clean)
    if len(body) < 700:
        return None, 1.0
    return body, 1.0 - len(clean) / len(sts)


def consec_mean(sts, model):
    chunks = [" ".join(sts[i:i + 2]) for i in range(0, len(sts) - 1, 2)]
    if len(chunks) < 3:
        return 1.0
    E = model.encode(chunks[:60], normalize_embeddings=True,
                     batch_size=128, show_progress_bar=False)
    cs = [float(np.dot(E[i], E[i + 1])) for i in range(len(E) - 1)]
    return float(np.mean(cs))


def main(target=8000, shards=(1, 2, 3, 4, 5)):
    import sentencepiece as spm
    from sentence_transformers import SentenceTransformer
    sp = spm.SentencePieceProcessor(model_file=TOK)
    eos = sp.eos_id()
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    os.makedirs(OUT, exist_ok=True)
    kept = dirty = lowc = dup = short = 0
    n_written = len([f for f in os.listdir(OUT) if f.endswith(".txt")])
    for si in shards:
        fp = os.path.join(CZ, f"shard_{si:05d}.bin")
        if not os.path.exists(fp) or kept >= target:
            break
        mm = np.memmap(fp, dtype=np.uint16, mode="r")
        eos_idx = np.where(mm == eos)[0]
        print(f"shard {si}: {len(mm):,} токенов, {len(eos_idx):,} доков", flush=True)
        start = 0
        for end in eos_idx:
            if kept >= target:
                break
            if end - start < 250:              # ~700+ знаков даже до декода
                start = end + 1
                continue
            ids = mm[start:end].astype(np.int32).tolist()
            start = end + 1
            text = sp.decode(ids)
            if len(text) > 12000:
                text = text[:12000]
            body, _frac = clean_doc(text)
            if body is None:
                dirty += 1; continue
            ok_cs = consec_mean(sentences(body), st)
            if ok_cs < 0.45:
                lowc += 1; continue
            h = hashlib.md5(body.encode()).hexdigest()
            if h in SEEN_MD5:
                dup += 1; continue
            SEEN_MD5.add(h)
            n_written += 1
            with open(os.path.join(OUT, f"big_{n_written:06d}.txt"), "w") as fo:
                fo.write(body)
            kept += 1
            if kept % 250 == 0:
                print(f"  {kept} доков (откл: мусор {dirty}, несвязных {lowc}, "
                      f"дублей {dup}, коротких {short})", flush=True)
        del mm
    print(f"\n✅ принято {kept} доков → {OUT}")
    print(f"   отклонено: мусор {dirty}, несвязных {lowc}, дублей {dup}, коротких {short}")


if __name__ == "__main__":
    tgt = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    main(target=tgt)
