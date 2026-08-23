"""
BUILD CORPUS GLM — декодирование corpus-zero (uint16 токены) обратно в текст.
0agi/corpus-zero/shard_*.bin --sentencepiece--> 0glm/corpus/doc_XXXXX.txt

Зеркало build_corpus_zero.py в обратную сторону: там текст → токены,
здесь токены → текст (документы разделены EOS).
"""
import argparse
from pathlib import Path
import numpy as np
import sentencepiece as spm

GLM = Path(__file__).resolve().parent
CZ = GLM.parent / "0agi" / "corpus-zero"
OUT = GLM / "corpus"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(GLM.parent / "0agi/tokenizer/0agi-univ2.model"))
    ap.add_argument("--docs", type=int, default=2000, help="how many docs to write")
    ap.add_argument("--shards", type=int, default=1, help="how many shards to walk")
    ap.add_argument("--min-tokens", type=int, default=80, help="skip docs shorter than this")
    ap.add_argument("--max-chars", type=int, default=8000, help="truncate long docs")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    existing = list(OUT.glob("doc_*.txt"))
    if existing and not args.refresh:
        print(f"📦 already {len(existing)} docs in {OUT} — use --refresh to rebuild")
        return

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    eos = sp.eos_id()
    print(f"tokenizer vocab={sp.get_piece_size()} eos={eos}")

    n_written = 0
    for si in range(args.shards):
        if n_written >= args.docs: break
        fp = CZ / f"shard_{si:05d}.bin"
        if not fp.exists(): continue
        mm = np.memmap(fp, dtype=np.uint16, mode="r")
        eos_idx = np.where(mm == eos)[0]
        print(f"shard {si}: {len(mm):,} tokens, {len(eos_idx):,} docs")
        start = 0
        for end in eos_idx:
            if end - start < args.min_tokens:
                start = end + 1
                continue
            ids = mm[start:end].astype(np.int32).tolist()
            start = end + 1
            text = sp.decode(ids)
            if len(text) > args.max_chars: text = text[:args.max_chars]
            if len(text) < 200: continue
            n_written += 1
            (OUT / f"doc_{n_written:06d}.txt").write_text(text, encoding="utf-8")
            if n_written % 500 == 0: print(f"  written {n_written}")
            if n_written >= args.docs: break
        del mm
    print(f"✅ {n_written} docs → {OUT}")


if __name__ == "__main__":
    main()
