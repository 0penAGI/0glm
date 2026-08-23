#!/usr/bin/env python
"""rebuild_caches (v15): пересборка ВСЕХ кэшей поля после ребилда пула.
Порядок: sems (MiniLM, долго) → positions → affect → PCA → hub.
Запускать КАЖДЫЙ раз после --refresh --rescan --corpus-only."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import granular_text_field as g


def main():
    pool = g.load_pool()
    lens = {lv: len(pool[f"{n}_feats"])
            for lv, n in ((0, "micro"), (1, "meso"), (2, "macro"))}
    print(f"пул: μ={lens[0]} σ={lens[1]} Ω={lens[2]}")

    print("\n1/5 семантика (MiniLM)…")
    sems = g.build_semantics(pool, force=False)  # свежий кэш не пере-энкодим

    print("\n2/5 позиции в документах…")
    g.build_positions(pool, force=True)

    print("\n3/5 аффект (VADER)…")
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        out = {}
        for lv in (0, 1, 2):
            arr = np.zeros((lens[lv], 2), dtype=np.float32)
            texts = pool[f"{['micro','meso','macro'][lv]}_texts"]
            for i, t in enumerate(texts):
                sc = sia.polarity_scores(t[:1000])
                arr[i, 0] = np.clip(sc["compound"], -1, 1)
                arr[i, 1] = np.clip(sc["pos"] + sc["neg"]
                                    + t.count("!") * 0.1 + t.count("?") * 0.05, 0, 1)
            out[lv] = arr
            print(f"   {['micro','meso','macro'][lv]}: {len(texts)}")
        np.savez(str(g.GLM_DIR / "pool" / "text_affect_v1.npz"),
                 micro=out[0], meso=out[1], macro=out[2])
    except Exception as e:
        print(f"   ⚠️ affect пропущен: {e}")

    print("\n4/5 PCA семантик…")
    g.fit_sem_projection(pool, sems)

    print("\n5/6 анти-хаб…")
    if os.path.exists(g.HUB_CACHE):
        os.remove(g.HUB_CACHE)
    import json
    with open(g.CLUSTERS_CACHE) as f:
        clusters = json.load(f)
    eng2 = g.TextGrainEngine(pool, clusters)
    eng2.attach_sems(sems)
    g.compute_hub_scores(eng2, force=True)

    print("\n6/6 док-эмбеддинги (первый этап ретрива)…")
    g.build_doc_embs(pool, sems, force=True)

    print("\n✅ все кэши пересобраны")


if __name__ == "__main__":
    main()
