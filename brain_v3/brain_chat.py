"""Мозг v0 в диалоге: грузит ОДИН файл (brain_v0.pt) и больше ничего.
Планирование — в весах (qsem-условие вопросом), чтение — адресуемое
припоминание из встроенной памяти. Провенанс каждой реплики сохранён."""
import sys, os, json, re, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import granular_text_field as g

def load_brain(path):
    b = torch.load(path, map_location=g.DEVICE, weights_only=False)
    model = g.MultiNavigator(feat_dim=b["feat_dim"]).to(g.DEVICE)
    model.load_state_dict(b["model_state"], strict=True)
    model.eval()
    pool = b["pool"]
    sems = [b["sems"][str(lv)] if str(lv) in b["sems"] else b["sems"][lv]
            for lv in range(3)]
    sems = [np.asarray(s, dtype=np.float32) for s in sems]
    # проекции всех гранул (для строк контекста)
    comp, mu = b["proj_components"], b["proj_mean"]
    all_sems = np.concatenate(sems, axis=0)
    all_proj = ((all_sems - mu) @ comp).astype(np.float32)
    all_f = np.concatenate([pool["micro_feats"], pool["meso_feats"],
                            pool["macro_feats"]]).astype(np.float32)
    aff = b["affect"]
    all_a = np.concatenate([np.asarray(aff[0], np.float32),
                            np.asarray(aff[1], np.float32),
                            np.asarray(aff[2], np.float32)], axis=0)
    # матрица припоминания: микро+мезо (говоримые уровни)
    n_mic, n_mes = len(pool["micro_texts"]), len(pool["meso_texts"])
    M = np.concatenate([sems[0], sems[1]], axis=0)
    ok_mask = np.asarray(b.get("ok_mask"),
                         dtype=bool) if b.get("ok_mask") is not None else None
    coords = [(0, i) for i in range(n_mic)] + [(1, i) for i in range(n_mes)]
    texts = list(pool["micro_texts"]) + list(pool["meso_texts"])
    inv = {}
    for ti, traj in enumerate(pool["trajectories"]):
        for l, i in traj:
            if int(l) <= 1:
                inv[(int(l), int(i))] = ti
    return dict(model=model, pool=pool, sems=sems, all_proj=all_proj,
                all_f=all_f, all_a=all_a, M=M, coords=coords, texts=texts,
                doc_embs=np.asarray(b["doc_embs"], dtype=np.float32),
                feat_dim=b["feat_dim"], inv=inv, ok_mask=ok_mask,
                policy=b.get("policy", {}), meta=b.get("meta", {}))
def row(B, lv, idx):
    gi = g._level_to_all_f_idx(lv, idx,
        len(B["pool"]["micro_feats"]), len(B["pool"]["meso_feats"]))
    return np.concatenate([B["all_f"][gi], B["all_a"][gi], B["all_proj"][gi]])

def ask(B, Q, st, n_steps=24, trace=False):
    pol = B["policy"]; CL = pol.get("context_len", 12); AR = pol.get("anti_repeat", 4)
    v = st.encode([Q])[0]; qn = (v / np.linalg.norm(v)).astype(np.float32)
    # ── глаза: куда смотреть первым (док-первый по встроенной памяти) ──
    dsim = B["doc_embs"] @ qn
    d = int(np.argmax(dsim))
    members = [(int(l), int(i)) for l, i in B["pool"]["trajectories"][d] if int(l) <= 1]
    off = {0: 0, 1: len(B["pool"]["micro_texts"])}
    ok = []
    for k, (lv, ix) in enumerate(members):
        t = B["texts"][off[lv] + ix] if ix < len(B["sems"][lv]) else None
        if t is not None and not g._pool_junk(t):
            ok.append(k)
    if not ok:
        return "(пусто: док без чистых гранул)", []
    S = np.array([B["sems"][members[k][0]][members[k][1]] for k in ok])
    kb = int(np.argmax(S @ qn))
    pos = ok[kb]
    # ── контекст окна: реальные зерна до входа ──
    hist = [members[k] for k in ok[:pos + 1]][-CL:]
    rows = [row(B, lv, ix) for lv, ix in hist]
    while len(rows) < CL:
        rows.insert(0, rows[0])
    # ── rollout: поле ведёт, память отдаёт зёрна ──
    spoken, out_parts, rows_ext = [], [], []
    seen_texts = set()
    for step in range(n_steps):
        states = torch.tensor(np.array(rows[-CL:]), dtype=torch.float32,
                              device=g.DEVICE).unsqueeze(0)
        with torch.no_grad():
            heads = B["model"](states, stream_idx=0,
                               cond=torch.zeros(1, g.FEAT_DIM, device=g.DEVICE),
                               qsem=torch.tensor(qn).unsqueeze(0).to(g.DEVICE))
        pred = F_norm(heads[5][0])
        # ЧЕРЕП-ПРОТОКОЛ: поле ведёт само (v17 учился на целых зёрнах и жёстких
        # негативах). Никаких магнитов/гейтов — только рабочая память.
        sims = B["M"] @ pred
        if B.get("ok_mask") is not None:   # память отвечает только целыми зёрнами
            sims[~B["ok_mask"]] = -1
        for ci in set(spoken):        # рабочая память: не повторяться
            sims[ci] = -1
        bi = int(np.argmax(sims))
        lv, ix = B["coords"][bi]
        txt = B["texts"][bi]
        th = hash(" ".join(str(txt).split())[:200])
        if th in seen_texts:          # междоковые дубли: тот же текст из другого дока
            continue
        seen_texts.add(th)
        spoken.append(bi)
        out_parts.append((txt, float(sims[bi]), (lv, ix)))
        rows.append(row(B, lv, ix))
    return assemble(out_parts), out_parts

def F_norm(t): return (t / (t.norm() + 1e-9)).cpu().numpy().astype(np.float32)

def assemble(parts):
    out = []
    for p, _, _ in parts:
        p = " ".join(p.split()).strip()
        if not p: continue
        p = re.sub(r"^[QA]:\s*", "", p)
        p = p[0].upper() + p[1:]
        if p[-1] not in ".!?…»:": p += "."
        out.append(p)
    return " ".join(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="0glm/checkpoints/brain_v0.pt")
    ap.add_argument("--ask", default=None)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()
    print(f"🧠 грузим мозг: {args.brain}")
    B = load_brain(args.__dict__["brain"])
    print(f"   память: {len(B['texts'])} гранул | {B['meta'].get('lineage','')}")
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    turns = []
    while True:
        Q = args.ask if args.ask else input("\n❓ ").strip()
        if Q.lower() in ("/exit", "/quit"): break
        if Q.lower() == "/new": turns.clear(); print("🧹"); continue
        if turns:
            print(f"🔗 сессия: {len(turns)} ход (память диалога v0 — без бленда)")
        text, tr = ask(B, Q, st, n_steps=args.steps)
        print("\n" + text + "\n")
        if tr:
            sims = [s for _, s, _ in tr]
            docs = sorted({B["inv"].get(c, -1) for _, _, c in tr})
            print(f"🎯 recall: mean cos={np.mean(sims):.3f} | зёрен {len(tr)} | "
                  f"доков {len([d for d in docs if d >= 0])}")
        turns.append(text)
        if args.ask: break


