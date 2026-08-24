"""Генератор демонстраций для дистилляции ride в поле (v20).
Учитель = ride v20b (слепая серия 5:0). Логируем каждую тройку
(контекст-нарастающим итогом ЭПИЗОДА, вопрос хода, выбранное зерно)
включая шаги после гейтов — поле впервые увидит испорченные контексты
с правильным продолжением.
Диалоги: эпизоды 1-4 ходов (уточнение/развитие/переключение темы).
Холдаут: 15% кластеров не участвуют в демо — чистый замер генерализации.
Блок дуги СКОПИРОВАН из audio_bridge.main() (v20b) — при изменении там
синхронизировать здесь. TODO(refactor): вынести в общую функцию."""
import sys, os, json, re, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import granular_text_field as g
import audio_bridge as ab

POOL_SEED = 20260825
rng = np.random.default_rng(POOL_SEED)

pool = g.load_pool()
clusters = json.load(open(g.CLUSTERS_CACHE))
canon = g.canonical_mask(pool)
engine = g.TextGrainEngine(pool, clusters)
engine.attach_sems(g.build_semantics(pool))
Dembs = g.build_doc_embs(pool, engine.sems)
sp = g.load_sem_projection()
sems_plain = g.build_semantics(pool)
_pj = g.project_sems(sems_plain, *sp)
proj_all = np.concatenate([_pj[0], _pj[1], _pj[2]], axis=0)
affect = g.load_affect()
n_mic, n_mes = len(pool["micro_texts"]), len(pool["meso_texts"])
lookup = ab._build_traj_lookup(pool)

# ── словарь фона для tf-idf терминов ──
STOP = set("the a an of to in on for and or is are was were be been it its this that these those with as at by from about into over after before more most than then when what how why who which do does did can could will would should their there they them he she his her you your we our us i not no yes if but so such very much many few also may might must shall".split())
print("📚 фон терминов...")
bg = collections.Counter()
for t in rng.choice(np.array(pool["micro_texts"]), size=min(20000, n_mic), replace=False):
    for w in g.WORD_RE.findall(str(t).lower()):
        if w not in STOP and len(w) > 3:
            bg[w] += 1

def terms_of(text, k=3):
    cnt = collections.Counter(w for w in g.WORD_RE.findall(text.lower())
                              if w not in STOP and len(w) > 3)
    scored = [(c * np.log(1 + len(bg) / (1 + bg[w])), w) for w, c in cnt.items() if c >= 2]
    scored.sort(reverse=True)
    return [w for _, w in scored[:k]] or [w for w, _ in cnt.most_common(k)]

def cap_first(s): return s[0].upper() + s[1:] if s else s

# ── сиды вопросов: макро-зёрна канонических доков вне холдаута ──
def holdout(cid): return (hash(("hold", int(cid))) % 100) < 15
T1 = ["What is {}?", "How does {} work?", "Why is {} important?",
      "What causes {}?", "How does {} affect people?", "What role does {} play?"]
TF = ["What about {}?", "And how does {} fit into this?", "Tell me about {}.",
      "Why does {} matter here?", "How is {} connected to this?"]

macro_docs = collections.defaultdict(list)
for d, traj in enumerate(pool["trajectories"]):
    for lv, ix in traj:
        if int(lv) == 2 and int(ix) < len(pool["macro_texts"]) and canon[2][int(ix)]:
            cid = int(clusters["macro"][min(int(ix), len(clusters["macro"]) - 1)])
            if not holdout(cid):
                macro_docs[d].append(int(ix))
            break

seeds = []
for d, macros in sorted(macro_docs.items()):
    if len(seeds) >= 1200:
        break
    ix = macros[0]
    ts = terms_of(str(pool["macro_texts"][ix]))
    if not ts:
        continue
    seeds.append(dict(doc=d, x=ts[0], y=(ts[1] if len(ts) > 1 else ts[0]),
                      z=(ts[-1] if len(ts) > 2 else ts[0])))
print(f"🎯 сидов: {len(seeds)} (холдаут-кластеры исключены)")

st = None
def encode(qs):
    global st
    if st is None:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return st.encode(qs, show_progress_bar=False, convert_to_numpy=True)

NAVJUNK = re.compile(r"^(see \w+,? for a|read more at:|for a (complete|full) list"
                     r"|click here|subscribe |follow us on|share this article)", re.I)
ARC_JUNK = lambda t: g._pool_junk(t) or bool(NAVJUNK.match(t.strip()))

def tail_vec(coord, k=10):
    if coord not in lookup:
        return None
    ti, j = lookup[coord]
    traj = pool["trajectories"][ti]
    vs = []
    while j < len(traj) and len(vs) < k:
        lv, ix = int(traj[j][0]), int(traj[j][1])
        if lv <= 1:
            t = pool[f"{['micro','meso'][lv]}_texts"][ix]
            if not ARC_JUNK(t):
                vs.append(engine.sems[lv][ix])
        j += 1
    if not vs:
        return None
    v = np.mean(vs, axis=0); n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-6 else None

def select_ride(qn, visited):
    """копия блока дуги main() v20b (док-первый + MMR + анти-ревизит) + _ride_answer"""
    dsims = Dembs @ qn
    ranked = [int(d) for d in np.argsort(-dsims)[:64] if int(d) not in visited]
    top_docs = ranked[:16]
    if not top_docs:
        return [], None, None
    cand_coords, tails, cand_docs = [], [], []
    for d in top_docs:
        members = [(int(lv), int(ix)) for lv, ix in pool["trajectories"][d] if int(lv) <= 1]
        ok = [k for k, (lv, ix) in enumerate(members)
              if not ARC_JUNK(pool[f"{['micro','meso'][lv]}_texts"][ix])]
        if not ok:
            continue
        S = np.array([engine.sems[members[k][0]][members[k][1]] for k in ok])
        kb = int(np.argmax(S @ qn))
        lv, ix = members[ok[kb]]
        tv = tail_vec((lv, ix))
        if tv is None:
            continue
        cand_coords.append((lv, ix)); tails.append(tv); cand_docs.append(d)
    if not cand_coords:
        return [], None, None
    T = np.array(tails)
    trels = T @ qn
    order = list(np.argsort(-trels))
    anchors = [{"sem": T[order[0]], "coord": cand_coords[order[0]]}]
    while len(anchors) < 4:
        bi, bv = None, -1e9
        for ii, i in enumerate(order):
            if float(trels[i]) < 0.32: continue
            if float(T[i] @ T[order[0]]) < 0.30: continue
            mxs = max(float(T[i] @ A["sem"]) for A in anchors)
            if mxs > 0.85: continue
            v = float(trels[i]) - 0.7 * mxs
            if v > bv: bv, bi = v, ii
        if bi is None or bv < 0.08:
            break
        anchors.append({"sem": T[order[bi]], "coord": cand_coords[order[bi]]})
    for A in anchors:
        visited.add(cand_docs[cand_coords.index(A["coord"])])
    rsteps = ab._ride_answer(engine, pool, anchors,
                             max(4, 24 // max(1, len(anchors))), 24)
    return rsteps, anchors, (cand_coords[order[0]], cand_docs[order[0]])

def row_of(lv, ix):
    gi = g._level_to_all_f_idx(lv, ix, n_mic, n_mes)
    return np.concatenate([np.concatenate([pool["micro_feats"], pool["meso_feats"],
                                           pool["macro_feats"]])[gi],
                           affect[lv][ix], proj_all[gi]])

out_path = str(g.GLM_DIR / "pool" / "demos_v2.jsonl")
ALL_F = np.concatenate([pool["micro_feats"], pool["meso_feats"], pool["macro_feats"]])
fout = open(out_path, "w")
n_triples, n_turns, n_rev = 0, 0, 0
for si, sd in enumerate(seeds):
    sess_turns, visited, ep_rows, prev = [], set(), [], None
    plan = [T1[si % len(T1)].format(cap_first(sd["x"]))]

    r3 = si % 6
    if r3 in (1, 4):
        plan.append(TF[si % len(TF)].format(cap_first(sd["y"])))
    elif r3 in (2, 5):
        plan += [TF[(si // 3) % len(TF)].format(cap_first(sd["y"])),
                 f"Now explain {sd['z']} instead."]
    for ti, Q in enumerate(plan):
        if sess_turns:
            _pq, _pa = sess_turns[-1]
            qe = encode([Q])[0]; cv = encode([_pa[:400]])[0]
            mix = 0.7 * qe + 0.3 * cv
            qn = (mix / (np.linalg.norm(mix) + 1e-9)).astype(np.float32)
        else:
            qe = encode([Q])[0]
            qn = (qe / (np.linalg.norm(qe) + 1e-9)).astype(np.float32)
        before = len(visited)
        rsteps, anchors, entry = select_ride(qn, visited)
        if len(visited) > before:
            n_rev += 1
        if not rsteps:
            break
        n_turns += 1
        # старт эпизода: контекст = предыдущие реальные зёрна дока входа
        # (зеркало бутстрапа brain_chat: поле всегда видит реальное прошлое)
        if ti == 0 and not ep_rows and entry is not None:
            (lv0, ix0), d0 = entry
            if (lv0, ix0) in lookup:
                _, j0 = lookup[(lv0, ix0)]
                traj = pool["trajectories"][d0]
                pre = []
                jj = j0
                while jj >= 0 and len(pre) < 11:
                    plv, pix = int(traj[jj][0]), int(traj[jj][1])
                    if plv <= 1 and canon[plv][pix] and \
                            not ARC_JUNK(pool[f"{['micro','meso'][plv]}_texts"][pix]):
                        pre.append(row_of(plv, pix))
                    jj -= 1
                ep_rows[:0] = pre[::-1]
        text_parts = []
        for s in rsteps:
            lv, ix = s["level"], s["idx"]
            if lv > 1:
                continue
            tgt_gi = g._level_to_all_f_idx(lv, ix, n_mic, n_mes)
            p_lv, p_ix = (prev if prev else (lv, ix))
            p_gi = g._level_to_all_f_idx(p_lv, p_ix, n_mic, n_mes)
            ln = ["micro", "meso", "macro"][lv]
            pair = {
                "ctx": np.array(ep_rows[-11:], dtype=np.float32) if ep_rows
                       else np.zeros((11, 66), dtype=np.float32),
                "cluster": int(clusters[ln][min(ix, len(clusters[ln]) - 1)]),
                "level": lv,
                "params": g.extract_params_from_feats(ALL_F[p_gi], ALL_F[tgt_gi]),
                "density": float(np.clip(ALL_F[tgt_gi][20] / 40.0, 0, 1)),
                "delta_next": np.clip(ALL_F[tgt_gi] - ALL_F[p_gi], -3, 3).astype(np.float32) / 3.0,
                "cond": np.zeros(g.FEAT_DIM, dtype=np.float32),
                "affect_next": np.asarray(affect[lv][ix], dtype=np.float32),
                "sem_next": engine.sems[lv][ix].astype(np.float16),
                "qsem": qn.astype(np.float16),
                "src": "demo",
                "doc": int(s.get("doc", -1)), "gidx": int(ix),
                "ep": si, "turn": ti,
            }
            fout.write(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in pair.items()}, ensure_ascii=False) + "\n")
            n_triples += 1
            ep_rows.append(row_of(lv, ix))
            prev = (lv, ix)
            text_parts.append(s["text"])
        sess_turns.append((Q, " ".join(text_parts)))
    if (si + 1) % 50 == 0:
        print(f"  эпизод {si+1}/{len(seeds)}: троек {n_triples}, ходов {n_turns}")
fout.close()
print(f"✅ {n_triples} троек | {n_turns} ходов | {len(seeds)} эпизодов | "
      f"анти-ревизит сработал: {n_rev} → {out_path}")
