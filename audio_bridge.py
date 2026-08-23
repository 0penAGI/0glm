"""
0GLM Этап 5-v0: z0-мост текст → аудио БЕЗ переобучения.

Цепочка: feat_head(z) уже предсказывает Δстилометрию → text_params (8 слотов,
семантика текста) → словарь M → target audio params (8 слотов, семантика звука)
→ retrieval ближайшего зерна из пула 0mge по его ВХОДНОМУ transition-параметру.

Важно: индексируем переходы, не абсолютные свойства зерна — ищем «переход похожего
характера». Diversity-check в манифесте: если уникальных исходников < 20% — у моста
нет смысловой вариативности (корпус слишком узкий по жанру).

Формула аудио-параметров зеркалит 0mge/granular_field.py::extract_params_from_feats
локально (там librosa в импортах — тащить не хотим).
FEAT_DIM аудио = 22:
  0 mean_spectrum, 1 std_spectrum, 2 centroid, 3 bandwidth, 4 flatness, 5 rolloff,
  6-11 mel_band[0..5], 12 mean_frame_energy, 13 std_frame_energy, 14 peak_frame_energy,
  15 std_diff_frame_energy, 16 flux_pos, 17 flux_abs, 18 ratio_lo, 19 ratio_mid,
  20 ratio_hi, 21 low_freq_ratio
"""
import argparse
import datetime
import hashlib
import json
import os

import numpy as np

import granular_text_field as g

MGE_POOL = str(g.GLM_DIR.parent / "0mge" / "granular_pool_lite.npz")
AUDIO_INDEX_CACHE = str(g.GLM_DIR / "pool" / "audio_index_v1.npz")

# имена для читаемых манифестов
TEXT_SLOTS = ["heat", "compression", "intensity", "cohesion", "novelty", "rhythm", "echo", "reserv"]
AUDIO_SLOTS = ["pitch", "stretch", "amp", "pan", "density", "pos_off", "reverse", "reserv"]


def extract_audio_params(feat_prev, feat_next):
    """Дословное зеркало 0mge::extract_params_from_feats."""
    p = np.zeros(8, dtype=np.float32)
    d = feat_next - feat_prev
    p[0] = float(np.clip(d[2] * 12, -1, 1))                              # pitch ← centroid
    p[1] = float(np.clip(d[15] * 10, -1, 1))                             # stretch ← choppiness
    p[2] = float(np.clip(d[0] / (abs(feat_prev[0]) + 1e-6), -1, 1))      # amp ← energy ratio
    p[3] = float(np.clip(d[19] * 5, -1, 1))                              # pan ← mid-balance
    p[4] = float(np.clip(d[17] * 10, -1, 1))                             # density ← flux
    p[5] = float(np.clip(d[5] * 3, -1, 1))                               # pos_off ← rolloff
    p[6] = float(1.0 if d[0] < -0.01 else -1.0)                          # reverse (бинарный)
    return p


def build_audio_index(force=False):
    """Индекс входных transition-параметров всех зёрен пула 0mge.
    Кэш: pool/audio_index_v1.npz (+sources в jsonl рядом)."""
    if not force and os.path.exists(AUDIO_INDEX_CACHE):
        z = np.load(AUDIO_INDEX_CACHE)
        metas = [json.loads(l) for l in open(AUDIO_INDEX_CACHE.replace(".npz", ".jsonl"))]
        print(f"🎛️ Audio index cache: {len(metas)} зёрен")
        return z["params"], metas
    z = np.load(MGE_POOL, allow_pickle=True)
    feats = {0: z["micro_feats"], 1: z["meso_feats"], 2: z["macro_feats"]}
    srcs = {0: z["micro_sources"], 1: z["meso_sources"], 2: z["macro_sources"]}
    P, metas = [], []
    for traj in z["trajectories"]:
        for k in range(1, len(traj)):
            pl, pi = int(traj[k - 1][0]), int(traj[k - 1][1])
            tl, ti = int(traj[k][0]), int(traj[k][1])
            s = srcs[tl]
            rec = s[min(ti, len(s) - 1)]
            if isinstance(rec, np.str_) or isinstance(rec, str):
                path = str(rec)
            elif hasattr(rec, "__len__"):
                path = str(rec[0])          # ndarray/list вида [path, offset]
            else:
                path = str(rec)
            path = path.strip().strip("[]'\"").split("\n")[0].strip()
            P.append(extract_audio_params(feats[pl][pi], feats[tl][ti]))
            metas.append({"level": tl, "idx": ti, "src": str(path)})
    P = np.array(P, dtype=np.float32)
    np.savez(AUDIO_INDEX_CACHE, params=P)
    with open(AUDIO_INDEX_CACHE.replace(".npz", ".jsonl"), "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    print(f"🎛️ Audio index: {len(P)} переходов → {AUDIO_INDEX_CACHE}")
    return P, metas


def M_text_to_audio(tp):
    """Кросс-модальный словарь (зафиксирован в PLAN.md, Этап 5).
    text slots: heat compression intensity cohesion novelty rhythm echo reserv"""
    a = np.zeros(8, dtype=np.float32)
    heat, comp, inten, cohe, nov, rhythm, echo = tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], tp[6]
    a[0] = np.clip(comp, -1, 1)                          # pitch ← compression
    a[1] = np.clip(rhythm, -1, 1)                        # stretch ← rhythm
    a[2] = np.clip(0.7 * heat + 0.3 * inten, -1, 1)      # amp ← возбуждение
    a[3] = np.clip(cohe, -1, 1)                          # pan ← cohesion
    a[4] = np.clip(0.6 * heat + 0.8 * nov, -1, 1)        # density ← agitation+novelty
    a[5] = np.clip(-0.5 * comp + 0.3 * nov, -1, 1)       # pos_off
    a[6] = float(np.sign(echo) * np.sign(heat))          # reverse ← флип при повторе
    return a


class AudioRetriever:
    """Brute-force ближайших зёрен по 6 непрерывным слотам (reverse исключаем — бинарный)."""

    def __init__(self, params, metas):
        self.P = params[:, :6]
        self.metas = metas
        self.recent = []

    def query(self, target, exclude_last=True):
        t = np.asarray(target[:6], dtype=np.float32)
        D = np.linalg.norm(self.P - t, axis=1)
        order = np.argsort(D)
        for j in order[:64]:
            key = (self.metas[j]["level"], self.metas[j]["idx"])
            if exclude_last and key in self.recent[-3:]:
                continue
            self.recent.append(key)
            if len(self.recent) > 16: self.recent.pop(0)
            return j, float(D[j])
        j = int(order[0])
        return j, float(D[j])


def sonify_stream(primary_steps, retr):
    """narrative-стрим: соседние выбранные гранулы → text_params → M → retrieval."""
    out = []
    for i in range(1, len(primary_steps)):
        prev, cur = primary_steps[i - 1], primary_steps[i]
        tp = g.extract_params_from_feats(prev["feat"], cur["feat"])
        target = M_text_to_audio(tp)
        j, dist = retr.query(target)
        m = retr.metas[j]
        out.append({
            "step": i,
            "text_excerpt": cur["text"],
            "text_params": {k: round(float(v), 3) for k, v in zip(TEXT_SLOTS, tp)},
            "target_audio": {k: round(float(v), 3) for k, v in zip(AUDIO_SLOTS, target)},
            "grain": {"level": m["level"], "idx": m["idx"], "src": os.path.basename(str(m["src"]))},
            "dist": round(dist, 3),
        })
    return out


def _sonify_stream_z(steps_, retr, model, alpha=0.5, proj=None, aff=None):
    """z-мост (ДЕФОЛТ аудио ride с v19c): прогоняет навигатор ПО фактическим
    шагам ride, состояния как в generate_multi ([стиль|аффект|сем-PCA]).
    Направление перехода = бленд unit(Δpred) и unit(Δact) с весом α,
    амплитуда ВСЕГДА фактическая (feat_head регрессирует к среднему —
    бленд значениями душил громкость). Текстовый путь не трогает."""
    import torch
    dev = next(model.parameters()).device
    FD = g.FEAT_DIM
    was_training = model.training
    model.eval()
    out, ctx = [], []
    with torch.no_grad():
        for i in range(1, len(steps_)):
            prev, cur = steps_[i - 1], steps_[i]
            lv_p, ix_p = int(prev["level"]), int(prev["idx"])
            lv_c, ix_c = int(cur["level"]), int(cur["idx"])
            fp32 = np.asarray(prev["feat"], dtype=np.float32)
            fc32 = np.asarray(cur["feat"], dtype=np.float32)
            st_prev = [fp32]
            if aff is not None:
                st_prev.append(np.asarray(aff[lv_p][ix_p], dtype=np.float32))
            if proj is not None:
                st_prev.append(np.asarray(proj[lv_p][ix_p], dtype=np.float32))
            state_prev = np.concatenate(st_prev)
            ctx.append(state_prev)
            st_t = torch.tensor(np.array(ctx[-12:]), dtype=torch.float32,
                                device=dev).unsqueeze(0)
            res = model(st_t)
            zz = res[4]
            pred32 = np.clip(fp32 +
                             model.feat_head(zz).squeeze(0).cpu().numpy() * 3.0,
                             -10, 10)
            # v19b: z даёт НАПРАВЛЕНИЕ, масштаб — всегда фактический.
            # Бленд направлений + фактическая амплитуда дельты: громкость
            # не падает (feat_head регрессирует к среднему и душил |Δ|).
            d_p, d_a = pred32 - fp32, fc32 - fp32
            n_p = float(np.linalg.norm(d_p)) + 1e-6
            n_a = float(np.linalg.norm(d_a)) + 1e-6
            dir_mix = alpha * (d_p / n_p) + (1.0 - alpha) * (d_a / n_a)
            n_m = float(np.linalg.norm(dir_mix)) + 1e-6
            eff32 = (fp32 + dir_mix / n_m * n_a).astype(np.float32)
            tp = g.extract_params_from_feats(fp32, eff32)
            target = M_text_to_audio(tp)
            j, dist = retr.query(target)
            m = retr.metas[j]
            d_pred, d_act = pred32 - fp32, fc32 - fp32
            out.append({
                "step": i,
                "text_excerpt": cur["text"],
                "text_params": {k: round(float(v), 3) for k, v in zip(TEXT_SLOTS, tp)},
                "target_audio": {k: round(float(v), 3) for k, v in zip(AUDIO_SLOTS, target)},
                "grain": {"level": m["level"], "idx": m["idx"],
                          "src": os.path.basename(str(m["src"]))},
                "z_dir_cos": round(float(np.dot(
                    d_pred / (np.linalg.norm(d_pred) + 1e-6),
                    d_act / (np.linalg.norm(d_act) + 1e-6))), 3)})
    if was_training:
        model.train()
    return out


def diversity_report(manifest):
    srcs = [e["grain"]["src"] for e in manifest]
    uniq = set(srcs)
    frac = len(uniq) / max(1, len(srcs))
    from collections import Counter
    top = Counter(srcs).most_common(3)
    ok = frac >= 0.20
    print(f"\n🌈 DIVERSITY: {len(uniq)} уникальных исходников / {len(srcs)} шагов = {frac:.0%}"
          + ("  ✅ ок" if ok else "  ⚠️ <20% — у моста нет вариативности!"))
    print(f"   топ-исходники: {[(s, c) for s, c in top]}")
    return frac


# ══════════════════════════════════════════════════════════════
# WAV РЕНДЕР по манифесту (зеркало 0mge::synthesize, но зёрна берём
# из манифеста (level, idx), а не из их cluster_map)
# ══════════════════════════════════════════════════════════════
SR = 22050                       # константа 0mge
LEVEL_AMP = {0: 0.4, 1: 0.6, 2: 1.0}


def _load_chunk(path, offset, n_samps):
    y, fsr = __import__("soundfile").read(path, dtype="float32", always_2d=False)
    if y.ndim > 1: y = y.mean(axis=1)
    if fsr != SR:
        t = np.linspace(0, len(y) - 1, int(len(y) * SR / fsr))
        y = np.interp(t, np.arange(len(y)), y).astype(np.float32)
    off = min(int(offset), max(0, len(y) - 1))
    chunk = y[off:off + n_samps]
    if len(chunk) < n_samps:
        chunk = np.pad(chunk, (0, n_samps - len(chunk)))
    return chunk.astype(np.float32)


def render_manifest(manifest_path, wav_path=None, sec_per_step=0.5, jitter_sigma=0.0):
    import soundfile as sf
    import zlib
    man = json.load(open(manifest_path))["manifest"]
    z = np.load(MGE_POOL, allow_pickle=True)
    srcs = {0: z["micro_sources"], 1: z["meso_sources"], 2: z["macro_sources"]}

    step_n = int(SR * sec_per_step)
    lens = [int(srcs[e["grain"]["level"]][e["grain"]["idx"]][2]) for e in man]
    total = len(man) * step_n + max(lens) + SR // 2
    audio = np.zeros((2, total), dtype=np.float32)
    rng = np.random.default_rng(42)
    # джиттер M v1: логнормальный множитель amp на шаг; детерминирован по имени
    # файла руки (у A/B разные реализации), нормирован чтобы среднее усиление = 1
    rng_j = np.random.default_rng(zlib.crc32((wav_path or manifest_path).encode()))
    jit_norm = float(np.exp(jitter_sigma ** 2 / 2))

    for i, e in enumerate(man):
        lv, gi = e["grain"]["level"], e["grain"]["idx"]
        fp, off, ns = srcs[lv][gi]
        chunk = _load_chunk(str(fp), off, int(ns))
        p = e["target_audio"]
        rate = (2 ** (p["pitch"] / 12)) * (2 ** p["stretch"])   # слоты [-1,1]
        if abs(rate - 1) > 0.05 and 0.25 < rate < 4:
            tl = int(len(chunk) / rate)
            chunk = np.interp(np.linspace(0, len(chunk) - 1, tl),
                              np.arange(len(chunk)), chunk).astype(np.float32)
        if p.get("reverse", -1) > 0.5: chunk = chunk[::-1].copy()
        g_jit = float(rng_j.lognormal(0.0, jitter_sigma)) / jit_norm if jitter_sigma > 0 else 1.0
        amp = max(0.05, (p["amp"] + 1) / 2 * g_jit) * LEVEL_AMP[lv]
        pan = float(np.clip(p["pan"], -1, 1))
        gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        n_ov = int(np.clip(round((p["density"] + 1) / 2 * 6), 1, 6))
        pos_shift = int(p["pos_off"] * len(chunk))
        hop = max(1, step_n // 4)
        for o in range(n_ov):
            jit = rng.integers(-hop // 3, hop // 3) if o > 0 else 0
            pos = int(np.clip(i * step_n + pos_shift + jit, 0, total - len(chunk)))
            al = min(pos + len(chunk), total) - pos
            env = np.hanning(al) * amp * (0.7 ** o)
            audio[0, pos:pos + al] += chunk[:al] * env * gl
            audio[1, pos:pos + al] += chunk[:al] * env * gr

    peak = float(np.abs(audio).max())
    if peak > 0.99: audio *= 0.99 / peak
    nz = np.where(np.abs(audio).max(axis=0) > 1e-4)[0]          # трим тишины
    if len(nz):
        end = min(total, nz[-1] + SR // 4)
        audio = audio[:, :end]
    wav_path = wav_path or manifest_path.replace(".json", ".wav")
    sf.write(wav_path, audio.T, SR)
    print(f"🔊 WAV: {wav_path}  ({total/SR:.1f}s, {len(man)} зёрен, peak={peak:.2f})")
    return wav_path


def ab_pair(manifest_path, jitter_sigma=0.0, sec_per_step=0.5):
    """Слепой A/B M v1: зёрна и порядок одинаковы; у shuffled-руки target_audio
    перемешаны. ОБЕ руки получают ОДИНАКОВЫЙ по σ джиттер (разные реализации) —
    тестируется семантическая корреляция, а не живость."""
    from copy import deepcopy
    data = json.load(open(manifest_path))
    man = data["manifest"]
    rng = np.random.default_rng(datetime.datetime.now().microsecond)
    perm = rng.permutation(len(man))
    shuf = deepcopy(man)
    for i, j in enumerate(perm):
        shuf[i]["target_audio"] = man[j]["target_audio"]
    base = manifest_path.replace(".json", "")
    m1, m2 = base + "_v1.json", base + "_v2.json"
    for path, mm in [(m1, man), (m2, shuf)]:
        json.dump({"meta": data["meta"], "manifest": mm}, open(path, "w"), ensure_ascii=False)
    w1, w2 = render_manifest(m1, sec_per_step=sec_per_step, jitter_sigma=jitter_sigma), \
             render_manifest(m2, sec_per_step=sec_per_step, jitter_sigma=jitter_sigma)
    m_is_a = bool(rng.random() < 0.5)
    key = {"A": "M" if m_is_a else "shuffled",
           "B": "shuffled" if m_is_a else "M", "jitter_sigma": jitter_sigma}
    os.rename(w1 if m_is_a else w2, base + "_A.wav")
    os.rename(w2 if m_is_a else w1, base + "_B.wav")
    kpath = base + "_abkey.json"
    json.dump(key, open(kpath, "w"))
    rl = base + "_readalong.txt"
    if not os.path.exists(rl):
        with open(rl, "w") as f:
            for e in man:
                mm_, ss_ = divmod(int(e["step"] * sec_per_step), 60)
                f.write(f"[{mm_:02d}:{ss_:02d}] {e['text_excerpt']}\n")
    print(f"\n🎲 СЛЕПАЯ ПАРА (σ={jitter_sigma}, обе руки с джиттером):\n"
          f"  A: {base}_A.wav\n  B: {base}_B.wav\n  readalong: {rl}\n"
          f"  ключ (не подглядывать): {kpath}")
    return key


STICKY_CFG = {"sem_w": 0.9, "topic_w": 0.45, "floor": 0.0, "veto": False,
              "anchor": "last", "micro": True, "bend": 0.15, "release": 3,
              "sticky_bonus": 0.8, "sticky_gamma": 0.93}   # чемпион свипа


def _build_traj_lookup(pool):
    """(level, idx) → (doc_i, pos_j) — обратный индекс траекторий."""
    lookup = {}
    for ti, traj in enumerate(pool.get("trajectories", [])):
        for j, (lv, ix) in enumerate(traj):
            lookup[(int(lv), int(ix))] = (ti, j)
    return lookup


def _ride_answer(engine, pool, anchors, per_phase, n_steps, trace=None):
    """Езда по траекториям (v17): связность берём из РЕАЛЬНЫХ документов.
    Якорь дуги → точка входа в док → едем вперёд по микро-звеньям в исходном
    порядке. Мезо-окна берём только если они не дублируют уже взятый текст.
    (3) структурные зёрна (Q:/A:/заголовки) скипаются по ходу поездки.
    (2) гейт дрейфа: скользящее окно сем уходит от цели фазы — фаза обрывается.
    trace: список событий для интроспекции (--trace)."""
    lookup = _build_traj_lookup(pool)

    def _words(t):
        return set(g.WORD_RE.findall(t.lower()))

    def _trec(ev):
        if trace is not None:
            trace.append(ev)

    steps, seen_words = [], []
    used_docs = set()
    extra = 0               # v18c: недоиспользованный ход фаз

    def _overlap_ok(t):
        w = _words(t)
        if not w:
            return False
        for prev in seen_words[-6:]:
            inter = len(w & prev)
            if inter and inter / max(1, min(len(w), len(prev))) > 0.5:
                return False
        return True

    for a_i, anc in enumerate(anchors):
        coord = tuple(int(x) for x in anc["coord"])
        if coord not in lookup or len(steps) >= n_steps:
            continue
        ti, j = lookup[coord]
        used_docs.add(ti)
        traj = pool["trajectories"][ti]
        _trec({"ev": "phase_start", "phase": a_i + 1, "doc": ti,
               "entry_j": j, "runway": len(traj) - 1 - j})
        taken = 0
        budget = per_phase + extra      # v18c: недоеханные фазы отдают ход дальше
        slide = []          # скользящее окно сем взятых шагов
        init_cos = None
        exhausted = False
        while j < len(traj) and taken < budget and len(steps) < n_steps:
            lv, ix = int(traj[j][0]), int(traj[j][1])
            ln = ["micro", "meso", "macro"][lv]
            t = pool[f"{ln}_texts"][ix]
            j += 1
            if g._pool_junk(t):        # (3) заголовки/Q&A не входят в выдачу
                _trec({"ev": "skip", "phase": a_i + 1, "doc": ti,
                       "level": lv, "idx": ix, "why": "junk", "head": t[:50]})
                continue
            if lv == 2:                # макро дублирует контент параграфа
                continue
            if not _overlap_ok(t):
                _trec({"ev": "skip", "phase": a_i + 1, "doc": ti,
                       "level": lv, "idx": ix, "why": "overlap",
                       "head": t[:50]})
                continue
            sem = engine.sems[lv][ix]
            # (2) гейт дрейфа: окно последних 4 шагов против цели фазы
            if anc.get("sem") is not None:
                slide.append(sem)
                if len(slide) > 4:
                    slide.pop(0)
                if taken >= 3 and len(slide) == 4:
                    w = np.mean(slide, axis=0)
                    nw = float(np.linalg.norm(w))
                    c = float(np.dot(w / nw, anc["sem"])) if nw > 1e-6 else 0.0
                    if init_cos is None:
                        init_cos = c
                    floor = max(0.30, (init_cos or c) - 0.25)
                    if c < floor:
                        _trec({"ev": "drift_gate", "phase": a_i + 1,
                               "doc": ti, "taken": taken,
                               "cos": round(c, 3), "floor": round(floor, 3)})
                        print(f"   ⛵ дрейф фазы {a_i+1}: cos окна {c:.2f} < {floor:.2f} — прыжок к след. якорю")
                        break
            steps.append({
                "level": lv, "idx": ix, "text": t,
                "feat": pool[f"{ln}_feats"][ix],
                "sem": sem, "origin": "ride", "doc": ti})
            _trec({"ev": "step", "phase": a_i + 1, "doc": ti,
                   "level": lv, "idx": ix, "taken": taken,
                   "head": t[:60]})
            seen_words.append(_words(t))
            taken += 1
        if taken < budget:
            # траектория кончилась (не гейт): отдаём недоеянный ход следующим,
            # но не больше +2 — глубокий доезд тащит док в смежные темы
            exhausted = j >= len(traj)
            if exhausted:
                _trec({"ev": "phase_end", "phase": a_i + 1, "doc": ti,
                       "why": "trajectory_end", "taken": taken,
                       "carried": min(2, budget - taken - 1)})
                extra = min(2, extra + budget - taken - 1)
            else:
                _trec({"ev": "phase_end", "phase": a_i + 1, "doc": ti,
                       "why": "global_step_cap", "taken": taken})
    return steps


def _qa_boost(engine, pool, question, k=8, n_hits=48):
    """Ретрив релевантных гранул вопроса → буст-кластеры + сид-контекст
    (зеркало qa.cmd_ask, вынесено для моста). Буст/сид строятся по топ-k,
    полный список хитов возвращается для якорей дуги."""
    from qa import Retriever
    retr = Retriever(pool)
    hits = retr.topk(question, k=n_hits)
    print(f"\n🔍 Хиты вопроса:")
    for t, s, _fc in hits[:3]:
        print(f"   [{s:.3f}] {t[:80]}...")
    inv = {}
    for cid, members in engine.cluster_map.items():
        for ln, gi in members:
            inv[(ln, gi)] = cid
    n_micro = len(pool["micro_texts"]); n_meso = len(pool["meso_texts"])
    def _flat2coord(f):
        if f < n_micro: return (0, f)
        if f < n_micro + n_meso: return (1, f - n_micro)
        return (2, f - n_micro - n_meso)
    boosted = set()
    hit_feats = []
    for t, s, fc in hits[:k]:
        ln_c, gi_c = _flat2coord(fc)
        f = pool[["micro_feats", "meso_feats", "macro_feats"][ln_c]][gi_c]
        hit_feats.append(f)
        cid = inv.get((ln_c, gi_c))          # кластер самого зерна — точный
        if cid is not None: boosted.add(cid)
    all_f = np.concatenate([pool["micro_feats"], pool["meso_feats"], pool["macro_feats"]])
    ctx_seed = np.array(hit_feats[:g.CONTEXT_LEN], dtype=np.float32)
    if len(ctx_seed) < g.CONTEXT_LEN:
        pad = all_f[np.random.choice(len(all_f), g.CONTEXT_LEN - len(ctx_seed), replace=False)]
        ctx_seed = np.concatenate([ctx_seed, pad], axis=0)
    q_feat = g.extract_feat_from_text(question)
    return boosted, ctx_seed, (q_feat if q_feat is not None else np.zeros(g.FEAT_DIM, dtype=np.float32)), \
        [(t, s, _flat2coord(fc)) for t, s, fc in hits]


def sensitivity_probe(retr):
    """Три синтетических настроения → должны попадать в разные части индекса."""
    probes = {
        "calm":   np.array([-.5, -.3, -.5, .3, -.5, -.2, 0, 0], dtype=np.float32),
        "excited": np.array([.8, .1, .9, -.2, .6, .5, 0, 0], dtype=np.float32),
        "chaotic": np.array([.9, -.7, .5, -.6, .9, .9, 0, 0], dtype=np.float32),
    }
    print("\n🎯 SENSITIVITY (одинаковый ли звук для разных настроений?):")
    cents = {}
    for name, tp in probes.items():
        tgt = M_text_to_audio(tp)
        # без exclusion: чистый отклик индекса
        t = tgt[:6]
        D = np.linalg.norm(retr.P - t, axis=1)
        j = int(np.argmin(D))
        cents[name] = retr.P[j]
        print(f"   {name:<8} → {retr.metas[j]['src'][:40]:<42} dist={D[j]:.3f}")
    d01 = float(np.linalg.norm(cents['calm'] - cents['excited']))
    d02 = float(np.linalg.norm(cents['calm'] - cents['chaotic']))
    print(f"   расхождение calm↔excited: {d01:.3f}, calm↔chaotic: {d02:.3f} (>0.3 = реагирует)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--render-manifest", type=str, default=None,
                    help="не генерить текст, только отрендерить WAV из готового json")
    ap.add_argument("--ab", type=str, default=None,
                    help="манифест → слепая пара A/B (шафл params, те же зёрна)")
    ap.add_argument("--ask", type=str, default=None,
                    help="вопрос → ответ текстом + WAV-озвучка (sticky-конфиг)")
    ap.add_argument("--q-w", type=float, default=0.6,
                    help="сила q-магнита (0 = выкл)")
    ap.add_argument("--arc", type=int, default=4,

                    help="дуга ответа: число фаз со своими якорями (0 = выкл)")
    ap.add_argument("--hub-w", type=float, default=1.0,
                    help="анти-хаб: штраф шаблонно-общим зёрнам (0 = выкл)")
    ap.add_argument("--ride", action="store_true",
                    help="езда по траекториям: связность из реальных документов (v16)")
    ap.add_argument("--trace", action="store_true",
                    help="интроспекция: как строился ответ (кандидаты, якоря, дрейф)")
    ap.add_argument("--z-audio", action="store_true",
                    help="дополнительно собрать слепую пару zab_*_A/B.wav "
                         "(дефолт vs feats-only) для прослушивания")
    ap.add_argument("--no-z-audio", action="store_true",
                    help="откат: аудио-переходы только из фактических feats")
    ap.add_argument("--z-alpha", type=float, default=0.5,
                    help="вес направления z в бленде (0=как без z)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="K шагов фазы поиска (буст 0.95, sticky выкл); 0 = без расписания")
    ap.add_argument("--adaptive", action="store_true",
                    help="адаптивный гейт: p_stay × cos(зерно, вопрос)")
    ap.add_argument("--sec-per-step", type=float, default=0.5)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="σ логнормального джиттера amp (M v1), 0 = выкл")
    args = ap.parse_args()

    if args.ab:
        ab_pair(args.ab, jitter_sigma=args.jitter, sec_per_step=args.sec_per_step)
        return

    A, metas = build_audio_index()
    retr = AudioRetriever(A, metas)

    pool = g.load_pool()
    with open(g.CLUSTERS_CACHE) as f: clusters = json.load(f)
    engine = g.TextGrainEngine(pool, clusters)
    engine.attach_sems(g.build_semantics(pool))
    # v19: дефолт — прод-поле v15; v11 остался только явным --model
    _v15 = os.path.join(os.path.dirname(str(g.MODEL_MULTI_CACHE)),
                        "text_navigator_v15_field.pt")
    mp = args.model or (_v15 if os.path.exists(_v15) else g.MODEL_MULTI_CACHE)
    if not os.path.exists(mp):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), mp)
        if os.path.exists(alt): mp = alt
    model, sem_proj, aff_ctx, pos_ctx = _load_navigator(mp)

    if args.ask:
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        qemb = _st.encode([args.ask])[0]
        qn = (qemb / (np.linalg.norm(qemb) + 1e-9)).astype(np.float32)
        boost, ctx_seed, q_feat, hit_list = _qa_boost(engine, pool, args.ask)
        hit_texts = [t for t, _, _ in hit_list]
        hit_coords = [c for _, _, c in hit_list]
        cfg = dict(STICKY_CFG)
        if args.warmup > 0:
            cfg["sticky_warmup"] = args.warmup
            cfg["boost_p_warmup"] = 0.95
        if aff_ctx is not None:
            # настроение вопроса → аффективный якорь генерации (v7):
            # эмоциональная окраска ответа следует тону запроса
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                _sc = SentimentIntensityAnalyzer().polarity_scores(args.ask)
                _qv = float(np.clip(_sc["compound"], -1, 1))
                _qa_ = float(np.clip(_sc["pos"] + _sc["neg"]
                                     + (args.ask.count("!") + args.ask.count("?")) * 0.08, 0, 1))
            except Exception:
                _qv, _qa_ = 0.0, 0.0
            cfg["aff_w"] = 1.5
            cfg["mood"] = np.array([_qv, _qa_], dtype=np.float32)
            print(f"🎭 mood вопроса: valence={_qv:+.2f} arousal={_qa_:.2f}")
        if args.q_w > 0:
            # q-магнит (v11): вопрос тянет выборку зёрен всё время генерации
            cfg["q_ref"] = qn
            cfg["q_w"] = args.q_w
            print(f"🧲 q-магнит: w={args.q_w}")
        if args.hub_w > 0:
            cfg["hub_w"] = float(args.hub_w)
            cfg["no_cite"] = True
            print(f"🚫 анти-хаб: w={args.hub_w} + анти-цитаты")
        trace_arc, trace_ev = None, None
        if args.arc > 0 and hit_texts:
            # дуга ответа v18: первый этап — ВСЕ 12k доков (кэш doc_embs),
            # не 6%-сэмпл зернового ретрива. Топ-D дока → лучшее зерно входа
            # → хвостовая оценка (куда ride повезёт) → MMR по хвостам.
            lookup = _build_traj_lookup(pool)
            Dembs = g.build_doc_embs(pool, engine.sems)
            dsims = Dembs @ qn
            top_docs = [int(d) for d in np.argsort(-dsims)[:16]]
            print(f"🚪 док-поиск: топ-{len(top_docs)} из {len(Dembs)} доков, "
                  f"sim .{int(dsims[top_docs[0]]*1000)}/1000")

            def _tail_vec(coord, k=10):
                """Среднее сем следующих k чистых микро/мезо звеньев траектории."""
                if coord not in lookup:
                    return None
                ti, j = lookup[coord]
                traj = pool["trajectories"][ti]
                vs = []
                while j < len(traj) and len(vs) < k:
                    lv, ix = int(traj[j][0]), int(traj[j][1])
                    if lv <= 1:
                        t = pool[f"{['micro', 'meso', 'macro'][lv]}_texts"][ix]
                        if not g._pool_junk(t):
                            vs.append(engine.sems[lv][ix])
                    j += 1
                if not vs:
                    return None
                v = np.mean(vs, axis=0)
                n = float(np.linalg.norm(v))
                return (v / n).astype(np.float32) if n > 1e-6 else None

            # зерно входа в каждом топ-доке: лучший чистый гран по sim к вопросу
            cand_coords, cand_texts, tails, trace_cands = [], [], [], []
            for d in top_docs:
                members = [(int(lv), int(ix)) for lv, ix in pool["trajectories"][d]
                           if int(lv) <= 1]
                ok = [k for k, (lv, ix) in enumerate(members)
                      if not g._pool_junk(pool[f"{['micro', 'meso'][lv]}_texts"][ix])]
                if not ok:
                    continue
                S = np.array([engine.sems[members[k][0]][members[k][1]] for k in ok])
                # v18c: вход = ОСТРЫЙ семантический матч; короткий runway
                # компенсируется переносом бюджета следующей фазе (extra)
                kb = int(np.argmax(S @ qn))
                lv, ix = members[ok[kb]]
                coord = (lv, ix)
                tv = _tail_vec(coord)
                if tv is None:
                    continue
                cand_coords.append(coord)
                cand_texts.append(pool[f"{['micro', 'meso'][lv]}_texts"][ix])
                tails.append(tv)
                trace_cands.append({"doc": d, "dsim": round(float(dsims[d]), 3),
                                    "entry": pool[f"{['micro', 'meso'][lv]}_texts"][ix][:70]})
            T = np.array(tails) if tails else np.zeros((1, 384), np.float32)
            trels = T @ qn
            order = list(np.argsort(-trels))
            anchors = [{"sem": T[order[0]], "coord": cand_coords[order[0]]}]
            fam0 = T[order[0]]
            while len(anchors) < args.arc:
                bi, bv = None, -1e9
                for ii, i in enumerate(order):
                    if float(trels[i]) < 0.32: continue   # слабо релевантный хвост ≠ фаза
                    if float(T[i] @ T[order[0]]) < 0.30: continue  # чужая семья: якорь №1 задаёт тему
                    mxs = max(float(T[i] @ A_["sem"]) for A_ in anchors)
                    if mxs > 0.85: continue               # уже сказано
                    v = float(trels[i]) - 0.7 * mxs
                    if v > bv: bv, bi = v, ii
                if bi is None:
                    break                                 # нового релевантного нет
                anchors.append({"sem": T[order[bi]],
                                "coord": cand_coords[order[bi]]})
            # интроспекция: причины по каждому кандидату
            chosen = {tuple(int(x) for x in A_["coord"]) for A_ in anchors}
            for ri, row in enumerate(trace_cands):
                row["trel"] = round(float(trels[ri]), 3)
                row["fam"] = round(float(T[ri] @ fam0), 3)
                row["chosen"] = cand_coords[ri] in chosen
                if not row["chosen"]:
                    row["why_not"] = ("trel<.32" if trels[ri] < 0.32 else
                                      "чужая семья" if row["fam"] < 0.30 else
                                      "дубль фазы (mxs>.85)" if
                                      max(float(T[ri] @ A_["sem"]) for A_ in anchors) > 0.85
                                      else "не выиграл MMR")
            trace_arc = {"top_docs": len(top_docs), "candidates": trace_cands,
                         "n_anchors": len(anchors)}
            if len(anchors) >= 2:
                per = max(4, int(round(args.steps / len(anchors))))
                cfg["arc"] = {"anchors": anchors, "per_phase": per}
                print(f"🎬 дуга: {len(anchors)} фаз × {per} шагов")
            else:
                print("🎬 дуга выкл: меньше 2 релевантных якорей")
        if args.ride:
            # v16: связность из траекторий документов. Якоря дуги = точки входа,
            # внутри фазы едем по реальному тексту. Навигатор и магниты отдыхают.
            arc_cfg = cfg.get("arc") or {}
            anchors = arc_cfg.get("anchors") or (
                [{"coord": hit_coords[0]}] if hit_coords else [])
            trace_ev = [] if args.trace else None
            rsteps = _ride_answer(engine, pool, anchors,
                                  arc_cfg.get("per_phase",
                                              max(4, args.steps // 4)),
                                  args.steps, trace=trace_ev)
            print(f"🛤️ ride: {len(rsteps)} зёрен из {len({s['doc'] for s in rsteps})} доков")
            steps = [rsteps]
            text = g.stitch_narrative(rsteps, engine)
            z_arr = np.zeros((max(1, len(rsteps)), 64), dtype=np.float32)
        else:
            text, z_arr, steps = g.generate_multi(
                model, engine, pool, n_steps=args.steps, seed=args.seed,
                temp=args.temperature, target_stats=q_feat,
                ctx_init=ctx_seed, boost_clusters=boost, sem_cfg=cfg,
                ref_sem=(qn if args.adaptive else None),
                sem_ctx=sem_proj, affect_ctx=aff_ctx, pos_ctx=pos_ctx)
        manifest = sonify_stream(steps[0], retr)
        cs = [float(np.dot(engine.sems[s["level"]][s["idx"]], qn)) for s in steps[0]]
        print(f"🎯 relevance: mean cos={np.mean(cs):.3f} max={np.max(cs):.3f}")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(g.OUT, f"sonify_qa_{ts}.json")
        with open(out, "w") as f:
            json.dump({"meta": {"question": args.ask, "steps": args.steps,
                                "seed": args.seed,
                                "mode": "ride" if args.ride else "attract",
                                "cfg": ("ride: doc-first+arc+drift-gate" if args.ride
                                        else "U_micro+sticky(.8,.93)"),
                                "model": os.path.basename(mp),
                                "model_used_in_mode": not args.ride,
                                "full_text": text},
                       "manifest": manifest}, f, ensure_ascii=False, indent=1)
        # v19c: z-мост — ДЕФОЛТ аудио-переходов ride (слепая серия за z:
        # 2 одиночных + батарея 5). Направление от feat_head, масштаб
        # фактический. Текст/трейс не меняются; --no-z-audio = feats-only.
        if args.ride and model is not None and not args.no_z_audio:
            man_z = _sonify_stream_z(steps[0], retr, model, alpha=args.z_alpha,
                                     proj=sem_proj, aff=aff_ctx)
            with open(out) as f:
                mm = json.load(f)
            for so, sz in zip(mm["manifest"], man_z):
                so["dist_plain"] = so.get("dist")
                for k in ("text_params", "target_audio", "grain", "z_dir_cos"):
                    so[k] = sz[k]
                so.pop("dist", None)
            mm["meta"]["audio_variant"] = f"z-dir alpha={args.z_alpha}"
            with open(out, "w") as f:
                json.dump(mm, f, ensure_ascii=False, indent=1)
            print(f"🌊 аудио: z-dir α={args.z_alpha} (дефолт; --no-z-audio откат)")
        rl = out.replace(".json", "_readalong.txt")
        with open(rl, "w") as f:
            for e in manifest:
                mm, ss = divmod(int(e["step"] * args.sec_per_step), 60)
                f.write(f"[{mm:02d}:{ss:02d}] {e['text_excerpt']}\n")
        _a = g.stitch_narrative(steps[0], engine)
        if len(_a) > 1200:   # не режем на полуслове
            _cut = _a.rfind(" ", 1000, 1200)
            _a = _a[:_cut if _cut > 0 else 1200] + " …"
        print(f"\n{'─'*60}\nQ: {args.ask}\nA: {_a}\n{'─'*60}")
        print(f"💾 {out}\n📖 {rl}")
        if args.trace:
            tr = {"question": args.ask, "arc": trace_arc,
                  "events": trace_ev or []}
            tp = out.replace(".json", "_trace.json")
            with open(tp, "w") as f:
                json.dump(tr, f, ensure_ascii=False, indent=1)
            print("\n🔍 КАК СТРОИЛСЯ ОТВЕТ")
            print(f"  док-этап: топ-{(trace_arc or {}).get('top_docs')} доков")
            for c in (trace_arc or {}).get("candidates", []):
                mark = "✅ ЯКОРЬ" if c.get("chosen") else f"   ({c.get('why_not')})"
                print(f"  #{c['doc']:>5} dsim={c['dsim']:.3f} "
                      f"tail={c['trel']:.3f} fam={c['fam']:.3f} "
                      f"{mark} «{c['entry']}»")
            ph = None
            for ev in trace_ev or []:
                if ev["ev"] == "phase_start":
                    ph = ev["phase"]
                    print(f"  ▶ фаза {ph}: вход в док #{ev['doc']} (зерно {ev['entry_j']})")
                elif ev["ev"] == "drift_gate":
                    print(f"    ⛵ дрейф: окно cos {ev['cos']} < пол {ev['floor']} "
                          f"после {ev['taken']} шагов — смена фазы")
                elif ev["ev"] == "skip":
                    print(f"    ✗ {'µσΩ'[ev['level']]}{ev['idx']} "
                          f"[{ev['why']}] «{ev['head']}…»")
                elif ev["ev"] == "step":
                    print(f"    · док#{ev['doc']} {'µσ'[ev['level']]}{ev['idx']} "
                          f"«{ev['head']}…»")
            print(f"🔍 трейс: {tp}")
        render_manifest(out, sec_per_step=args.sec_per_step)
        if args.z_audio:
            import zlib
            # слепая пара: текущий дефолт против feats-only, стороны случайны
            with open(out) as f:
                cur_mm = json.load(f)
            lbl_cur = cur_mm["meta"].get("audio_variant", "feats-only")
            man_alt = sonify_stream(steps[0], retr)
            alt_mm = {"meta": dict(cur_mm["meta"], audio_variant="feats-only"),
                      "manifest": [dict(m, **{k: p[k] for k in
                                              ("text_params", "target_audio",
                                               "grain")})
                                   for m, p in zip(cur_mm["manifest"], man_alt)]}
            out_alt = out.replace(".json", "_alt.json")
            with open(out_alt, "w") as f:
                json.dump(alt_mm, f, ensure_ascii=False, indent=1)
            w_cur = render_manifest(out, sec_per_step=args.sec_per_step)
            w_alt = render_manifest(out_alt, sec_per_step=args.sec_per_step)
            rng = np.random.default_rng(zlib.crc32(out.encode()))
            a_is_cur = bool(rng.integers(0, 2))
            base = os.path.join(g.OUT, f"zab_{ts}")
            os.rename(w_cur, base + ("_A.wav" if a_is_cur else "_B.wav"))
            os.rename(w_alt, base + ("_B.wav" if a_is_cur else "_A.wav"))
            key = {"A": lbl_cur if a_is_cur else "feats-only",
                   "B": "feats-only" if a_is_cur else lbl_cur}
            with open(base + "_key.json", "w") as f:
                json.dump(key, f, indent=1)
            print(f"\n🎧 Z-A/B: {base}_A.wav | {base}_B.wav")
            print("   послушай вслепую, потом загляни в _key.json")
        return

    if args.render_manifest:
        render_manifest(args.render_manifest, sec_per_step=args.sec_per_step)
        return

    A, metas = build_audio_index()
    retr = AudioRetriever(A, metas)

    pool = g.load_pool()
    with open(g.CLUSTERS_CACHE) as f: clusters = json.load(f)
    engine = g.TextGrainEngine(pool, clusters)
    engine.attach_sems(g.build_semantics(pool))
    # v19: дефолт — прод-поле v15; v11 остался только явным --model
    _v15 = os.path.join(os.path.dirname(str(g.MODEL_MULTI_CACHE)),
                        "text_navigator_v15_field.pt")
    mp = args.model or (_v15 if os.path.exists(_v15) else g.MODEL_MULTI_CACHE)
    if not os.path.exists(mp):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), mp)
        if os.path.exists(alt): mp = alt
    model, sem_proj, aff_ctx, pos_ctx = _load_navigator(mp)

    cfg = {"sem_w": 0.9, "topic_w": 0.45, "floor": 0.0, "veto": False,
           "anchor": "last", "micro": True}
    text, z_arr, steps = g.generate_multi(model, engine, pool,
                                          n_steps=args.steps, seed=args.seed,
                                          temp=args.temperature, sem_cfg=cfg,
                                          sem_ctx=sem_proj, affect_ctx=aff_ctx, pos_ctx=pos_ctx)
    manifest = sonify_stream(steps[0], retr)
    diversity_report(manifest)
    sensitivity_probe(retr)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(g.OUT, f"sonify_{ts}.json")
    with open(out, "w") as f:
        json.dump({"meta": {"steps": args.steps, "seed": args.seed,
                            "model": os.path.basename(mp),
                            "bridge": "M_text_to_audio v0",
                            "full_text": text},
                   "manifest": manifest}, f, ensure_ascii=False, indent=1)
    print(f"\n💾 {out}")
    rl = out.replace(".json", "_readalong.txt")
    with open(rl, "w") as f:
        for e in manifest:
            mm, ss = divmod(int(e["step"] * args.sec_per_step), 60)
            f.write(f"[{mm:02d}:{ss:02d}] {e['text_excerpt']}\n")
    print(f"📖 readalong: {rl}")
    render_manifest(out, sec_per_step=args.sec_per_step)
    for e in manifest[:4]:
        print(f"  [{e['step']:>2}] «{e['text_excerpt'][:60]}…» → "
              f"{e['grain']['src']} (dist={e['dist']})")


def _load_navigator(mp):
    """Загрузка навигатора с автоопределением ширины входа:
    feat_dim=FEAT_DIM → v1/v3/v4; FEAT_DIM+SEM_PROJ_DIM → v5_semin;
    +AFFECT_DIM → v6_affect ([стиль|аффект|сем-PCA]);
    +1 → v12_pos ([стиль|аффект|сем-PCA|позиция в доке]). Возвращает блоки
    контекста, соответствующие ширине."""
    import torch as _t
    sd = _t.load(mp, map_location="cpu", weights_only=False)
    fd = int(sd.get("feat_dim", g.FEAT_DIM)) if isinstance(sd, dict) else g.FEAT_DIM
    model = g.MultiNavigator(feat_dim=fd).to(g.DEVICE)
    g.safe_load(model, mp)
    model.eval()
    proj, aff, pos = None, None, None
    if fd != g.FEAT_DIM:
        spl = g.load_sem_projection()
        if spl is None:
            raise RuntimeError("модель %d-dim требует pool/sem_proj_v1.npz" % fd)
        proj = g.project_sems(g.build_semantics(g.load_pool()), *spl)
        base = g.FEAT_DIM + g.AFFECT_DIM + g.SEM_PROJ_DIM
        if fd in (base, base + 1):
            aff = g.load_affect()
        if fd == base + 1:
            pos = g.build_positions(g.load_pool())
            print("📍 pos_ctx активен (v12)")
        print(f"🧭 navigator input {fd}-dim (стиль"
              f"{'+' + str(g.AFFECT_DIM) + ' аффект' if aff is not None else ''}"
              f"+{g.SEM_PROJ_DIM} сем{'+' if pos else ''}{'1 поз' if pos else ''}), sem_ctx активен")
    return model, proj, aff, pos


if __name__ == "__main__":
    main()
