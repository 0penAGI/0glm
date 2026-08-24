"""BAKE v0 сборка: ОДИН файл-мозг = веса планировщика + вся память внутри.
Снаружи не нужны ни пул, ни кэши — только сам артефакт (+ энкодер вопроса,
он сенсор, не память). Провенанс сохранён: тексты гранул лежат внутри дословно."""
import sys, os, json, hashlib, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import granular_text_field as g

def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(n):
            h.update(chunk)
    return h.hexdigest()[:16]

t0 = time.time()
pool = g.load_pool()
sems = g.build_semantics(pool)
sp = g.load_sem_projection()
affect = g.load_affect()
with open(g.CLUSTERS_CACHE) as f:
    clusters = json.load(f)

FD = g.FEAT_DIM + g.AFFECT_DIM + g.SEM_PROJ_DIM
model = g.MultiNavigator(feat_dim=FD).to(g.DEVICE)
g.safe_load(model, str(g.GLM_DIR / "checkpoints" / "text_navigator_v21_planner.pt"))
model.eval()

doc_embs = g.build_doc_embs(pool, sems)

# маска ЦЕЛЫХ+КАНОНИЧЕСКИХ зёрен для припоминания: поле училось навигации
# среди целых зёрен чистого корпуса — память не подсовывает ему то,
# чего не было в задаче (дубли синдиката вычищены canonical_mask)
canon = g.canonical_mask(pool)
texts3 = [pool["micro_texts"], pool["meso_texts"], pool["macro_texts"]]
ok_mask = np.zeros(len(pool["micro_texts"]) + len(pool["meso_texts"]), dtype=bool)
for lv in (0, 1):
    for i, t in enumerate(texts3[lv]):
        ok_mask[(0 if lv == 0 else len(texts3[0])) + i] = \
            len(t.split()) >= 8 and not g._pool_junk(t) and canon[lv][i]
print(f"🧹 целых каноничных зёрен в памяти: {int(ok_mask.sum())}/{len(ok_mask)}")

# второй проход по маске памяти:
# 1) метаданные-цитаты (pp./doi/издательства) — не проза, учитель их не брал;
# 2) МЕЖУРОВНЕВЫЕ дубли: канон чистил уровни раздельно, копия микро↔мезо жила
import re as _re
_cit = _re.compile(r"(pp\.\s*\d|doi:|issn|©|all rights reserved|retrieved \d"
                   r"|edp sciences|springer|elsevier|\bwiley\b|nature publishing"
                   r"|oxford university press|cambridge university press)", _re.I)
_nrm = lambda t: _re.sub(r"[^a-zа-я0-9]+", " ", t.lower()).strip()
buckets = {}
for i in range(len(ok_mask)):
    if not ok_mask[i]:
        continue
    t = texts3[0][i] if i < len(texts3[0]) else texts3[1][i - len(texts3[0])]
    if _cit.search(t):
        ok_mask[i] = False
        continue
    nrm_t = _nrm(t)
    toks = set(nrm_t.split())
    b = nrm_t[:60]
    dup = False
    for tk_ in buckets.get(b, ()):
        jac = len(toks & tk_) / max(1, len(toks | tk_))
        contain = toks <= tk_ or tk_ <= toks      # обрезки-варианты синдиката
        if jac > 0.8 or (contain and min(len(toks), len(tk_)) >= 8):
            dup = True
            break
    if dup:
        ok_mask[i] = False
    else:
        buckets.setdefault(b, []).append(frozenset(toks))
print(f"🧹 после цитат+межуровневых дублей: {int(ok_mask.sum())}")

brain = {
    # ── веса (планировщик) ──
    "model_state": model.state_dict(),
    "feat_dim": FD,
    # ── память (датасет целиком, без потерь) ──
    "pool": {k: pool[k] for k in pool.files} if hasattr(pool, "files") else dict(pool),
    "sems": {lv: np.asarray(sems[lv]).astype(np.float16) for lv in (0, 1, 2)},
    "clusters": clusters,
    "proj_components": sp[0], "proj_mean": sp[1],
    "affect": affect,
    "doc_embs": doc_embs.astype(np.float32),
    "ok_mask": ok_mask,
    # ── политика выбора (константы, отобранные форензикой) ──
    "policy": {"context_len": g.CONTEXT_LEN, "recall_floor": 0.0,
               "anti_repeat": 4, "max_per_answer": 64},
    # ── провенанс ──
    "meta": {
        "born": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lineage": "v21_planner: 53K демо + док-дискриминативные негативы",
        "pool_sha": sha(g.POOL_CACHE), "sem_cache_sha": sha(str(g.GLM_DIR / "pool" / "text_sem_mini_v1.npz")),
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "note": "память встроена дословно; планирование в весах; чтение = адресуемое припоминание",
    },
}
out = str(g.GLM_DIR / "checkpoints" / "brain_v3.pt")
torch.save(brain, out)
mb = os.path.getsize(out) / 2**20
print(f"✅ мозг v1 запечён: {out} ({mb:.0f} МБ) за {time.time()-t0:.0f}с")
