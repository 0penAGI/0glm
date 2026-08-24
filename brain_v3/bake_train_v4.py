"""BAKE v2: нормальное запекание — поведение в веса, не в рантайм.
Три изменения против v0 (все в ТРЕНИРОВКЕ):
1) Пары только между ЦЕЛЫМИ зернами (>=8 слов, не junk): качество припоминания
   становится свойством поля. Огрызки выпадают из задачи целиком.
2) Батчи собираются из ОДНОГО кластера → in-batch негативы семантически близкие,
   InfoNCE учит различать тонко (лечение pred·q=.27).
3) Дропаут фич контекста p=0.1 против аттракторов в rollout.
Тёплый старт от чистого v15. Рантайм после этого срезается до черепа."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import granular_text_field as g
from torch.utils.data import DataLoader

pool = g.load_pool()
clusters = json.load(open(g.CLUSTERS_CACHE))
sems = g.build_semantics(pool)
sp = g.load_sem_projection(); proj = g.project_sems(sems, *sp)
proj_all = np.concatenate([proj[0], proj[1], proj[2]], axis=0)
affect = g.load_affect()
all_f = np.concatenate([pool["micro_feats"], pool["meso_feats"], pool["macro_feats"]])
n_micro, n_meso = len(pool["micro_feats"]), len(pool["meso_feats"])
CL = g.CONTEXT_LEN
MIN_W = 8          # целое зерно: от восьми слов
texts3 = [pool["micro_texts"], pool["meso_texts"], pool["macro_texts"]]

def whole(lv, ix):
    if lv > 2 or ix >= len(texts3[lv]): return False
    t = texts3[lv][ix]
    return len(t.split()) >= MIN_W and not g._pool_junk(t) and canon[lv][ix]

canon = g.canonical_mask(pool)
print("\n📐 BAKE-v3 pairs: целые + канонические зёрна...")
pairs = []
n_traj_kept = 0
for traj in pool.get("trajectories", []):
    good = [(l, i) for l, i in traj if int(l) <= 2 and whole(int(l), int(i))]
    if len(good) < CL + 2: continue
    n_traj_kept += 1
    _vs = [sems[int(l)][int(i)] for l, i in good[:16]]
    _m = np.mean(_vs, axis=0)
    qsem = (_m / (np.linalg.norm(_m) + 1e-9)).astype(np.float16)
    for k in range(CL, len(good)):
        ctx_feats = []
        ok = True
        for j in range(k - CL, k):
            lev, idx = good[j]
            gi = g._level_to_all_f_idx(lev, idx, n_micro, n_meso)
            row = np.concatenate([all_f[gi], affect[lev][idx], proj_all[gi]])
            ctx_feats.append(row)
        if not ok: continue
        t_lev, t_idx = good[k]
        p_lev, p_idx = good[k - 1]
        t_gi = g._level_to_all_f_idx(t_lev, t_idx, n_micro, n_meso)
        p_gi = g._level_to_all_f_idx(p_lev, p_idx, n_micro, n_meso)
        ln = ["micro", "meso", "macro"][t_lev]
        cid = int(clusters[ln][min(t_idx, len(clusters[ln]) - 1)])
        pairs.append({
            "ctx": np.array(ctx_feats, dtype=np.float32),
            "cluster": cid,
            "level": t_lev,
            "params": g.extract_params_from_feats(all_f[p_gi], all_f[t_gi]),
            "density": float(np.clip(all_f[t_gi][20] / 40.0, 0, 1)),
            "delta_next": np.clip(all_f[t_gi] - all_f[p_gi], -3, 3).astype(np.float32) / 3.0,
            "cond": np.zeros(g.FEAT_DIM, dtype=np.float32),
            "affect_next": np.asarray(affect[t_lev][t_idx], dtype=np.float32),
            "sem_next": sems[t_lev][t_idx].astype(np.float16),
            "qsem": qsem,
        })
print(f"  ✅ {len(pairs)} пар из {n_traj_kept} траекторий (целые зёрна)")

FD = g.FEAT_DIM + g.AFFECT_DIM + g.SEM_PROJ_DIM
model = g.MultiNavigator(feat_dim=FD).to(g.DEVICE)
g.safe_load(model, str(g.GLM_DIR / "checkpoints" / "text_navigator_v20_planner.pt"))

# ── демонстрации ride (учитель 5:0): контексты с дрейфом + правильные ходы ──
# ── индекс для дискриминативного члена: кластер → док → семы целых
# каноничных зёрен. Негатив = тот же кластер, ЧУЖОЙ док («парафраз-ловушка»):
# поле учится различать продолжение линии и пересказ из соседней статьи.
canon_all = g.canonical_mask(pool)
g2doc = {}
for _ti, _tr in enumerate(pool["trajectories"]):
    for _l, _i in _tr:
        if int(_l) <= 1:
            g2doc[(int(_l), int(_i))] = _ti
clust_pool = {}
for lv in (0, 1):
    ln = ["micro", "meso"][lv]
    for ix in range(len(texts3[lv])):
        t = texts3[lv][ix]
        if not canon_all[lv][ix] or len(t.split()) < 8 or g._pool_junk(t):
            continue
        cid = int(clusters[ln][min(ix, len(clusters[ln]) - 1)])
        clust_pool.setdefault(cid, {}).setdefault(g2doc.get((lv, ix), -1), []).append(
            np.asarray(sems[lv][ix], dtype=np.float16))
print(f"🗂 кластеров в негатив-пуле: {len(clust_pool)}")

K_NEG = 3
demo_pairs = []
with open(str(g.GLM_DIR / "pool" / "demos_v2.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        d.pop("src", None); d.pop("ep", None); d.pop("turn", None)
        c = np.array(d["ctx"], dtype=np.float32)          # выравнивание окна до 11
        if len(c) < g.CONTEXT_LEN - 1:
            c = np.concatenate([np.zeros((g.CONTEXT_LEN - 1 - len(c), c.shape[1]),
                                         dtype=np.float32), c], axis=0)
            d["ctx"] = c
        # K негативов: тот же кластер, ЧУЖОЙ док
        tgt = np.asarray(d["sem_next"], dtype=np.float32)
        per_doc = clust_pool.get(int(d["cluster"]), {})
        foreign = [s for doc, ss in per_doc.items()
                   if doc != int(d.get("doc", -2)) for s in ss]
        negs = []
        if foreign:
            pick = np.random.choice(len(foreign),
                                    size=min(K_NEG, len(foreign)), replace=False)
            negs = [foreign[int(p)] for p in pick]
        while len(negs) < K_NEG:
            negs.append(np.zeros(384, dtype=np.float16))
        d["neg_sems"] = np.array(negs, dtype=np.float16)
        demo_pairs.append(d)
print(f"🎓 демо-троек: {len(demo_pairs)}")
all_pairs = pairs + demo_pairs

# ── поток батчей: 75% кластерных (жёсткие негативы) + 10% глобальных +
# 15% ДЕМО (дистилляция поведения учителя, включая коррекции дрейфа) ──
BS = g.BATCH_SIZE
order = np.argsort(np.array([p["cluster"] for p in pairs]), kind="stable")
cl_batches = [order[i:i + BS] for i in range(0, len(pairs) - BS + 1, BS)]
n_rnd = max(1, int(len(cl_batches) * 0.13))
rnd_batches = [np.random.choice(len(pairs), BS, replace=False) for _ in range(n_rnd)]
ds_off = len(pairs)
dm_batches = [np.array(ds_off + rng_i) for rng_i in
              (np.random.choice(len(demo_pairs), BS, replace=True) for _ in range(int(len(cl_batches) * 0.18)))]
batches = cl_batches + rnd_batches + dm_batches

class _DS(g.MultiPairDS):
    """+ негативы «чужой док» и флаг демо для дискриминативного члена"""
    def __getitem__(self, i):
        p = self.p[i]
        out = list(super().__getitem__(i))
        if "neg_sems" in p:
            neg = torch.tensor(np.asarray(p["neg_sems"], dtype=np.float32))
            out.append(neg)
            out.append(torch.tensor(1.0))
        else:
            out.append(torch.zeros(K_NEG, g.SEM_DIM))
            out.append(torch.tensor(0.0))
        return tuple(out)

print(f"🔥 BAKE-v4 TRAINING: {len(cl_batches)} кластерных + {n_rnd} глобальных + "
      f"{len(dm_batches)} демо-батчей × BS={BS}")

ds = _DS(all_pairs)
opt = torch.optim.AdamW(model.parameters(), lr=g.LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8000)
model.train(); losses = []; t0 = time.time(); step = 0
N_STEPS = 8000
while step < N_STEPS:
    np.random.shuffle(batches)
    for bidx in batches:
        if step >= N_STEPS: break
        batch = [torch.stack([ds[int(i)][j] for i in bidx]).to(g.DEVICE)
                 for j in range(14)]
        ctx, tgt_c, tgt_l, tgt_p, tgt_dn, tgt_d, cond, stream_idx, tgt_sem, tgt_top, tgt_aff, qsem, nsem, isdemo = batch
        # дропаут фич контекста: поле не должно залипать за одним каналом
        keep = (torch.rand_like(ctx) > 0.1).float() / 0.9
        ctx = ctx * keep
        cl, lv, pr, dn, zz, sem_pred, topic_pred, aff_pred = model(
            ctx, stream_idx=stream_idx, cond=cond, qsem=qsem)
        loss_c = torch.nn.functional.cross_entropy(cl, tgt_c)
        loss_l = 0.5 * torch.nn.functional.cross_entropy(lv, tgt_l)
        loss_p = torch.nn.functional.mse_loss(pr, tgt_p)
        loss_dn = torch.nn.functional.mse_loss(dn.squeeze(-1), tgt_dn)
        pred_d = model.feat_head(zz)
        cos = torch.nn.functional.cosine_embedding_loss(
            pred_d, tgt_d, torch.ones(pred_d.shape[0], device=g.DEVICE))
        loss_z = 0.25 * (cos + torch.nn.functional.mse_loss(pred_d, tgt_d))
        ctx_proj = model.proj(model.feat_enc(ctx)).mean(dim=1)
        loss_at = 0.1 * torch.nn.functional.mse_loss(model.attractor_state[stream_idx], ctx_proj)
        sp_ = torch.nn.functional.normalize(sem_pred, dim=-1)
        tsn = torch.nn.functional.normalize(tgt_sem, dim=-1)
        logits_sems = sp_ @ tsn.t() / 0.1
        tgt_rank = torch.arange(sp_.shape[0], device=g.DEVICE)
        w_lvl = torch.tensor([0.3, 1.0, 0.7], device=g.DEVICE)[tgt_l]
        loss_sem = (torch.nn.functional.cross_entropy(logits_sems, tgt_rank,
                   reduction="none") * w_lvl).mean()
        # дискриминативный член (только демо): цель должна бить негативы
        # «тот же кластер, чужой док» — против коллажа из парафразов
        if bool((isdemo > 0).any()):
            pos_sim = (sp_ * tsn).sum(-1, keepdim=True)
            neg_sim = torch.einsum("bd,bkd->bk", sp_, nsem)
            logits_d = torch.cat([pos_sim, neg_sim], dim=-1) / 0.1
            loss_disc = 0.5 * torch.nn.functional.cross_entropy(
                logits_d, torch.zeros(logits_d.shape[0], dtype=torch.long,
                                      device=g.DEVICE)) * (isdemo > 0).float().mean()
        else:
            loss_disc = torch.zeros((), device=g.DEVICE)
        loss_aff = torch.nn.functional.mse_loss(aff_pred, tgt_aff)
        loss = (loss_c + loss_l + loss_p + 0.5 * loss_dn + loss_z + loss_at
                + 0.2 * loss_sem + 0.5 * loss_aff + loss_disc)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(loss.item()); step += 1
        if step % 500 == 0:
            avg = np.mean(losses[-500:]); e = time.time() - t0
            print(f"  step {step:5d}/{N_STEPS}  loss={avg:.4f} (c={loss_c:.3f} l={loss_l:.3f} "
                  f"p={loss_p:.3f} z={loss_z:.3f} semNCE={loss_sem:.3f} aff={loss_aff:.3f})  "
                  f"{e:.0f}s  ETA {e/step*(N_STEPS-step):.0f}s")
print(f"\n   ✅ {time.time()-t0:.1f}s, loss={np.mean(losses[-100:]):.4f}")
out = str(g.GLM_DIR / "checkpoints" / "text_navigator_v21_planner.pt")
torch.save({"model_state": model.state_dict(), "feat_dim": FD}, out)
print(f"✅ запечён планировщик v6 (док-дискриминативные негативы) → {out}")
