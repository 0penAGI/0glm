"""Форензика запека: где врёт поле, а где — мой рантайм.
На каждом шаге rollout логируем: чистый предсказ sem (без магнита),
кого выбрал бы чистый argmax, кого выбрал магнит, длины зёрен."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import granular_text_field as g
import brain_chat as bc

B = bc.load_brain("0glm/checkpoints/brain_v3.pt")
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
QS = ["How does climate change affect the ocean?", "What is a black hole?",
      "Why do humans need sleep?", "How does a Python dict work internally?",
      "What causes El Nino?"]

n_mic = len(B["pool"]["micro_texts"])
def F(x): return x / (np.linalg.norm(x) + 1e-9)

for q in QS[:2]:   # двух вопросов хватит для картины
    qn = F(st.encode([q])[0].astype(np.float32))
    d2sim = B["doc_embs"] @ qn
    d = int(np.argmax(d2sim))
    mems = [(int(l), int(i)) for l, i in B["pool"]["trajectories"][d] if int(l) <= 1]
    ok = [k for k, (lv, ix) in enumerate(mems)
          if ix < len(B["sems"][lv]) and not g._pool_junk(B["texts"][(0 if lv==0 else n_mic)+ix])]
    S = np.array([B["sems"][mems[k][0]][mems[k][1]] for k in ok])
    k0 = int(np.argmax(S @ qn)); lv, ix = mems[ok[k0]]
    ci = (0 if lv == 0 else n_mic) + ix
    rows, spoken, log = [bc.row(B, lv, ix)], [], []
    print(f"\n=== {q}\nвход: [{len(B['texts'][ci].split())}w] {B['texts'][ci][:70]!r}")
    for step in range(24):
        states = torch.tensor(np.array(rows[-11:]), dtype=torch.float32,
                              device=g.DEVICE).unsqueeze(0)
        with torch.no_grad():
            heads = B["model"](states, stream_idx=0,
                               cond=torch.zeros(1, g.FEAT_DIM, device=g.DEVICE),
                               qsem=torch.tensor(qn).unsqueeze(0).to(g.DEVICE))
        pred = F(heads[5][0].cpu().numpy())
        sims_pred = B["M"] @ pred              # ЧИСТОЕ поле
        tgt = F(pred + 0.25 * qn)
        sims_mag = B["M"] @ tgt                # с магнитом
        for c in spoken: sims_pred[c] = -1; sims_mag[c] = -1
        i_p, i_m = int(np.argmax(sims_pred)), int(np.argmax(sims_mag))
        tp, tm = B["texts"][i_p], B["texts"][i_m]
        log.append(dict(step=step,
            pure_top=f"[{len(tp.split())}w cos={sims_pred[i_p]:.3f}] {tp[:48]!r}",
            mag_top=f"[{len(tm.split())}w cos={sims_mag[i_m]:.3f}] {tm[:48]!r}",
            agree=i_p == i_m,
            pred_q=float(pred @ qn),               # куда поле смотрит относительно темы
            chosen_q=float(sims_mag[i_m] and (F(B['M'][i_m]) @ qn))))
        bi = i_m; lv_, ix_ = B["coords"][bi]
        rows.append(bc.row(B, lv_, ix_)); spoken.append(bi)
    ag = sum(l["agree"] for l in log)
    print(f"поле vs магнит совпали: {ag}/24")
    print(f"pred·qmean={np.mean([l['pred_q'] for l in log]):.3f}  "
          f"chosen·qmean={np.mean([l['chosen_q'] for l in log]):.3f}")
    for l in log[:12]:
        a = "==" if l["agree"] else "!="
        print(f" s{l['step']:02d} {a} ПОЛЕ {l['pure_top']}")
        if not l["agree"]: print(f"           МАГН {l['mag_top']}")
