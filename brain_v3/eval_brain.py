import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import brain_chat as bc
import granular_text_field as g

QS = ["How does climate change affect the ocean?", "What is a black hole?",
      "Why do humans need sleep?", "How does a Python dict work internally?",
      "What causes El Nino?"]
print("🧠 грузим мозг...")
B = bc.load_brain("0glm/checkpoints/brain_v3.pt")
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
outdir = "0glm/output/brain_eval"
os.makedirs(outdir, exist_ok=True)
res = {}
for i, q in enumerate(QS, 1):
    text, tr = bc.ask(B, q, st, n_steps=24)
    qn = st.encode([q])[0]; qn /= np.linalg.norm(qn)
    at = st.encode([text[:1000]])[0]; at /= np.linalg.norm(at)
    rel = float(qn @ at)
    sims = [s for _, s, _ in tr] if tr else [0]
    docs = len({B["inv"].get(c, -1) for _, _, c in tr}) if tr else 0
    open(f"{outdir}/q{i}.txt", "w").write(text)
    res[q] = dict(rel=round(rel, 3), recall=round(float(np.mean(sims)), 3), docs=docs)
    print(f"  {q[:44]:46} rel={rel:.3f} recall={np.mean(sims):.3f} доков={docs}")
json.dump(res, open(f"{outdir}/metrics.json", "w"), ensure_ascii=False, indent=1)
