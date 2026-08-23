"""
TEXT GRAIN FIELD v1 — зеркало granular_field.py (0MGE) на тексте
корпус → иерархия гранул → multi-stream Navigator → text engine → weave → z-trajectory

0MGE: waveform → micro/meso/macro grains → 22-dim spectral feats → clusters
0GLM: текст   → phrase/sent/para grains → 32-dim stylometric feats → clusters
"""
import os, re, json, time, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import MiniBatchKMeans
warnings.filterwarnings("ignore")

GLM_DIR = Path(__file__).resolve().parent
ROOT = GLM_DIR.parent

# Основной датасет — декодированный corpus-zero (0agi), поверх — весь воркспейс
SCAN_DIRS = [str(GLM_DIR / "corpus"), str(GLM_DIR / "corpus_qa"), str(ROOT)]
SKIP_DIRS = {
    "node_modules", ".git", "venv", ".venv", "__pycache__", "checkpoints", "hf_cache",
    "cache", "models", "samples", "corpus-zero", "dist", "build", "release", "installers",
    ".pytest_cache", ".vite", "audio", "temp", "tmp", "output", "artifacts", "datasets",
    "dataset", "ACE-Step", "FramePack", "StreamDiffusion", "cosmos-predict2", "coqpit",
    "ml-fastvlm", "minecraft", "starstoke", "language_tool", "BlackHole", "one", "new",
    "bin", "lib", "static", "effects", "characters", "souls", "field", "core", "api",
    "gui", "examples", "Operations", "Protocols", "Architecture", "reference_images",
    "nft_images", "screenshots", "agent_screenshots", "photo_cache", "tts_cache",
    "vortex_logs", "state_data", "saved_models", "checkpoints_latent_memory",
    "checkpoints_stable", "latent_checkpoints", "enhanced_checkpoints", "adapters_0penAGI",
    "echo_lora_adapters", "zephyr_ollama", "ollama_export", "organism_data", "woa18_data",
    "reddit_cache", "aether_backups", "aether_state", "field_workspace", ".openclaw",
    ".qwen", ".claude", "ai_creations", "generated_videos", "temp_convert", "temp_deps",
    "temp_files", "ton_keystore", "data_zephyr", "data_zephyr_dcat", "data_zephyr_enhanced",
    "data_zephyr_full", "data_zephyr_safe", "data_zephyr_v5", "zh_lora_dataset",
}
TEXT_EXTS = {".md", ".txt"}
MAX_FILE_BYTES = 2_000_000
MIN_FILE_BYTES = 200

# Уровни гранул (зеркало MICRO/MESO/MACRO_FRAMES): фраза / предложение(1-2) / абзац
MICRO_MIN_W, MICRO_MAX_W = 3, 24
MESO_WINDOW = 2
MACRO_MIN_W = 5
MAX_MICRO_PER_DOC, MAX_MESO_PER_DOC, MAX_MACRO_PER_DOC = 300, 100, 30

FEAT_DIM = 32          # 0MGE: 22 spectral → 0GLM: 32 stylometric
N_CLUSTERS = 1024
STATE_DIM = 48
CONTEXT_LEN = 12
HIDDEN_DIM = 192
N_HEADS = 4
N_LAYERS = 3
BATCH_SIZE = 128
TRAIN_STEPS = 8000
LR = 3e-4
Z_DIM = 64             # joint style space: text grain → z → (этап 5) audio grain
SEM_DIM = 384          # плотная семантика гранул: MiniLM all-MiniLM-L6-v2
TOPIC_EMA = 0.85       # дискурсный аттрактор: тема дрейфует медленнее стиля

N_STREAMS = 6
N_ATTRACTORS = 6       # one per stream — learned global direction for long-range coherence
STREAM_NAMES = ["narrative", "dialogue", "description", "argument", "lyric", "fragment"]
STREAM_WEIGHTS = [1.0, 0.6, 0.7, 0.5, 0.35, 0.15]   # зеркало STREAM_WEIGHTS

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
POOL_DIR = GLM_DIR / "pool"; POOL_DIR.mkdir(exist_ok=True)
QA_MAP = str(POOL_DIR / "qa_map.json")   # doc filename → стилометрия запроса (cond, 0GLM-Q)
FILELIST_CACHE = str(POOL_DIR / "filelist.json")
POOL_CACHE = str(POOL_DIR / "text_pool_v1.npz")
GRAINS_JSONL = str(POOL_DIR / "text_pool_v1.jsonl")
CLUSTERS_CACHE = str(POOL_DIR / "clusters_v1.json")
MODEL_MULTI_CACHE = str(GLM_DIR / "checkpoints" / "text_navigator_v11_cleanpool.pt")
OUT = str(GLM_DIR / "output"); os.makedirs(OUT, exist_ok=True)

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:['’\-][A-Za-zА-Яа-яЁё]+)*")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
PHRASE_SPLIT_RE = re.compile(r"[,;:]|\s—\s|\s–\s")
VOWEL_RE = re.compile(r"[aeiouyаеёиоуыэюя]", re.IGNORECASE)

STOPWORDS = frozenset("""a an the and or but if then else of to in on at by for with from as is are
was were be been being am i you he she it we they me him her us them my your his its our their mine
yours this that these those there here what which who whom whose when where why how not no nor so
than too very can will just should now would could shall may might must do does did done have has had
о и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было
вот от меня еще нет из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь
опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам
чтоб без будто чего раз тоже себе под будет ж тогда кто этот говорил того потому этого какой совсем
ним здесь этом один почти мой тем чтобы нее кажется сейчас были куда зачем всех никогда сегодня можно
при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три
эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю
между""".split())
DISCOURSE = frozenset("however therefore thus moreover furthermore nevertheless although though because since hence accordingly indeed meanwhile otherwise однако поэтому впрочем хотя значит следовательно причем зато".split())
PRONOUNS = frozenset("i you he she it we they me him her us them my your his its our their who whom whose я ты он она оно мы вы они меня тебе его ее нам вас их мной тобой им ей нами вами ими кто чей который".split())
ADJ_SUF = ("ful", "ous", "ive", "able", "ible", "less", "ish", "al", "ic", "ый", "ая", "ое", "ые", "ий", "яя", "ее", "ие", "ой")
VERB_SUF = ("ed", "ing", "ize", "ise", "fy", "ть", "ся", "ла", "ло", "ли", "ет", "ют", "ит", "ат", "ят")


def md5_fast(p):
    import subprocess
    try: return subprocess.run(["md5", "-q", p], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: return None


def scan_text(dirs):
    """Зеркало scan_audio: обход dirs, фильтр расширений, md5-dedup, filelist cache."""
    if os.path.exists(FILELIST_CACHE):
        with open(FILELIST_CACHE) as f: cached = json.load(f)
        files = [p for p in cached if os.path.exists(p)]
        if files: print(f"📦 Filelist cache: {len(files)}"); return files
    seen = {}
    for d in dirs:
        if not os.path.isdir(d): continue
        for root, _, fnames in os.walk(d):
            rel = os.path.relpath(root, d)
            if any(s == rel or s in rel.split(os.sep) for s in SKIP_DIRS): continue
            for fn in fnames:
                if os.path.splitext(fn)[1].lower() not in TEXT_EXTS: continue
                fp = os.path.join(root, fn)
                try: sz = os.path.getsize(fp)
                except OSError: continue
                if sz < MIN_FILE_BYTES or sz > MAX_FILE_BYTES: continue
                h = md5_fast(fp)
                if h and h not in seen: seen[h] = fp
    files = list(seen.values())
    with open(FILELIST_CACHE, "w") as f: json.dump(files, f)
    return files


def extract_feat_from_text(text):
    """Стилометрический вектор гранулы — зеркало extract_feat_from_stft.

    Layout (FEAT_DIM=32):
      0 log_len_words      1 mean_word_len       2 std_word_len        3 lexical_diversity
      4 hapax_ratio        5 stopword_ratio      6 long_word_ratio     7 short_word_ratio
      8 digit_ratio        9 upper_ratio        10 punct_density      11 comma_density
     12 period_density    13 quest_excl_density 14 colon_semi_density 15 quote_density
     16 dash_density      17 ellipsis_density   18 paren_density      19 newline_density
     20 mean_sent_words   21 std_sent_words     22 burstiness         23 syllables_per_word
     24 bigram_repeat     25 drift_half         26 discourse_ratio    27 pronoun_ratio
     28 dialogue_marker   29 adj_suffix_ratio   30 verb_suffix_ratio  31 content_density
    """
    ws = WORD_RE.findall(text)
    n = len(ws)
    if n < 3: return None
    lw = [w.lower() for w in ws]
    wlens = np.array([len(w) for w in ws], dtype=np.float64)
    nchars = max(1, len(text))
    alpha = sum(c.isalpha() for c in text)
    feat = []

    # 0-2 length stats
    feat.append(float(np.log1p(n)))
    feat.append(float(wlens.mean()))
    feat.append(float(wlens.std()))
    # 3-4 lexical diversity
    uniq = len(set(lw))
    counts = {}
    for w in lw: counts[w] = counts.get(w, 0) + 1
    feat.append(uniq / n)
    feat.append(sum(1 for v in counts.values() if v == 1) / n)
    # 5-7 word classes
    sw = sum(1 for w in lw if w in STOPWORDS)
    feat.append(sw / n)
    feat.append(float((wlens > 7).mean()))
    feat.append(float((wlens <= 3).mean()))
    # 8-9 char classes
    feat.append(sum(c.isdigit() for c in text) / nchars)
    feat.append(sum(c.isupper() for c in text) / max(1, alpha))
    # 10-19 punctuation densities per word
    t = lambda ch: text.count(ch) / n
    punct_all = sum(text.count(c) for c in ",.;:!?…—–()«»\"'")
    feat.append(punct_all / n)
    feat.append(t(","))
    feat.append(t(".") + t("…"))
    feat.append((t("?") + t("!")) / 2)
    feat.append(t(":") + t(";"))
    feat.append((t("«") + t("»") + t('"')) / 2)
    feat.append(t("—") + t("–"))
    feat.append(t("…"))
    feat.append((t("(") + t(")")) / 2)
    feat.append(text.count("\n") / n)
    # 20-22 sentence rhythm
    sents = [s for s in SENT_SPLIT_RE.split(text) if s.strip()]
    scounts = np.array([max(1, len(WORD_RE.findall(s))) for s in sents], dtype=np.float64)
    feat.append(float(scounts.mean()))
    feat.append(float(scounts.std()))
    feat.append(float(np.clip(scounts.std() / (scounts.mean() + 1e-6), 0, 5)))
    # 23 syllables proxy
    feat.append(sum(max(1, len(VOWEL_RE.findall(w))) for w in ws) / n)
    # 24 bigram repetition
    bigrams = list(zip(lw, lw[1:]))
    feat.append(1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0)
    # 25 lexical drift between halves
    half = n // 2
    a, b = set(lw[:half]), set(lw[half:])
    feat.append(1 - len(a & b) / max(1, len(a | b)))
    # 26-30 lexicon classes
    feat.append(sum(1 for w in lw if w in DISCOURSE) / n)
    feat.append(sum(1 for w in lw if w in PRONOUNS) / n)
    feat.append((t("«") + t('"') + t("—")) / n)
    feat.append(sum(1 for w in lw if w.endswith(ADJ_SUF)) / n)
    feat.append(sum(1 for w in lw if w.endswith(VERB_SUF)) / n)
    # 31 content density
    feat.append(1 - sw / n)

    v = np.nan_to_num(np.array(feat[:FEAT_DIM], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(v, -10, 10)


def paragraph_spans(text):
    spans, start, off = [], None, 0
    for line in text.split("\n"):
        if line.strip():
            if start is None: start = off
        else:
            if start is not None: spans.append((start, off)); start = None
        off += len(line) + 1
    if start is not None: spans.append((start, len(text)))
    return spans


def sentence_spans(ptext, base):
    out, pos = [], 0
    for chunk in SENT_SPLIT_RE.split(ptext):
        if not chunk.strip(): continue
        idx = ptext.find(chunk, pos)
        if idx < 0: continue
        out.append((base + idx, base + idx + len(chunk)))
        pos = idx + len(chunk)
    return out


def phrase_spans(stext, base):
    cuts = [0] + [m.end() for m in PHRASE_SPLIT_RE.finditer(stext)] + [len(stext)]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        if stext[a:b].strip(): out.append((base + a, base + b))
    return out


def _merge_small(spans, text, min_w):
    merged = []
    for a, b in spans:
        w = len(WORD_RE.findall(text[a:b]))
        if merged and w < min_w:
            pa, pb = merged[-1]
            merged[-1] = (pa, b)
        elif w < min_w and not merged:
            merged.append((a, b))
        else:
            merged.append((a, b))
    return merged


def _chunk_long(a, b, text, max_w):
    ws = list(WORD_RE.finditer(text[a:b]))
    if len(ws) <= max_w: return [(a, b)]
    out, step = [], max_w // 2
    for i in range(0, len(ws) - max_w + 1, step):
        out.append((a + ws[i].start(), a + ws[i + max_w - 1].end()))
    return out


def extract_all(files):
    """Зеркало extract_all: документы → micro/meso/macro гранулы → фичи → траектории."""
    print(f"\n🧬 Extracting text grains from {len(files)} docs...")
    t0 = time.time()
    feats = {0: [], 1: [], 2: []}
    texts = {0: [], 1: [], 2: []}
    sources = {0: [], 1: [], 2: []}
    trajectories = []
    errors = 0

    for i, fp in enumerate(files):
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            e = time.time() - t0; r = (i + 1) / (e + 0.001)
            print(f"\r  [{i+1}/{len(files)}] μ={len(feats[0])} σ={len(feats[1])} Ω={len(feats[2])} err={errors} {r:.1f}d/s  ", end="", flush=True)
        try:
            text = Path(fp).read_text(encoding="utf-8", errors="ignore")
            if len(text) < MIN_FILE_BYTES: errors += 1; continue
            traj = []

            paras = paragraph_spans(text)
            for pa, pb in paras:
                ptext = text[pa:pb]
                if len(WORD_RE.findall(ptext)) < MACRO_MIN_W: continue
                if _pool_junk(ptext): continue
                if len(feats[2]) >= MAX_MACRO_PER_DOC * (len(files) or 1): pass
                f = extract_feat_from_text(ptext)
                if f is None: continue
                feats[2].append(f); texts[2].append(ptext.strip()); sources[2].append([fp, pa, pb])
                traj.append([2, len(feats[2]) - 1])

                sents = sentence_spans(ptext, pa)
                cnt_meso = 0
                for k in range(len(sents)):
                    if cnt_meso >= MAX_MESO_PER_DOC: break
                    window = sents[k:k + MESO_WINDOW]
                    sa, sb = window[0][0], window[-1][1]
                    stext = text[sa:sb]
                    if _pool_junk(stext): continue
                    f = extract_feat_from_text(stext)
                    if f is None: continue
                    feats[1].append(f); texts[1].append(stext.strip()); sources[1].append([fp, sa, sb])
                    traj.append([1, len(feats[1]) - 1])
                    cnt_meso += 1

                cnt_micro = 0
                for sa, sb in sents:
                    if cnt_micro >= MAX_MICRO_PER_DOC: break
                    stext = text[sa:sb]
                    pieces = _merge_small(phrase_spans(stext, sa), text, MICRO_MIN_W)
                    for fa, fb in pieces:
                        if cnt_micro >= MAX_MICRO_PER_DOC: break
                        for ca, cb in _chunk_long(fa, fb, text, MICRO_MAX_W):
                            gtext = text[ca:cb]
                            nw = len(WORD_RE.findall(gtext))
                            if nw < MICRO_MIN_W: continue
                            if _pool_junk(gtext): continue
                            f = extract_feat_from_text(gtext)
                            if f is None: continue
                            feats[0].append(f); texts[0].append(gtext.strip()); sources[0].append([fp, ca, cb])
                            traj.append([0, len(feats[0]) - 1])
                            cnt_micro += 1
            if traj: trajectories.append(traj)
        except Exception:
            errors += 1

    elapsed = time.time() - t0
    print(f"\n  ✅ {elapsed:.1f}s — micro={len(feats[0])} meso={len(feats[1])} macro={len(feats[2])}")
    return {
        "micro_feats": np.array(feats[0], dtype=np.float32) if feats[0] else np.zeros((0, FEAT_DIM), np.float32),
        "meso_feats": np.array(feats[1], dtype=np.float32) if feats[1] else np.zeros((0, FEAT_DIM), np.float32),
        "macro_feats": np.array(feats[2], dtype=np.float32) if feats[2] else np.zeros((0, FEAT_DIM), np.float32),
        "micro_texts": texts[0], "meso_texts": texts[1], "macro_texts": texts[2],
        "micro_sources": sources[0], "meso_sources": sources[1], "macro_sources": sources[2],
        "trajectories": trajectories,
    }


def save_pool(pool):
    np.savez(POOL_CACHE,
             micro_feats=pool["micro_feats"], meso_feats=pool["meso_feats"], macro_feats=pool["macro_feats"],
             trajectories=np.array(pool["trajectories"], dtype=object),
             micro_sources=np.array(pool["micro_sources"], dtype=object),
             meso_sources=np.array(pool["meso_sources"], dtype=object),
             macro_sources=np.array(pool["macro_sources"], dtype=object))
    with open(GRAINS_JSONL, "w", encoding="utf-8") as f:
        for ln in ("micro", "meso", "macro"):
            for txt, src in zip(pool[f"{ln}_texts"], pool[f"{ln}_sources"]):
                f.write(json.dumps({"level": ln, "doc": src[0], "start": int(src[1]), "end": int(src[2]), "text": txt}, ensure_ascii=False) + "\n")
    print(f"💾 Pool saved: {POOL_CACHE} ({os.path.getsize(POOL_CACHE)/1e6:.1f} MB) + {GRAINS_JSONL}")


def load_pool():
    d = np.load(POOL_CACHE, allow_pickle=True)
    pool = {k: (d[k].tolist() if k in ("trajectories", "micro_sources", "meso_sources", "macro_sources") else d[k]) for k in d.files}
    by_level = {"micro": [], "meso": [], "macro": []}
    with open(GRAINS_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_level[r["level"]].append(r["text"])
    for ln in ("micro", "meso", "macro"):
        assert len(by_level[ln]) == len(pool[f"{ln}_feats"]), f"{ln}: texts/feats mismatch"
        pool[f"{ln}_texts"] = by_level[ln]
    return pool


def build_clusters(pool):
    """Идентично 0MGE: MiniBatchKMeans 1024, seed 42."""
    print("🔬 Clustering grains...")
    all_f = np.concatenate([pool["micro_feats"], pool["meso_feats"], pool["macro_feats"]])
    if len(all_f) > 80000:
        idx = np.random.choice(len(all_f), 80000, replace=False)
        fit_f = all_f[idx]
    else:
        fit_f = all_f
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=1024, n_init=3, random_state=42)
    kmeans.fit(fit_f)
    all_labels = kmeans.predict(all_f)
    n_m, n_s = len(pool["micro_feats"]), len(pool["meso_feats"])
    clusters = {
        "micro": [int(x) for x in all_labels[:n_m]],
        "meso": [int(x) for x in all_labels[n_m:n_m + n_s]],
        "macro": [int(x) for x in all_labels[n_m + n_s:]],
    }
    with open(CLUSTERS_CACHE, "w") as f: json.dump(clusters, f)
    print(f"  ✅ {N_CLUSTERS} clusters")
    return clusters


# ══════════════════════════════════════════════════════════════
# NAVIGATOR (зеркало 1-в-1, отличается только FEAT_DIM)
# ══════════════════════════════════════════════════════════════
class Navigator(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, state_dim=STATE_DIM, hidden=HIDDEN_DIM,
                 ctx=CONTEXT_LEN, n_clusters=N_CLUSTERS):
        super().__init__()
        self.feat_enc = nn.Linear(feat_dim, state_dim)
        self.pos = nn.Parameter(torch.randn(1, ctx, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=N_HEADS,
            dim_feedforward=hidden * 2, dropout=0.1, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.proj = nn.Linear(state_dim, hidden)
        self.cluster_head = nn.Linear(hidden, n_clusters)
        self.level_head = nn.Linear(hidden, 3)
        self.params_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 8), nn.Tanh())
        self.cond_proj = nn.Linear(4, hidden)

    def forward(self, states, cond=None):
        B, K, _ = states.shape
        z = self.proj(self.feat_enc(states)) + self.pos[:, :K, :]
        z = self.transformer(z)[:, -1, :]
        if cond is not None:
            z = z + self.cond_proj(cond)
        return self.cluster_head(z), self.level_head(z), self.params_head(z)

    @torch.no_grad()
    def step(self, states, temp=0.8, cond=None):
        self.eval()
        if states.dim() == 2: states = states.unsqueeze(0)
        cl, lv, pr = self.forward(states, cond=cond)
        c = F.softmax(cl.squeeze(0) / temp, dim=-1)
        l = F.softmax(lv.squeeze(0) / temp, dim=-1)
        cluster = torch.multinomial(c, 1).item()
        level = torch.multinomial(l, 1).item()
        p = pr.squeeze(0).cpu().numpy()
        return cluster, level, p


# ══════════════════════════════════════════════════════════════
# MULTI-STREAM NAVIGATOR (зеркало MultiNavigator + z-head)
# Стримы = регистры текста вместо частотных полос.
# Аттракторное поле перенесено БЕЗ ИЗМЕНЕНИЙ — это ядро переноса.
# ══════════════════════════════════════════════════════════════
class MultiNavigator(nn.Module):
    """6 independent streams: narrative, dialogue, description, argument, lyric, fragment.
    Shared backbone transformer + per-stream heads + attractor field + z-head."""

    def __init__(self, feat_dim=FEAT_DIM, state_dim=STATE_DIM, hidden=HIDDEN_DIM,
                 ctx=CONTEXT_LEN, n_clusters=N_CLUSTERS, n_streams=N_STREAMS, z_dim=Z_DIM,
                 n_layers=None):
        super().__init__()
        self.n_streams = n_streams
        self.z_dim = z_dim
        self.feat_dim = feat_dim
        self.feat_enc = nn.Linear(feat_dim, state_dim)
        self.pos = nn.Parameter(torch.randn(1, ctx, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=N_HEADS,
            dim_feedforward=hidden * 2, dropout=0.1, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers if n_layers is not None else N_LAYERS)
        self.proj = nn.Linear(state_dim, hidden)

        self.stream_cluster = nn.Linear(hidden, n_clusters)
        self.stream_level = nn.Linear(hidden, 3)
        self.stream_params = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 8), nn.Tanh())
        self.stream_density = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1), nn.Sigmoid())
        self.affect_head = nn.Linear(hidden, AFFECT_DIM)   # valence/arousal следующей гранулы

        self.cross_stream = nn.MultiheadAttention(hidden, num_heads=2, batch_first=True)
        self.stream_embed = nn.Parameter(torch.randn(1, n_streams, hidden) * 0.05)
        # cond — всегда стилевой вектор запроса (FEAT_DIM), даже когда контекст шире (0GLM-X)
        self.cond_proj = nn.Linear(FEAT_DIM, hidden)

        # ── Attractor field: learned global state per stream (из 0MGE без изменений) ──
        self.attractor_embed = nn.Parameter(torch.randn(n_streams, hidden) * 0.05)
        self.attractor_proj = nn.Linear(hidden, hidden)
        self.attractor_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.attractor_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Tanh())
        self.register_buffer("attractor_state", torch.zeros(n_streams, hidden))

        # ── Joint z-space: каждый шаг порождает «звучание» гранулы ──
        self.z_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, z_dim), nn.Tanh())
        # z предсказывает СТИЛОМЕТРИЮ следующей гранулы (FEAT_DIM), не семантику:
        # Δfeat-регрессия, expected_feat и аудио-слоты остаются в стилевом пространстве
        self.feat_head = nn.Linear(z_dim, FEAT_DIM)

        # ── Семантические головы (0GLM-V): градиент формирует hidden state,
        #    кодирующий «о чём я сейчас» → cluster logits становятся семантическими ──
        self.sem_head = nn.Linear(hidden, SEM_DIM)     # семантика следующей гранулы
        self.topic_head = nn.Linear(hidden, SEM_DIM)   # текущая тема дискурса

    def forward(self, states, stream_idx=0, cond=None):
        B, K, _ = states.shape
        z = self.proj(self.feat_enc(states)) + self.pos[:, :K, :]
        z = self.transformer(z)
        ctx_z = z[:, -1, :]

        if isinstance(stream_idx, int):
            se = self.stream_embed[:, stream_idx:stream_idx + 1].expand(B, -1, -1)
            as_ = self.attractor_state[stream_idx:stream_idx + 1].expand(B, -1, -1)
        elif stream_idx.dim() == 0:
            se = self.stream_embed[:, stream_idx:stream_idx + 1].expand(B, -1, -1)
            as_ = self.attractor_state[stream_idx:stream_idx + 1].expand(B, -1, -1)
        else:
            se = self.stream_embed.squeeze(0)[stream_idx].unsqueeze(1)
            as_ = self.attractor_state[stream_idx].unsqueeze(1)

        crossed, _ = self.cross_stream(se, z, z)

        ag = self.attractor_gate(torch.cat([ctx_z, as_.squeeze(1)], dim=-1))
        a_pull = ag * self.attractor_proj(as_.squeeze(1))

        h = (ctx_z + crossed.squeeze(1) + a_pull) / 3

        if cond is not None:
            h = h + self.cond_proj(cond)

        return (self.stream_cluster(h), self.stream_level(h),
                self.stream_params(h), self.stream_density(h), self.z_head(h),
                F.normalize(self.sem_head(h), dim=-1),
                F.normalize(self.topic_head(h), dim=-1),
                self.affect_head(h))

    def update_attractors(self, states, stream_idx, temp=0.8):
        """Зеркало 0MGE: EMA 0.9/0.1 аттрактора к текущему контексту."""
        with torch.no_grad():
            if states.dim() == 2: states = states.unsqueeze(0)
            z = self.proj(self.feat_enc(states))
            z = self.transformer(z)
            ctx_z = z[:, -1, :]
            if isinstance(stream_idx, int):
                as_ = self.attractor_state[stream_idx:stream_idx + 1].expand_as(ctx_z)
                new_state = self.attractor_update(torch.cat([ctx_z, as_], dim=-1))
                self.attractor_state[stream_idx] = 0.9 * self.attractor_state[stream_idx] + 0.1 * new_state.mean(0)
            else:
                idx = stream_idx if isinstance(stream_idx, list) else stream_idx.tolist()
                n = len(idx)
                as_ = self.attractor_state[idx]
                ctx_z_exp = ctx_z.expand(n, -1)
                new_state = self.attractor_update(torch.cat([ctx_z_exp, as_], dim=-1))
                for i, s in enumerate(idx):
                    self.attractor_state[s] = 0.9 * self.attractor_state[s] + 0.1 * new_state[i]

    @torch.no_grad()
    def step(self, states, stream_idx=0, temp=0.8, cond=None, prior=None, prior_lam=0.35):
        """prior: P(cluster|prev) — эмпирическая discourse-грамматика. Вход soft:
        финальное распределение = (1-λ)·softmax навигатора + λ·prior, жёстких
        запретов нет (матрица уже сглажена Лапласом)."""
        self.eval()
        if states.dim() == 2: states = states.unsqueeze(0)
        cl, lv, pr, dn, zz, sem_pred, topic_pred, _aff = self.forward(states, stream_idx=stream_idx, cond=cond)
        c = F.softmax(cl.squeeze(0) / temp, dim=-1)
        if prior is not None:
            c = (1.0 - prior_lam) * c + prior_lam * prior
        c = c / c.sum()
        l = F.softmax(lv.squeeze(0) / temp, dim=-1)
        return {
            "cluster": torch.multinomial(c, 1).item(),
            "level": torch.multinomial(l, 1).item(),
            "params": pr.squeeze(0).cpu().numpy(),
            "density": float(dn.squeeze(0).cpu()),
            "z": zz.squeeze(0).cpu().numpy(),
            "sem_pred": sem_pred.squeeze(0).cpu().numpy(),
            "topic_pred": topic_pred.squeeze(0).cpu().numpy(),
        }


# ══════════════════════════════════════════════════════════════
# TEXT ENGINE (зеркало GranularEngine: cluster_map + выборка + weave)
# ══════════════════════════════════════════════════════════════
class TextGrainEngine:
    def __init__(self, pool, clusters):
        lm = {"micro": 0, "meso": 1, "macro": 2}
        self.feats = {0: pool["micro_feats"], 1: pool["meso_feats"], 2: pool["macro_feats"]}
        self.texts = {0: pool["micro_texts"], 1: pool["meso_texts"], 2: pool["macro_texts"]}
        self.sems = None    # LSA-семантика {level: arr} — опционально (A/B --sem-rerank)
        self.affect = None  # valence/arousal {level: arr} — аффективная петля (v7)
        self.cluster_map = {}
        for ln, ids in clusters.items():
            for j, cid in enumerate(ids):
                cid = int(cid)
                if cid not in self.cluster_map: self.cluster_map[cid] = []
                self.cluster_map[cid].append((lm[ln], j))

    def attach_sems(self, sems):
        self.sems = sems
        # центроиды кластеров в сем-пространстве — для semantic-veto (перекидка кластера)
        self.cluster_sems = {}
        if sems is not None:
            for cid, members in self.cluster_map.items():
                vs = [sems[ln][gi] for ln, gi in members]
                c = np.mean(vs, axis=0)
                nrm = float(np.linalg.norm(c))
                if nrm > 0: c = c / nrm
                self.cluster_sems[cid] = c

    def attach_affect(self, affect):
        self.affect = affect

    def nearest_cluster_by_sem(self, anchor, exclude=()):
        """Ближайший к якорю кластер по сем-центроиду (semantic-veto escape)."""
        best, best_d = None, -2.0
        for cid, c in self.cluster_sems.items():
            if cid in exclude: continue
            dcos = float(np.dot(c, anchor))
            if dcos > best_d: best_d, best = dcos, cid
        return best

    def sample(self, cluster, level, rng=None):
        rng = rng or np.random
        grains = self.cluster_map.get(int(cluster), [])
        if not grains: return None
        same = [g for g in grains if g[0] == level]
        ln, gi = (same[rng.randint(len(same))] if same else grains[rng.randint(len(grains))])
        return {"level": ln, "idx": gi, "text": self.texts[ln][gi], "feat": self.feats[ln][gi]}

    def sample_attractor(self, cluster, level, expected_feat, recent_idx=None, rng=None,
                         expected_sem=None, topic=None, ref=None, pressure=0,
                         sem_w=0.9, topic_w=0.45, floor=0.0, veto=False, sticky_fn=None,
                         aff_anchor=None, aff_w=0.0, q_ref=None, q_w=0.0,
                         recent_sems=None, dup_w=0.0, q_agg="max",
                         hubs=None, hub_w=0.0, no_cite=False):
        """Аттракторная выборка: внутри кластера берём гранулу, чья стилометрия ближе
        всего к ожидаемому направлению z (expected_feat = ctx[-1] + feat_head(z)*3).
        Это и есть «мышление направлением» — z управляет выбором конкретной гранулы.
        Семантика (--sem-rerank): якорь expected_sem (последняя гранула) + дискурсный
        аттрактор topic (быстрый EMA).
        Микроаттрактор (ref+pressure): стабильный якорь темы topic_ref меняется только
        в RELEASE; давление (накопленные изгибы) добавляет восстанавливающую пружину
        к ref — на изгибах вектор удерживается."""
        rng = rng or np.random
        grains = self.cluster_map.get(int(cluster), [])
        if not grains: return None
        same = [g for g in grains if g[0] == level] or grains
        best, best_score, best_cos = None, -1e18, -2.0
        for ln, gi in same:
            d = -float(np.linalg.norm(self.feats[ln][gi] - expected_feat))
            cos = 1.0
            if self.sems is not None and expected_sem is not None:
                s = self.sems[ln][gi]
                cos = float(np.dot(s, expected_sem))
                d -= sem_w * (1.0 - cos)
                if floor > 0.0 and cos < floor:
                    d -= 50.0                      # жёсткий фильтр: кандидат «не по теме»
                if topic is not None:
                    d -= topic_w * (1.0 - float(np.dot(s, topic)))
                if ref is not None and pressure > 0:
                    d -= topic_w * pressure * 0.5 * (1.0 - float(np.dot(s, ref)))
                if q_ref is not None:
                    if q_ref.ndim == 1:
                        d -= q_w * (1.0 - float(np.dot(s, q_ref)))   # q-магнит: точка
                    else:
                        # распределённый магнит: облако направлений вокруг вопроса
                        sims = q_ref @ s
                        if q_agg == "mean":
                            d -= q_w * (1.0 - float(np.mean(sims)))   # центр масс: зерно должно держать ВСЁ облако
                        else:
                            d -= q_w * (1.0 - float(np.max(sims)))    # хоть один член — уже притянут
                if recent_sems and dup_w > 0.0:
                    # анти-повтор: повторяющиеся фразы = почти идентичные эмбеддинги.
                    # Штрафуем только околокопии (cos>0.93), живую вариативность не трогаем.
                    mx = max(float(np.dot(s, r)) for r in recent_sems)
                    d -= dup_w * max(0.0, mx - 0.93) * 10.0
            if hubs is not None and hub_w > 0.0:
                d -= hub_w * float(hubs[ln][gi])   # анти-хаб: конкретика > вода
            if no_cite and (CITE_RE.search(self.texts[ln][gi])
                            or JUNK_RE.search(self.texts[ln][gi])):
                d -= 3.0                           # библиография/инструкции — не ответ
            if recent_idx and (ln, gi) in recent_idx: d -= 2.0   # штраф за недавние повторы
            if sticky_fn is not None: d += sticky_fn(ln, gi)   # document stickiness (decay)
            # аффективная петля (v7): тянем зерно к аффекту предыдущего шага —
            # замкнутый аналог expected_feat для valence/arousal
            if self.affect is not None and aff_anchor is not None:
                da = float(np.abs(self.affect[ln][gi] - aff_anchor).sum())
                d -= aff_w * da
            d += rng.rand() * 0.05                          # лёгкий стохастический tie-break
            if d > best_score: best_score, best, best_cos = d, (ln, gi), cos
        if veto and best_cos < floor and best_cos > -2.0 and self.sems is not None:
            return {"veto": True}   # весь кластер не по теме — навигатору нужен другой кластер
        ln, gi = best
        out = {"level": ln, "idx": gi, "text": self.texts[ln][gi], "feat": self.feats[ln][gi],
               "score": best_score}
        if self.sems is not None: out["sem"] = self.sems[ln][gi]
        if self.affect is not None: out["affect"] = self.affect[ln][gi]
        return out

    def weave(self, stream_steps):
        """Зеркало synthesize_multi: narrative несёт поток, остальные вплетаются по density."""
        primary = stream_steps[0]
        others = stream_steps[1:]
        parts = []
        for i, s in enumerate(primary):
            parts.append(s["text"])
            for steps in others:
                if i < len(steps):
                    o = steps[i]
                    if float(o.get("density", 0)) > 0.55 and o.get("text"):
                        parts.append(o["text"])
        return _join_pieces(parts)


def _join_pieces(parts):
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if not p: continue
        p = re.sub(r"^[QA]:\s*", "", p)   # артефакты qa-доков в пуле
        p = p[0].upper() + p[1:]
        if p[-1] not in ".!?…»:": p += "."
        out.append(p)
    text = " ".join(out)
    text = re.sub(r"\.\.", ".", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


# ══════════════════════════════════════════════════════════════
# TRAINING (зеркало build_training_pairs + train_multi + z-loss)
# ══════════════════════════════════════════════════════════════
def extract_params_from_feats(feat_prev, feat_next):
    """Параметры рендера из Δ стилометрии соседних гранул (индексы см. layout выше)."""
    params = np.zeros(8, dtype=np.float32)
    diff = feat_next - feat_prev
    params[0] = float(np.clip(diff[21] * 10, -1, 1))   # heat ← std_sent_words
    params[1] = float(np.clip(diff[0], -1, 1))          # compression ← log_len_words
    params[2] = float(np.clip(diff[13] * 10, -1, 1))    # intensity ← quest_excl
    params[3] = float(np.clip(diff[5] * 10, -1, 1))     # cohesion ← stopword_ratio
    params[4] = float(np.clip(diff[4] * 10, -1, 1))     # novelty ← hapax_ratio
    params[5] = float(np.clip(diff[2] * 10, -1, 1))     # rhythm ← std_word_len
    params[6] = float(np.clip(diff[24] * 10, -1, 1))    # echo ← bigram_repeat
    params[7] = 0.0                                     # reserved
    return params


def _level_to_all_f_idx(lev, idx, n_micro, n_meso):
    if lev == 0: return min(idx, n_micro - 1)
    elif lev == 1: return n_micro + min(idx, n_meso - 1)
    else: return n_micro + n_meso + idx


# --- 0GLM-X: семантика НА ВХОДЕ (v5_semin) ---------------------------------
# Урок V/W: семантику нельзя извлечь лоссом, если её нет во входе — контекст
# из 32 стилометрических фич информационно слеп к теме. Решение: PCA-проекция
# MiniLM-эмбеддинга гранулы (384→SEM_PROJ_DIM) конкатенируется к стилю в окне
# контекста. z/params/expected_feat остаются чисто стилевыми (звук не трогаем).
SEM_PROJ_DIM = 32

def fit_sem_projection(pool, sems, dim=SEM_PROJ_DIM):
    S = np.concatenate([sems[0], sems[1], sems[2]], axis=0)
    mu = S.mean(axis=0)
    X = S - mu
    C = (X.T @ X) / max(len(X) - 1, 1)
    w, v = np.linalg.eigh(C)
    comps = np.ascontiguousarray(v[:, ::-1][:, :dim], dtype=np.float32)
    outp = str(GLM_DIR / "pool" / "sem_proj_v1.npz")
    np.savez(outp, components=comps, mean=mu.astype(np.float32))
    ev = float(w[::-1][:dim].sum() / max(w.sum(), 1e-9))
    print(f"📐 PCA семантик: {S.shape[1]}→{dim} (дисперсия {ev:.1%}) → {outp}")
    return comps, mu.astype(np.float32)


def load_sem_projection():
    p = GLM_DIR / "pool" / "sem_proj_v1.npz"
    if not os.path.exists(str(p)):
        return None
    d = np.load(str(p))
    return d["components"], d["mean"]


def project_sems(sems, comp, mean):
    return {lv: ((S - mean) @ comp).astype(np.float32) for lv, S in sems.items()}


# --- аффективный канал (директива 2026-08-22): valence/arousal гранулы ------
# valence = compound VADER; arousal = эмоциональная интенсивность (pos+neg,
# буст от !/?). Вход навигатора: [стиль32 | аффект2 | сем-PCA32]. Аудио-слоты
# M пока не тронуты; будущая мапа: arousal→amp/intensity напрямую.
AFFECT_DIM = 2

def load_affect():
    p = GLM_DIR / "pool" / "text_affect_v1.npz"
    if not os.path.exists(str(p)):
        return None
    d = np.load(str(p))
    return {0: d["micro"], 1: d["meso"], 2: d["macro"]}


HUB_CACHE = str(GLM_DIR / "pool" / "text_hub_v1.npz")

# библиография/цитаты: релевантны теме, но мусор в устном ответе (v14)
CITE_RE = re.compile(
    r"(et al\.|\b19\d{2}\b|\b20\d{2}\b|https?://|\bvol\.\s*\d|\bpp\.\s*\d"
    r"|\bjournal of\b|\"[^\"]{12,}\"[^\"]{0,80}\"\")", re.I)
# заготовки-инструкции/оглавления («- Explain how...», «Generate a report»)
JUNK_RE = re.compile(
    r"(^[\s>*-]*[-•·]\s|explain how\b|provide examples?\b|provide data\b"
    r"|generate a\b|write an?\b|\bstep \d+[:.)]?\s|key points?:|conclusion:"
    r"|\bintroduction\b\s*:|\bresponses?\s*:|^\s*[QA][:.)]\s|^\s*\d+\.\s+[A-Z]"
    r"|^\s*(summary|overview|conclusion|introduction|references|table of contents)\s*$"
    # v17: промо/реклама/бойлерплейт внутри тела документов
    r"|get feedback on (grammar|clarity)|check your paper\b|\bessay\b.{0,20}\bwriting service\b"
    r"|\d+\s+words? \(\d+(\.\d+)? pages?\)|listen to the story\b|all things considered\b"
    r"|\bnpr (food|music|news)\b|\bpost(ed)? by [A-Z]|donate (today|now)\b|make a donation\b"
    r"|tax-deductible donation|sign ?-?up for (our |the )?(newsletter|free)|subscribe to (our |the )?(newsletter|podcast|channel)"
    r"|follow us on (facebook|twitter|instagram)|all rights reserved|©\s*\d{4})", re.I | re.M)
# FAQ-заголовки и структурный мусор — режется на входе в пул (v15)
_POOL_HEADER_RE = re.compile(
    r"^\s*([QA][:.)]\s|[\d]+\.\s+[A-Z]|[-•·*]\s+\w|\(\s*\w+\s*\)$)", re.M)
_QW = ("what", "why", "how", "when", "where", "who", "is", "are", "can", "do",
       "does", "did", "should", "would", "will", "could", "which")


def _pool_junk(t: str) -> bool:
    """True → зерно не должно попадать в поле/выдачу (FAQ-заголовки, промо,
    оглавления). Ищем по всему тексту зерна: рекламные вставки живут в середине."""
    if not t.strip():
        return True
    first = t.lstrip().splitlines()[0]
    if _POOL_HEADER_RE.match(first) or JUNK_RE.search(t):
        return True
    words = first.split()
    if (first.endswith("?") and len(words) <= 10
            and words[0].lower().rstrip(",") in _QW):
        return True  # заголовок-вопрос без контекста («What is global warming its causes and effects?»)
    return False


DOC_EMBS_CACHE = GLM_DIR / "pool" / "doc_embs_v1.npz"


def build_doc_embs(pool, sems=None, force=False):
    """Эмбеддинги ВСЕХ документов (норм. среднее сем микро/мезо, cap 300).
    Первый этап ретрива v18: ищем ДОК целиком, а не зерно из 6%-сэмпла."""
    if DOC_EMBS_CACHE.exists() and not force:
        return np.load(DOC_EMBS_CACHE)["embs"]
    sems = sems if sems is not None else build_semantics(pool)
    trajs = pool.get("trajectories", [])
    embs = np.zeros((len(trajs), sems[0].shape[1]), np.float32)
    for ti, traj in enumerate(trajs):
        vs = [sems[int(lv)][int(ix)] for lv, ix in traj if int(lv) <= 1][:300]
        if not vs:
            continue
        v = np.mean(vs, axis=0)
        n = float(np.linalg.norm(v))
        embs[ti] = (v / n).astype(np.float32) if n > 1e-6 else v
    np.savez(str(DOC_EMBS_CACHE), embs=embs)
    print(f"💾 doc-эмбеддинги: {len(trajs)} доков → {DOC_EMBS_CACHE.name}")
    return embs


def compute_hub_scores(engine, force=False):
    """Анти-хаб (v14b): шаблонные фразы («In conclusion…») — хабы СЕМАНТИЧЕСКОГО
    (другие шаблоны), поэтому любой cos-магнит выбирает именно их.
    hub = 0.5·cos(зерно, центроид уровня) + 0.5·(средний cos к 16 ближайшим из
    случайной выборки 4096 зёрен того же уровня). Кэш pool/text_hub_v1.npz
    (при ребилде пула пересобирать вручную!)."""
    import os
    lens = tuple(len(engine.feats[l]) for l in range(3))
    if not force and os.path.exists(HUB_CACHE):
        d = np.load(HUB_CACHE)
        if all(int(d[f"n{l}"]) == lens[l] for l in range(3)):
            return {l: d[f"h{l}"] for l in range(3)}
    rng = np.random.default_rng(7)
    out = {}
    for lv in range(3):
        S = engine.sems[lv]; n = len(S)
        cent = S.mean(0); nc = float(np.linalg.norm(cent))
        cent = cent / nc if nc > 0 else cent
        ref = S[rng.choice(n, size=min(4096, n), replace=False)]
        sims = S @ ref.T                          # n × m
        kk = min(16, sims.shape[1])
        part = np.sort(sims, axis=1)[:, -kk:].mean(1)
        H = (0.5 * (S @ cent) + 0.5 * part).astype(np.float32)
        out[lv] = H
        print(f"   hub[{['micro','meso','macro'][lv]}]: медиана {np.median(H):.3f} "
              f"p90 {np.percentile(H,90):.3f}", flush=True)
    np.savez(HUB_CACHE, **{f"h{l}": out[l] for l in range(3)},
             **{f"n{l}": len(out[l]) for l in range(3)})
    return out


def stitch_narrative(steps, engine=None, seam_cos=0.35):
    """Сшивка потока с разметкой швов (v13): форензика показала — читателя ранят
    не скачки сами по себе (человеческий текст даёт ~35% пар cos<0.3), а БЕЗЗВУКНЫЕ
    стыки фрагментов. Каждый шов (смена origin или провал cos) → граница абзаца."""
    out, prev = [], None
    for s in steps:
        if prev is not None:
            seam = s.get("origin") != prev.get("origin")
            if not seam and engine is not None and engine.sems is not None:
                c = float(np.dot(engine.sems[s["level"]][s["idx"]],
                                 engine.sems[prev["level"]][prev["idx"]]))
                seam = c < seam_cos
            out.append("\n\n" if seam else " ")
        out.append(s["text"].strip())
        prev = s
    return "".join(out)


# --- позиционный канал (v12): «где я в документе» ---------------------------
# Диагноз: кластеризация стёрла документную структуру — навигатор слеп к тому,
# из какого дока зерно и как далеко по нему мы прошли. Позиция в траектории
# (0..1) на входе навигатора: пусть сам выучит дискурсную стадию.
def build_positions(pool, force=False):
    cache = str(GLM_DIR / "pool" / "text_pos_v1.npz")
    lens = {0: len(pool["micro_feats"]), 1: len(pool["meso_feats"]), 2: len(pool["macro_feats"])}
    if not force and os.path.exists(cache):
        z = np.load(cache)
        if len(z["p_micro"]) == lens[0]:
            print(f"📍 Positions cache: {sum(lens.values())} grains")
            return {0: z["p_micro"], 1: z["p_meso"], 2: z["p_macro"]}
    out = {lv: np.full(lens[lv], 0.5, dtype=np.float32) for lv in (0, 1, 2)}
    n_seen = 0
    for traj in pool.get("trajectories", []):
        L = len(traj)
        if L < 2: continue
        for j, (lev, idx) in enumerate(traj):
            if idx < lens[lev]:
                out[lev][idx] = np.float32(j / (L - 1))
                n_seen += 1
    np.savez(cache, p_micro=out[0], p_meso=out[1], p_macro=out[2])
    print(f"📍 Positions: {n_seen} grains из {sum(lens.values())} → {cache}")
    return out


def build_training_pairs(pool, clusters, sems=None, proj_all=None, affect_all=None,
                         lag_rows=0, pos_in_ctx=False):
    """lag_rows>0 (v8 «запаздывающая связь»): к 12 свежим строкам контекста
    добавляется строка-эхо — средняя стилометрия окна [t-2L, t-L), L=lag_rows.
    Это память поля: медленная связь с тем, что было давно.
    pos_in_ctx (v12): +1 колонка — позиция зерна в его документе (j/(L-1)).
    Траектория = последовательность дока, поэтому позиция известна точно."""
    print("\n📐 Building training pairs from trajectories...")
    qa_map = {}
    if os.path.exists(QA_MAP):
        with open(QA_MAP) as f: qa_map = json.load(f)
        print(f"   QA cond map: {len(qa_map)} docs")
    all_f = np.concatenate([pool["micro_feats"], pool["meso_feats"], pool["macro_feats"]])
    n_micro, n_meso = len(pool["micro_feats"]), len(pool["meso_feats"])
    pairs = []
    for traj in pool.get("trajectories", []):
        if len(traj) < CONTEXT_LEN + 2: continue
        # cond запроса: если траектория из QA-дока — все её пары получают cond запроса
        lev0, idx0 = traj[0]
        src = (pool.get("micro_sources") if lev0 == 0 else pool.get("meso_sources") if lev0 == 1 else pool.get("macro_sources"))
        q_feat = None
        if src and idx0 < len(src):
            base = os.path.basename(src[idx0][0])
            if base in qa_map: q_feat = np.array(qa_map[base], dtype=np.float32)
        run_topic = None   # EMA сем гранул [0..k-1] — цель topic_head (инкрементально)
        for k in range(CONTEXT_LEN, len(traj)):
            ctx_feats = []
            for j in range(k - CONTEXT_LEN, k):
                lev, idx = traj[j]
                gi = _level_to_all_f_idx(lev, idx, n_micro, n_meso)
                f32 = all_f[gi]
                if affect_all is not None:
                    f32 = np.concatenate([f32, affect_all[lev][idx]])
                if proj_all is not None:
                    f32 = np.concatenate([f32, proj_all[gi]])
                if pos_in_ctx:
                    f32 = np.concatenate([f32, np.array([j / (len(traj) - 1)], np.float32)])
                ctx_feats.append(f32)
            if lag_rows > 0:
                lo = max(0, k - 2 * lag_rows)
                lag_slice = traj[lo:max(lo, k - lag_rows)]
                if lag_slice:
                    lf = np.mean([all_f[_level_to_all_f_idx(lv_, ix_, n_micro, n_meso)]
                                  for lv_, ix_ in lag_slice], axis=0)
                    extra_w = ctx_feats[0].shape[0] - FEAT_DIM
                    row = np.concatenate([lf, np.zeros(extra_w, np.float32)])
                else:
                    row = np.zeros_like(ctx_feats[0])
                ctx_feats.append(row)
            target_lev, target_idx = traj[k]
            tgt_gi = _level_to_all_f_idx(target_lev, target_idx, n_micro, n_meso)
            prev_lev, prev_idx = traj[k - 1]
            prev_gi = _level_to_all_f_idx(prev_lev, prev_idx, n_micro, n_meso)
            params = extract_params_from_feats(all_f[prev_gi], all_f[tgt_gi])
            ln = ["micro", "meso", "macro"][target_lev]
            cluster_id = int(clusters[ln][min(target_idx, len(clusters[ln]) - 1)])
            density = float(np.clip(all_f[tgt_gi][20] / 40.0, 0, 1))
            # z-target: НАПРАВЛЕНИЕ движения стилометрии (Δfeat), не абсолютный срез —
            # абсолютные фичи текут медленно и дают околонулевой градиент
            delta_next = np.clip(all_f[tgt_gi] - all_f[prev_gi], -3, 3) / 3.0
            pair = {
                "ctx": np.array(ctx_feats, dtype=np.float32),
                "cluster": cluster_id,
                "level": target_lev,
                "params": params,
                "density": density,
                "delta_next": delta_next.astype(np.float32),
                "prev_feat": all_f[prev_gi].astype(np.float32),
                "cond": q_feat if q_feat is not None else np.zeros(FEAT_DIM, dtype=np.float32),
            }
            if affect_all is not None:
                pair["affect_next"] = np.asarray(affect_all[target_lev][target_idx], dtype=np.float32)
            if sems is not None:
                s_prev = sems[prev_lev][prev_idx]
                run_topic = s_prev if run_topic is None else \
                    TOPIC_EMA * run_topic + (1 - TOPIC_EMA) * s_prev
                tn = np.linalg.norm(run_topic)
                tt = run_topic / tn if tn > 0 else run_topic
                pair["sem_next"] = sems[target_lev][target_idx].astype(np.float16)
                pair["topic_next"] = tt.astype(np.float16)
            pairs.append(pair)
    print(f"  ✅ {len(pairs)} pairs" + (f" (+sem/topic targets)" if sems is not None else ""))
    return pairs


class PairDS(Dataset):
    def __init__(self, pairs): self.p = pairs
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        p = self.p[i]
        return (torch.tensor(p["ctx"]), torch.tensor(p["cluster"]),
                torch.tensor(p["level"]), torch.tensor(p["params"]))


class MultiPairDS(Dataset):
    def __init__(self, pairs, n_streams=N_STREAMS):
        self.p = pairs; self.n_streams = n_streams
        self.z16 = np.zeros(SEM_DIM, dtype=np.float16)
        self.a2 = np.zeros(AFFECT_DIM, dtype=np.float32)
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        p = self.p[i]
        stream_idx = np.random.randint(0, self.n_streams)
        cond = p.get("cond")
        if cond is None: cond = np.zeros(FEAT_DIM, dtype=np.float32)
        return (torch.tensor(p["ctx"]), torch.tensor(p["cluster"]),
                torch.tensor(p["level"]), torch.tensor(p["params"]),
                torch.tensor(p.get("density", 0.5), dtype=torch.float32),
                torch.tensor(p["delta_next"]),
                torch.tensor(cond),
                torch.tensor(stream_idx, dtype=torch.long),
                torch.tensor(p.get("sem_next", self.z16), dtype=torch.float32),
                torch.tensor(p.get("topic_next", self.z16), dtype=torch.float32),
                torch.tensor(p.get("affect_next", self.a2), dtype=torch.float32))


def train(model, pairs, n_steps=TRAIN_STEPS):
    ds = PairDS(pairs)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    print(f"\n🔥 TRAINING NAVIGATOR ({n_steps} steps, {DEVICE})")
    print(f"   Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | Pairs: {len(ds)}")
    model.train(); losses = []; t0 = time.time(); step = 0
    while step < n_steps:
        for batch in loader:
            if step >= n_steps: break
            ctx, tgt_c, tgt_l, tgt_p = [b.to(DEVICE) for b in batch]
            cl, lv, pr = model(ctx)
            loss = (F.cross_entropy(cl, tgt_c) + 0.5 * F.cross_entropy(lv, tgt_l)
                    + F.mse_loss(torch.tanh(pr), tgt_p))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item()); step += 1
            if step % 500 == 0:
                avg = np.mean(losses[-500:]); e = time.time() - t0
                print(f"  step {step:5d}/{n_steps}  loss={avg:.4f}  {e:.0f}s  ETA {e/step*(n_steps-step):.0f}s")
    print(f"\n   ✅ {time.time()-t0:.1f}s, loss={np.mean(losses[-100:]):.4f}")
    return model


def train_multi(model, pairs, n_steps=TRAIN_STEPS):
    ds = MultiPairDS(pairs)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    print(f"\n🔥 TRAINING MULTI-NAVIGATOR ({n_steps} steps, {DEVICE})")
    print(f"   Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | Pairs: {len(ds)}")
    model.train(); losses = []; t0 = time.time(); step = 0
    while step < n_steps:
        for batch in loader:
            if step >= n_steps: break
            ctx, tgt_c, tgt_l, tgt_p, tgt_dn, tgt_d, cond, stream_idx, tgt_sem, tgt_top, tgt_aff = \
                [b.to(DEVICE) for b in batch]
            cl, lv, pr, dn, zz, sem_pred, topic_pred, aff_pred = model(ctx, stream_idx=stream_idx, cond=cond)
            loss_c = F.cross_entropy(cl, tgt_c)
            loss_l = 0.5 * F.cross_entropy(lv, tgt_l)
            loss_p = F.mse_loss(pr, tgt_p)
            loss_dn = F.mse_loss(dn.squeeze(-1), tgt_dn)
            # z живёт направлением: cosine (куда движется стилометрия) + MSE (величина шага)
            pred_d = model.feat_head(zz)
            cos = F.cosine_embedding_loss(pred_d, tgt_d,
                                          torch.ones(pred_d.shape[0], device=DEVICE))
            loss_z = 0.25 * (cos + F.mse_loss(pred_d, tgt_d))
            ctx_proj = model.proj(model.feat_enc(ctx)).mean(dim=1)
            loss_at = 0.1 * F.mse_loss(model.attractor_state[stream_idx], ctx_proj)
            # семантика (0GLM-W): КОНТРАСТИВНЫЙ InfoNCE вместо регрессии к точному
            # вектору (регрессия гонялась за средним и ломала навигацию — v2_sem).
            # Задача: ранжировать РЕАЛЬНУЮ следующую гранулу выше негативов батча.
            # Веса уровней — из измеренной связности траекторий (micro .30 /
            # meso .70 / macro .57): где сигнал есть, там и учимся.
            sp = F.normalize(sem_pred, dim=-1)
            tsn = F.normalize(tgt_sem, dim=-1)
            logits_sems = sp @ tsn.t() / 0.1
            tgt_rank = torch.arange(sp.shape[0], device=DEVICE)
            w_lvl = torch.tensor([0.3, 1.0, 0.7], device=DEVICE)[tgt_l]
            loss_sem = (F.cross_entropy(logits_sems, tgt_rank, reduction="none") * w_lvl).mean()
            loss_top = torch.zeros((), device=DEVICE)   # topic_head выведен из лосса
            # аффект в цели (директива 2026-08-22): предсказание val/arousal следующей
            # гранулы формирует аффективную динамику hidden state
            loss_aff = F.mse_loss(aff_pred, tgt_aff)
            loss = (loss_c + loss_l + loss_p + 0.5 * loss_dn + loss_z + loss_at
                    + 0.2 * loss_sem + 0.5 * loss_aff)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item()); step += 1
            if step % 500 == 0:
                avg = np.mean(losses[-500:]); e = time.time() - t0
                print(f"  step {step:5d}/{n_steps}  loss={avg:.4f} (c={loss_c:.3f} l={loss_l:.3f} p={loss_p:.3f} z={loss_z:.3f} semNCE={loss_sem:.3f} aff={loss_aff:.3f})  {e:.0f}s  ETA {e/step*(n_steps-step):.0f}s")
    print(f"\n   ✅ {time.time()-t0:.1f}s, loss={np.mean(losses[-100:]):.4f}")
    return model


# ══════════════════════════════════════════════════════════════
# GENERATION (зеркало generate_multi: стримы + аттракторы + feedback + weave)
# ══════════════════════════════════════════════════════════════
def generate_multi(model, engine, pool, n_steps=24, seed=42, temp=0.8,
                   target_stats=None, noise_inject=0.0, ctx_init=None, boost_clusters=None,
                   sem_cfg=None, trans_matrix=None, ref_sem=None, sem_ctx=None,
                   affect_ctx=None, pos_ctx=None):
    sem_cfg = sem_cfg or {}
    if seed is not None: torch.manual_seed(seed); np.random.seed(seed)
    model.eval()
    print(f"\n📝 MULTI-STREAM TEXT: {n_steps} steps × {N_STREAMS} streams ({', '.join(STREAM_NAMES)})")

    all_f = np.concatenate([pool["micro_feats"], pool["meso_feats"], pool["macro_feats"]])
    # проекции семантик и аффект, выровненные по гранулам (0GLM-X + affect)
    proj_rows = None
    if sem_ctx is not None:
        _nm, _ne = len(pool["micro_feats"]), len(pool["meso_feats"])
        _pa = np.concatenate([sem_ctx[0], sem_ctx[1], sem_ctx[2]], axis=0)
        proj_rows = lambda lev, idx: _pa[_level_to_all_f_idx(lev, idx, _nm, _ne)]
    aff_rows = None
    if affect_ctx is not None:
        aff_rows = lambda lev, idx: np.asarray(affect_ctx[lev][min(int(idx), len(affect_ctx[lev]) - 1)],
                                               dtype=np.float32)
    pos_rows = None
    if pos_ctx is not None:
        pos_rows = lambda lev, idx: np.array([float(pos_ctx[lev][min(int(idx), len(pos_ctx[lev]) - 1)])],
                                             dtype=np.float32)

    def _extra(lv, i):
        parts = []
        if aff_rows is not None: parts.append(aff_rows(lv, i))
        if proj_rows is not None: parts.append(proj_rows(lv, i))
        if pos_rows is not None: parts.append(pos_rows(lv, i))
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    _want = getattr(model, "feat_dim", FEAT_DIM)
    # «запаздывающая связь» (v8): лишняя строка контекста — медленная память поля
    _lag = max(0, int(model.pos.shape[1]) - CONTEXT_LEN) if hasattr(model, "pos") else 0
    _hist = []   # стиль последних 2*LAG шагов narrative → эхо-строка

    def _mem_row():
        if _lag == 0 or len(_hist) < 2 * 12:
            w = FEAT_DIM + (AFFECT_DIM if affect_ctx is not None else 0) \
                + (SEM_PROJ_DIM if sem_ctx is not None else 0)
            return np.zeros(w, np.float32)
        return np.mean(_hist[:12], axis=0)

    if _lag > 0:
        print(f"   🕰 lag-memory: +{_lag} строка (эхо t-{2*12}..t-{12})")

    # аффективный якорь (v7): ФИКСИРОВАННЫЙ mood из затравки контекста.
    # Урок: якорь «последнее зерно» вырождается — 56% микрозёрен имеют ровно
    # нулевой VADER, петля сама затягивается в нейтральную яму.
    mood = None
    if aff_rows is not None and sem_cfg.get("aff_w", 0) > 0:
        if "mood" in sem_cfg:   # явное настроение (QA: аффект вопроса)
            mood = np.asarray(sem_cfg["mood"], dtype=np.float32)[:AFFECT_DIM]
        elif ctx_init is not None and ctx_init.shape[1] >= FEAT_DIM + AFFECT_DIM:
            mood = np.asarray(ctx_init[:, FEAT_DIM:FEAT_DIM + AFFECT_DIM].mean(axis=0),
                              dtype=np.float32)
        else:
            mood = np.zeros(AFFECT_DIM, np.float32)

    # q-магнит (v11): эмбеддинг вопроса как постоянный семантический магнит
    # всей выборки. Урок калибровки: gold-чанки дают cos .31-.50 против
    # вопроса — генерация обязана уметь держаться этой области поля.
    q_vec = sem_cfg.get("q_ref")
    q_agg = sem_cfg.get("q_agg", "max")
    if q_vec is not None:
        q_vec = np.asarray(q_vec, dtype=np.float32)
        if q_vec.ndim == 1:
            n = float(np.linalg.norm(q_vec))
            if n > 0: q_vec = q_vec / n
        else:
            ns = np.linalg.norm(q_vec, axis=1, keepdims=True)
            q_vec = q_vec / np.maximum(ns, 1e-9)

    cond = None
    if target_stats is not None:
        cond = torch.tensor(target_stats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        print(f"   Cond vector: dim={target_stats.shape[0] if hasattr(target_stats, 'shape') else len(target_stats)}")

    stream_steps = [[] for _ in range(N_STREAMS)]
    z_traj = []
    if ctx_init is not None:
        ctx = np.asarray(ctx_init, dtype=np.float32)
        if ctx.shape[1] == FEAT_DIM and _want != FEAT_DIM:
            ctx = np.concatenate([ctx, np.zeros((CONTEXT_LEN, _want - FEAT_DIM), np.float32)], axis=1)
    else:
        _gi = np.random.choice(len(all_f), CONTEXT_LEN, replace=True)
        rows = [all_f[_gi]]
        if _want != FEAT_DIM:
            _nm2, _ne2 = len(pool["micro_feats"]), len(pool["meso_feats"])
            _lv_of = lambda r: 0 if r < _nm2 else (1 if r < _nm2 + _ne2 else 2)
            _loc = lambda r: int(r) if r < _nm2 else (int(r) - _nm2 if r < _nm2 + _ne2 else int(r) - _nm2 - _ne2)
            ex = np.stack([_extra(_lv_of(r), _loc(r)) for r in _gi])
            rows.append(ex)
        ctx = np.concatenate(rows, axis=1)
    if _lag > 0:
        ctx = np.concatenate([ctx, np.zeros((_lag, ctx.shape[1]), np.float32)], axis=0)
    recent_idx = []      # защита от повторов гранул (FIFO)
    recent_sems = []     # анти-повтор фраз: семы последних выборов (FIFO)
    sem_win = [[] for _ in range(N_STREAMS)]   # скользящее окно сем (посл. 3) — якорь связности
    sem_win_aff = [[] for _ in range(N_STREAMS)]   # окно аффекта выбранных зёрен (v7)
    topic = None                    # дискурсный аттрактор (EMA по выбранным гранулам)
    prev_cid = [None] * N_STREAMS   # последний кластер стрима — вход discourse-приорa
    veto_count = [0]
    topic_pressure = [0]   # микроаттрактор: накопленное давление на смену темы
    switches = [0]         # контролируемые RELEASE-переходы темы
    topic_ref = [None]     # стабильный якорь дискурса — меняется только в RELEASE

    # document stickiness: бонус кандидату из текущего исходного документа,
    # затухающий как bonus * gamma^k (k = шагов уже проведено в доке).
    # Связность берётся из самого корпуса: соседние куски одного файла тематически
    # связаны, кластеризация стёрла границы — здесь восстанавливаем их на инференсе.
    sticky_fn = None
    doc_grains = None
    if sem_cfg.get("sticky_bonus", 0) > 0:
        sb, sg = float(sem_cfg["sticky_bonus"]), float(sem_cfg.get("sticky_gamma", 0.75))
        doc_code = {}
        for ln_k, arr in [(0, pool["micro_sources"]), (1, pool["meso_sources"]),
                          (2, pool["macro_sources"])]:
            uniq, codes = {}, np.empty(len(arr), dtype=np.int32)
            for i, rec in enumerate(arr):
                pth = str(rec[0] if hasattr(rec, "__len__") else rec).split("/")[-1]
                codes[i] = uniq.setdefault(pth, len(uniq))
            doc_code[ln_k] = codes
        # инверт: doc_id -> [(ln, gi), ...] — кандидаты через границы кластеров
        doc_grains = {}
        for ln_k in (0, 1, 2):
            for gi_v in range(len(doc_code[ln_k])):
                doc_grains.setdefault(int(doc_code[ln_k][gi_v]), []).append((ln_k, gi_v))
        # v13 «чтение по порядку» (форензика скачков: 36% прыжков — баг-сэмплинг
        # ВНУТРИ дока, средний cos .144): документ = последовательность, не мешок.
        # При sticky-«остаёмся» читаем СЛЕДУЮЩЕЕ зерно траектории этого дока.
        traj_pos, traj_seq = {}, {}
        for _ti, _tr in enumerate(pool.get("trajectories", [])):
            traj_seq[_ti] = _tr
            for _j, (_lv, _ix) in enumerate(_tr):
                traj_pos[(_lv, _ix)] = (_ti, _j)
        cur = [None, 0]   # [doc_id, шагов в доке]
        sticky_fn = lambda: None   # маркер активности для ветки ниже

    # анти-хаб (v14): считаем/грузим хабовость зёрен один раз на прогон
    hubs = None
    if float(sem_cfg.get("hub_w", 0.0)) > 0.0 and engine.sems is not None:
        hubs = compute_hub_scores(engine)

    model.attractor_state.zero_()

    for si in range(n_steps):
        if si % 10 == 0: print(f"\r  [{si}/{n_steps}]", end="", flush=True)
        ct = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        if noise_inject > 0:
            ct = ct + torch.randn_like(ct) * noise_inject

        step_z = []
        for s_idx in range(N_STREAMS):
            prior = None
            if trans_matrix is not None and prev_cid[s_idx] is not None:
                prior = torch.tensor(trans_matrix[prev_cid[s_idx]],
                                     dtype=torch.float32, device=DEVICE)
            result = model.step(ct, stream_idx=s_idx, temp=temp, cond=cond,
                                prior=prior, prior_lam=sem_cfg.get("trans_lam", 0.35))
            # retrieval-буст (0GLM-Q): перекидываем выбор в релевантные запросу
            # кластеры; в warmup-фазе вероятность максимальная (приземлиться в док)
            _bp = None
            if boost_clusters:
                warmup = int(sem_cfg.get("sticky_warmup", 0))
                _bp = (sem_cfg.get("boost_p_warmup", 0.95) if (warmup and si < warmup)
                       else sem_cfg.get("boost_p", 0.6))
            if boost_clusters and result["cluster"] not in boost_clusters and np.random.rand() < _bp:
                result["cluster"] = int(list(boost_clusters)[np.random.randint(len(boost_clusters))])
            # ── дуга ответа (v14): фазы со своими якорями (ввод→развитие→…→вывод).
            # Переключение = жёсткое переякорение topic + RELEASE дока + посадка
            # в ближайший к якорю кластер. Контролируемая структура вместо одного
            # магнитного бассейна, который сжимается в воду.
            if s_idx == 0:
                _arc = sem_cfg.get("arc")
                _arc_land = None
                if _arc and (si == 0 or (si > 0 and si % int(_arc["per_phase"]) == 0)):
                    ph = 0 if si == 0 else (si // int(_arc["per_phase"])) % len(_arc["anchors"])
                    if _arc.get("phase") != ph:
                        a = _arc["anchors"][ph]
                        na = float(np.linalg.norm(a["sem"]))
                        topic = a["sem"] / na if na > 0 else a["sem"].copy()
                        topic_ref[0] = topic.copy()
                        topic_pressure[0] = 0
                        cur[0] = None
                        _arc["phase"] = ph
                        _arc_land = tuple(a["coord"])   # фаза стартует В зерне якоря
                        print(f"\n  🎬 фаза {ph + 1}/{len(_arc['anchors'])}",
                              flush=True)
            # ожидаемая стилометрия следующей гранулы: контекст + предсказанный Δ из z
            with torch.no_grad():
                zt = torch.tensor(result["z"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                exp_delta = model.feat_head(zt).squeeze(0).cpu().numpy() * 3.0
            expected_feat = np.clip(ctx[CONTEXT_LEN - 1][:FEAT_DIM] + exp_delta, -10, 10)
            # якорь связности: last = предыдущая гранула (резкий сигнал);
            # window = среднее 3 (в LSA размывается); recency = 0.6/0.25/0.15;
            # head = предсказание sem_head самой модели (0GLM-V, «модель знает тему»)
            anchor = None
            if sem_cfg.get("anchor") == "head" and "sem_pred" in result:
                anchor = result["sem_pred"]
            elif sem_win[s_idx]:
                mode = sem_cfg.get("anchor", "last")
                if mode == "window":
                    w = np.mean(sem_win[s_idx][-3:], axis=0)
                elif mode == "recency":
                    tail = sem_win[s_idx][-3:]
                    wcs = ([0.6, 0.25, 0.15][-len(tail):])
                    w = np.sum([wc * v for wc, v in zip(wcs, reversed(tail))], axis=0)
                else:
                    w = sem_win[s_idx][-1]
                nrm = float(np.linalg.norm(w))
                if nrm > 0: anchor = w / nrm
            g = engine.sample_attractor(result["cluster"], result["level"],
                                        expected_feat, recent_idx=set(recent_idx),
                                        expected_sem=anchor, topic=topic,
                                        ref=(topic_ref[0] if s_idx == 0 else None),
                                        pressure=(topic_pressure[0] if s_idx == 0 else 0),
                                        floor=sem_cfg.get("floor", 0.0),
                                        veto=sem_cfg.get("veto", False),
                                        sem_w=sem_cfg.get("sem_w", 0.9),
                                        topic_w=sem_cfg.get("topic_w", 0.45),
                                        aff_anchor=mood,
                                        aff_w=sem_cfg.get("aff_w", 0.0),
                                        q_ref=q_vec, q_w=sem_cfg.get("q_w", 0.0),
                                        recent_sems=recent_sems,
                                        dup_w=sem_cfg.get("dup_w", 0.6),
                                        q_agg=q_agg, hubs=hubs,
                                        hub_w=float(sem_cfg.get("hub_w", 0.0)),
                                        no_cite=bool(sem_cfg.get("no_cite", False)))
            if g is not None and g.get("veto") and anchor is not None:
                # semantic-veto: весь кластер не по теме → перекидка в ближайший по смыслу
                alt = engine.nearest_cluster_by_sem(anchor, exclude={result["cluster"]})
                veto_count[0] += 1
                if alt is not None:
                    result["cluster"] = alt
                    g = engine.sample_attractor(alt, result["level"], expected_feat,
                                                recent_idx=set(recent_idx),
                                                expected_sem=anchor, topic=topic,
                                                sem_w=sem_cfg.get("sem_w", 0.9),
                                                topic_w=sem_cfg.get("topic_w", 0.45),
                                                q_ref=q_vec, q_w=sem_cfg.get("q_w", 0.0),
                                                recent_sems=recent_sems,
                                                dup_w=sem_cfg.get("dup_w", 0.6),
                                                q_agg=q_agg, hubs=hubs,
                                                hub_w=float(sem_cfg.get("hub_w", 0.0)),
                                                no_cite=bool(sem_cfg.get("no_cite", False)))
            if g is None or g.get("veto"): continue
            # ── канал-документ (narrative): вероятностный гейт p_stay=sb*sg^k.
            # При «остаёмся» берём лучшее зерно ДОКА по той же метрике (без бонуса);
            # иначе обычный кластерный аттрактор. Гейт нужен потому, что при
            # детерминированном сравнении канал-док выигрывает всегда (база
            # кандидатов ~10× больше кластерной) и sb/γ вырождаются в захлопку.
            if s_idx == 0 and doc_grains is not None and cur[0] is not None \
                    and si >= int(sem_cfg.get("sticky_warmup", 0)) \
                    and g is not None and not g.get("veto"):
                p_stay = float(np.clip(sb * (sg ** cur[1]), 0.0, 0.9))
                # адаптивный гейт: релевантный запросу док липнет сильнее,
                # нерелевантный отпускает навигатор раньше (QA-режим)
                if ref_sem is not None and sem_win[s_idx]:
                    al = float(np.dot(sem_win[s_idx][-1], ref_sem))
                    p_stay *= max(0.0, min(1.0, al))
                if np.random.rand() < p_stay:
                    rec_set0 = set(recent_idx)
                    # ── doc_walk (v13): преемник по траектории дока — порядок автора
                    seq_pick = None
                    last_g = stream_steps[s_idx][-1] if stream_steps[s_idx] else None
                    if last_g is not None and sem_cfg.get("doc_walk", False):
                        tp = traj_pos.get((last_g["level"], last_g["idx"]))
                        if tp is not None:
                            _ti, _j = tp
                            for _jj in range(_j + 1, min(_j + 3, len(traj_seq[_ti]))):
                                _lv, _ix = traj_seq[_ti][_jj]
                                if (_lv, _ix) in rec_set0: continue
                                seq_pick = (_lv, _ix)
                                break
                    if seq_pick is not None:
                        ln_d, gi_d = seq_pick
                        g = {"level": ln_d, "idx": gi_d, "origin": "docwalk",
                             "text": engine.texts[ln_d][gi_d], "feat": engine.feats[ln_d][gi_d],
                             "score": 0.0}
                        if engine.sems is not None:
                            g["sem"] = engine.sems[ln_d][gi_d]
                    elif last_g is not None and sem_cfg.get("doc_walk", False) \
                            and traj_pos.get((last_g["level"], last_g["idx"])) is not None:
                        pass   # док исчерпан → не остаёмся вовсе: работает кластерный аттрактор
                    else:
                        # мешок (fallback): конец дока / нет позиции / doc_walk выключен
                        cand = doc_grains.get(int(cur[0]), [])
                        Fm = np.stack([engine.feats[l][i] for l, i in cand])
                        dd = -np.linalg.norm(Fm - expected_feat, axis=1)
                        if anchor is not None and engine.sems is not None:
                            Sm = np.stack([engine.sems[l][i] for l, i in cand])
                            dd -= float(sem_cfg.get("sem_w", 0.9)) * (1.0 - Sm @ anchor)
                            if topic is not None:
                                dd -= float(sem_cfg.get("topic_w", 0.45)) * (1.0 - Sm @ topic)
                            if recent_sems and float(sem_cfg.get("dup_w", 0.6)) > 0.0:
                                # анти-повтор в мешке: тот же порог околокопий .93
                                Rm = np.stack(recent_sems)            # w×384
                                mx = (Sm @ Rm.T).max(axis=1)          # max cos к недавним
                                dd -= float(sem_cfg.get("dup_w", 0.6)) * 10.0 * \
                                    np.maximum(0.0, mx - 0.93)
                        rec_set = set(recent_idx)
                        dd -= 2.0 * np.fromiter(((l, i) in rec_set for l, i in cand),
                                                bool, len(cand))
                        _hw = float(sem_cfg.get("hub_w", 0.0))
                        if hubs is not None and _hw > 0.0:
                            dd -= _hw * np.fromiter((hubs[l][i] for l, i in cand),
                                                    np.float32, len(cand))
                        if sem_cfg.get("no_cite", False):
                            dd -= 3.0 * np.fromiter(
                                (1.0 if (CITE_RE.search(engine.texts[l][i])
                                         or JUNK_RE.search(engine.texts[l][i])) else 0.0
                                 for l, i in cand), np.float32, len(cand))
                        j = int(np.argmax(dd))
                        ln_d, gi_d = cand[j]
                        if gi_d >= len(engine.feats[ln_d]):   # защита от рассинхрона пулов
                            raise RuntimeError(
                                f"doc-channel OOB: ({ln_d},{gi_d}) >= {len(engine.feats[ln_d])}, "
                                f"cur={cur}, len(cand)={len(cand)}, "
                                f"levels={sorted(set(l for l,_ in cand))}, dd[j]={dd[j]}")
                        g = {"level": ln_d, "idx": gi_d, "origin": "doc",
                             "text": engine.texts[ln_d][gi_d], "feat": engine.feats[ln_d][gi_d],
                             "score": float(dd[j])}
                        if engine.sems is not None:
                            g["sem"] = engine.sems[ln_d][gi_d]
            # посадка фазы дуги: первый шаг фазы = точное зерно якоря
            if s_idx == 0 and _arc_land is not None:
                ln0, gi0 = _arc_land
                if gi0 < len(engine.feats[ln0]):
                    g = {"level": ln0, "idx": gi0, "origin": "arc",
                         "text": engine.texts[ln0][gi0], "feat": engine.feats[ln0][gi0],
                         "score": 0.0}
                    if engine.sems is not None:
                        g["sem"] = engine.sems[ln0][gi0]
                _arc_land = None
            if s_idx == 0 and sticky_fn is not None:
                dc = int(doc_code[g["level"]][g["idx"]])
                cur[0], cur[1] = (cur[0], cur[1] + 1) if dc == cur[0] else (dc, 1)
            recent_idx.append((g["level"], g["idx"]))
            if len(recent_idx) > 64: recent_idx.pop(0)
            if "sem" in g:
                sem_win[s_idx].append(g["sem"])
                recent_sems.append(g["sem"])
                if len(recent_sems) > 8: recent_sems.pop(0)
                released = False
                if topic is None:
                    topic = g["sem"]
                else:
                    # защёлка с гистерезисом (только narrative — он несёт дискурс):
                    # изгиб меряем от СТАБИЛЬНОГО topic_ref (не от дрейфующего EMA —
                    # иначе аттрактор уезжает вместе с кандидатом и изгибы невидимы);
                    # давление копится → пружина к ref усиливается;
                    # RELEASE_AFTER изгибов = настоящая смена темы → переякорение ref
                    if s_idx == 0 and sem_cfg.get("micro"):
                        if topic_ref[0] is None:
                            nrm2 = float(np.linalg.norm(g["sem"]))
                            topic_ref[0] = g["sem"] / nrm2 if nrm2 > 0 else g["sem"]
                        cos_r = float(np.dot(g["sem"], topic_ref[0]))
                        if cos_r < sem_cfg.get("bend", 0.15):
                            topic_pressure[0] += 1
                        else:
                            topic_pressure[0] = max(0, topic_pressure[0] - 1)
                        if topic_pressure[0] >= sem_cfg.get("release", 3):
                            nrm2 = float(np.linalg.norm(g["sem"]))
                            topic_ref[0] = g["sem"] / nrm2 if nrm2 > 0 else g["sem"]
                            topic_pressure[0] = 0
                            switches[0] += 1
                            released = True
                    if not released:
                        topic = TOPIC_EMA * topic + (1 - TOPIC_EMA) * g["sem"]
                        tn = np.linalg.norm(topic)
                        if tn > 0: topic = topic / tn
            if engine.affect is not None and "affect" in g:
                sem_win_aff[s_idx].append(np.asarray(g["affect"], dtype=np.float32))
                if len(sem_win_aff[s_idx]) > 3: sem_win_aff[s_idx].pop(0)
            p = np.clip(result["params"], -1, 1)
            prev_cid[s_idx] = int(result["cluster"])
            _nl = len(engine.feats[g["level"]])
            if g["idx"] >= _nl:
                raise RuntimeError(f"APPEND BAD origin={g.get('origin')} "
                                   f"({g['level']},{g['idx']}) n={_nl} text={g['text'][:60]!r}")
            # ВАЖНО: level зерна — из g (факт), не result (предсказание навигатора):
            # sample_attractor может отдать гранулу другого уровня (fallback/doc-канал),
            # и пара (level, idx) должна указывать на реальное место в пуле.
            stream_steps[s_idx].append({
                "cluster": result["cluster"], "level": g["level"],
                "idx": g["idx"],
                "origin": g.get("origin", "cluster"),
                "density": float(result["density"]),
                "heat": float(p[0]), "compression": float(p[1]), "intensity": float(p[2]),
                "cohesion": float(p[3]), "novelty": float(p[4]), "rhythm": float(p[5]),
                "echo": float(p[6]),
                "text": g["text"], "feat": g["feat"], "z": result["z"],
                "affect": (np.asarray(engine.affect[g["level"]][g["idx"]], dtype=np.float32)
                           if engine.affect is not None else None),
            })
            step_z.append(result["z"])

        model.update_attractors(ct, stream_idx=list(range(N_STREAMS)), temp=temp)

        if stream_steps[0]:
            last = stream_steps[0][-1]
            ex = _extra(last["level"], last["idx"])
        else:
            ex = np.zeros(max(_want - FEAT_DIM, 0), np.float32)
        base_fb = np.asarray(last["feat"] if stream_steps[0]
                             else all_f[np.random.randint(len(all_f))], dtype=np.float32)
        if base_fb.shape[0] + ex.shape[0] < _want:
            ex = np.concatenate([ex, np.zeros(_want - FEAT_DIM - ex.shape[0], np.float32)])
        if _lag > 0:
            ctx[:CONTEXT_LEN] = np.roll(ctx[:CONTEXT_LEN], -1, axis=0)
            ctx[CONTEXT_LEN - 1] = np.concatenate([base_fb, ex])
            _hist.append(base_fb)
            if len(_hist) > 24: _hist.pop(0)
            ctx[CONTEXT_LEN:] = _mem_row()[None, :]
        else:
            ctx = np.roll(ctx, -1, axis=0)
            ctx[-1] = np.concatenate([base_fb, ex])

        z_traj.append(np.mean(step_z, axis=0) if step_z else np.zeros(Z_DIM, dtype=np.float32))

    print(f"\r  Generating text...")
    if veto_count[0]: print(f"   semantic-veto: {veto_count[0]} кластерных перекидок")
    if switches[0]: print(f"   🔄 topic switches (RELEASE): {switches[0]}")
    text = engine.weave(stream_steps)
    z_arr = np.array(z_traj, dtype=np.float32)
    return text, z_arr, stream_steps


def build_semantics(pool, force=False):
    """Плотная семантика гранул: MiniLM all-MiniLM-L6-v2 (384d, L2-норма).
    Замена LSA-64: у LSA косинус между любыми прозами 0.4-0.6 — темы неразличимы,
    изгибов нет, защёлке не за что цепляться. Кэш pool/text_sem_mini_v1.npz."""
    import os
    cache = str(GLM_DIR / "pool" / "text_sem_mini_v1.npz")
    lens = {0: len(pool["micro_texts"]), 1: len(pool["meso_texts"]), 2: len(pool["macro_texts"])}
    if not force and os.path.exists(cache):
        z = np.load(cache)
        if len(z["sem_micro"]) == lens[0] and z["sem_micro"].shape[1] == SEM_DIM:
            print(f"🧠 MiniLM cache: {sum(lens.values())} grains × {SEM_DIM}d")
            return {0: z["sem_micro"], 1: z["sem_meso"], 2: z["sem_macro"]}
    from sentence_transformers import SentenceTransformer
    texts = [pool["micro_texts"], pool["meso_texts"], pool["macro_texts"]]
    all_t = texts[0] + texts[1] + texts[2]
    print(f"🧠 MiniLM: loading model...")
    st = SentenceTransformer("all-MiniLM-L6-v2", device=str(DEVICE))
    print(f"🧠 MiniLM: encoding {len(all_t)} grains on {DEVICE}...")
    sem = st.encode(all_t, batch_size=256, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True)
    out = {0: sem[:lens[0]], 1: sem[lens[0]:lens[0]+lens[1]], 2: sem[lens[0]+lens[1]:]}
    np.savez(cache, sem_micro=out[0], sem_meso=out[1], sem_macro=out[2])
    print(f"   ✅ saved → {cache}")
    return out


def build_transition_matrix(pool, clusters):
    """Discourse-грамматика из корпуса: P(cluster_next | cluster_prev) по реальным
    траекториям документов. Лаплас +1 → нулевых запретов нет (soft по построению)."""
    import os
    cache = str(GLM_DIR / "pool" / "trans_v1.npz")
    N = N_CLUSTERS
    if os.path.exists(cache):
        z = np.load(cache)
        if z["T"].shape == (N, N):
            print(f"🔗 Transition matrix cache {N}×{N}")
            return z["T"]
    C = {off: np.array(clusters[ln]) for off, ln in ((0, "micro"), (1, "meso"), (2, "macro"))}
    counts = np.ones((N, N), dtype=np.float32)   # Лаплас +1
    n_obs = 0
    for traj in pool.get("trajectories", []):
        prev_cid = None
        for lev, idx in traj:
            arr = C.get(int(lev))
            if arr is None or int(idx) >= len(arr):
                prev_cid = None; continue
            cid = int(arr[int(idx)])
            if prev_cid is not None and prev_cid != cid:
                counts[prev_cid, cid] += 1.0
                n_obs += 1
            prev_cid = cid
    T = counts / counts.sum(axis=1, keepdims=True)
    np.savez(cache, T=T)
    print(f"🔗 Transition matrix: {n_obs} переходов корпуса → {N}×{N} (Лаплас)")
    return T


def safe_load(model, path):
    """Загрузка с деградацией: ключи с несовпадающей формой (напр. cond_proj 4→32)
    сбрасываются в инициализацию вместо падения."""
    sd = torch.load(path, map_location=DEVICE, weights_only=False)["model_state"]
    msd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    dropped = sorted(set(sd) - set(filtered))
    if dropped: print(f"   ⚠️ reinit (shape mismatch): {dropped}")
    model.load_state_dict(filtered, strict=False)
    return model


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    p.add_argument("--gen-steps", type=int, default=24, help="Generation steps (each = 6 stream picks)")
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--refresh", action="store_true", help="Rebuild pool from scratch")
    p.add_argument("--max-docs", type=int, default=None, help="Limit corpus size")
    p.add_argument("--clusters", type=int, default=None, help="Override N_CLUSTERS")
    p.add_argument("--corpus-only", action="store_true", help="Use only 0glm/corpus (clean corpus-zero decode)")
    p.add_argument("--rescan", action="store_true", help="Ignore filelist cache")
    p.add_argument("--target-diversity", type=float, default=None)
    p.add_argument("--target-punct", type=float, default=None)
    p.add_argument("--target-sentlen", type=float, default=None)
    p.add_argument("--target-stopword", type=float, default=None)
    p.add_argument("--noise-inject", type=float, default=0.0)
    p.add_argument("--sem-rerank", action="store_true", help="LSA-семантика: связность темы на инференсе (A/B)")
    p.add_argument("--model", type=str, default=None, help="Path to trained model .pt")
    p.add_argument("--model-out", type=str, default=None, help="Куда сохранить чекпойнт (напр. v2_sem.pt)")
    args = p.parse_args()

    global N_CLUSTERS
    if args.clusters: N_CLUSTERS = args.clusters
    t0 = time.time()

    if os.path.exists(POOL_CACHE) and not args.refresh:
        print(f"📦 Loading pool: {POOL_CACHE}")
        pool = load_pool()
    else:
        if args.rescan and os.path.exists(FILELIST_CACHE):
            os.remove(FILELIST_CACHE)
        # --corpus-only = ТОЛЬКО вычищенный corpus/ (qa-доки дают мусорные
        # «Q:/A:»-зёрна в кластерах и генерациях); corpus_0agi = прошедшие
        # фильтр consec>=.45 доки из 0agi/corpus (v10-расширение поля);
        # corpus_0agi_big = шарды 1-5 с чисткой мусора на уровне корпуса (v15)
        dirs = [str(GLM_DIR / "corpus")] if args.corpus_only else SCAN_DIRS
        for extra_name in ("corpus_0agi", "corpus_0agi_big"):
            extra = GLM_DIR / extra_name
            if args.corpus_only and extra.exists():
                dirs.append(str(extra))
        files = scan_text(dirs)
        print(f"   {len(files)} docs")
        if args.max_docs: files = files[:args.max_docs]
        if not files: return
        pool = extract_all(files)
        save_pool(pool)

    print(f"\n📊 μ={len(pool['micro_feats'])} σ={len(pool['meso_feats'])} Ω={len(pool['macro_feats'])}")

    if os.path.exists(CLUSTERS_CACHE) and not args.refresh:
        with open(CLUSTERS_CACHE) as f: clusters = json.load(f)
        print(f"🔬 Clusters cache: {len(set(clusters['micro']))} ids")
    else:
        clusters = build_clusters(pool)

    engine = TextGrainEngine(pool, clusters)
    # семантики для обучения (0GLM-V): если MiniLM-кэш есть — цели sem/topic в парах
    train_sems = None
    if os.path.exists(str(GLM_DIR / "pool" / "text_sem_mini_v1.npz")):
        train_sems = build_semantics(pool)
    pairs = build_training_pairs(pool, clusters, sems=train_sems)

    model_out = args.model_out or MODEL_MULTI_CACHE
    if not args.generate_only and pairs:
        model_ms = MultiNavigator().to(DEVICE)
        if os.path.exists(model_out):
            print(f"📦 Loading multi-model (safe): {model_out}")
            safe_load(model_ms, model_out)
        model_ms = train_multi(model_ms, pairs, n_steps=args.train_steps)
        torch.save({"model_state": model_ms.state_dict()}, model_out)

    model_path = args.model if args.model else model_out
    model_ms = MultiNavigator().to(DEVICE)
    if os.path.exists(model_path):
        print(f"📦 Loading multi-model (safe): {model_path}")
        safe_load(model_ms, model_path)

    target_stats = None
    if any(x is not None for x in [args.target_diversity, args.target_punct, args.target_sentlen, args.target_stopword]):
        # cond теперь полный 32-dim стилометрический вектор — заполняем по layout индексам
        target_stats = np.zeros(FEAT_DIM, dtype=np.float32)
        if args.target_diversity is not None: target_stats[3] = args.target_diversity
        if args.target_stopword is not None: target_stats[5] = args.target_stopword
        if args.target_punct is not None: target_stats[10] = args.target_punct
        if args.target_sentlen is not None: target_stats[20] = args.target_sentlen

    if args.sem_rerank:
        engine.attach_sems(build_semantics(pool))
    text, z_arr, stream_steps = generate_multi(model_ms, engine, pool, n_steps=args.gen_steps,
        seed=args.seed, temp=args.temperature, target_stats=target_stats,
        noise_inject=args.noise_inject)

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_md = os.path.join(OUT, f"glm_{ts}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# 0GLM generation\n\n")
        f.write(f"- seed: {args.seed} · temp: {args.temperature} · steps: {args.gen_steps} · clusters: {N_CLUSTERS}\n")
        f.write(f"- attractor norms: {[round(float(model_ms.attractor_state[s].norm()), 2) for s in range(N_STREAMS)]}\n\n")
        f.write(text + "\n")
    np.savez(os.path.join(OUT, f"glm_{ts}_z.npz"), z=z_arr,
             attractors=model_ms.attractor_state.cpu().numpy())
    print(f"\n✅ {out_md}")
    print(f"✅ z-trajectory: {OUT}/glm_{ts}_z.npz  shape={z_arr.shape}")
    print("\n" + "─" * 60 + "\n" + text[:1200] + ("\n…" if len(text) > 1200 else "") + "\n" + "─" * 60)
    print(f"⏱️ {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
