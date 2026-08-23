# 0GLM — Granular Language Model

**Coherent speech grown on a laptop from oscillator-field grain dynamics — with traceability by design.**

0GLM is not another transformer. It is an attempt to build a small language system where
*interpretability is an architectural property*, not a post-hoc analysis. Every fragment of
generated text can be traced to a named source document, a named entry point, and a logged
selection reason — before, during, and after generation.

---

## TL;DR

| | |
|---|---|
| **What it is** | A navigable "field" of ~1.27M text grains (stylometric oscillators) with a trained navigator that answers questions by *riding real document trajectories* |
| **Headline result** | Connected, on-topic speech on a 16.5k-document corpus, trained end-to-end on a MacBook GPU in minutes — with per-step provenance for every sentence |
| **Second modality** | The same trajectory renders to audio; a predictive `z` signal steers transition direction (preferred in 7/7 blind listens by the author) |
| **Key property** | Traceability **by design**: `--trace` prints candidates, anchors, gates and rejection reasons for every step |

---

## Why

Transformers are black boxes; mechanistic interpretability spends whole teams trying to read
them *after* training. We invert the premise: make the generation mechanism legible up front.

The unit of language here is not a token but a **grain**: a contiguous chunk of text described
by a compact stylometric feature vector (32 dims: rhythm, lexical density, punctuation regime,
connectives, ...), enriched with semantics (MiniLM-L6, PCA-projected) and affect (valence /
arousal). Grains are clustered into a three-level hierarchy:

```
µ  micro-grains   (clause/sentence scale)
σ  meso-grains    (paragraph scale)
Ω  macro-grains   (document-section scale)
```

A transformer-based **navigator** is trained on millions of observed grain-to-grain transitions
to predict the next grain's cluster/level/register and its stylometric delta (`z`). Generation
then becomes navigation through this field instead of autoregressive token sampling.

## Architecture

```
 question ──► doc-first retrieval ──► arc anchors ──► RIDE along document trajectories
              (all 16,535 docs)      (entry points)        │
                                                           │  same trajectory
                                                           ▼
                                        TEXT ◄── stitch_narrative ──► AUDIO (sonification)
                                       (verbatim            transition targets ← M_text_to_audio
                                        author grains)      direction blend ← navigator z-head
```

Answer construction layers (v17→v19):

1. **Doc-first retrieval** — embed the question once, rank *documents* (not grains), take top-16.
2. **Arc anchors** — pick trajectory entry points spread across distinct document families.
3. **Ride** — inside each phase, follow the *author's own order* of that document
   (coherence is inherited from human writing, not enforced by penalties).
4. **Family gate** — every candidate must stay within the anchor's semantic family.
5. **Drift gate** — leave a phase when the local window's cosine falls below floor.
6. **Budget carryover** — unused steps transfer (capped) to the next phase.
7. **z-audio bridge** — the navigator's predicted stylometric direction blends into audio
   transition targets; amplitude always factual (predictions regress to the mean — use their
   *direction*, not their magnitude).

## Traceability by design

`--trace` emits, for every answer:

- every arc anchor: which document, which grain, why chosen (tail-cosine of the question);
- every step: source doc id, grain level/index, first words;
- every skipped candidate and *why* (below threshold / foreign family / duplicate / lost MMR);
- every drift-gate event with window cosine and floor;
- budget carryover events between phases.

Post-generation, any sentence maps back to `(document, position)` in the corpus. This is not
probing a trained network afterwards — the navigation layer was built to be read.

## Results (honest)

All blind tests were single-author panels; we flag exactly where that limits claims.

**Text coherence.** In paired blind reading, ride-mode answers were preferred over the legacy
attractor stack in 2/2 early comparisons *despite worse retrieval scores* — metrics alone were
blind to what readers saw. An 8-topic out-of-domain battery (casual, tasks, code): 5/8 clean,
2 soft within-family drift, 2 polysemy misses.

**Polysemy fixed by corpus density, not patches.** "Python dict" retrieved a GNU Dico server
manual (cos .319); "resume advice" retrieved questionnaire/essay spam (.317). Targeted
harvesting from raw shards (+4,067 docs) retrained the field: .490 and .454 respectively, while
Arctic stayed byte-identical (.421). Root cause treated, symptom untouched.

**Retrieval opens mechanisms.** Arctic amplification answers start from ice-albedo mechanics
(mean cos .421 / max .725); ocean answers reach ENSO expert fragments (.440/.762).

**Audio modality.** The same trajectory drives sound. A directional z-bridge became the audio
default after 7/7 blind preferences (2 singles + 5-topic series) by the author listener.
Diagnostics: predicted style direction matches actual document motion at mean cosine .607
(19/22 boundaries positive) — the field has learned document dynamics. *Caveat: n=1 listener;
independent panel is open.*

**Scale & cost.** Corpus 16,535 docs → 1,270,597 grains; navigator trains on a laptop
(M-series GPU, mixed MPS/CPU) in ~15 minutes to loss 6.42. No cluster, no API bills.

### What we do *not* claim

- No dialogue memory yet; each `--ask` is independent.
- The attractor-mode stack is retained as archive/ab-baseline, not a competitor claim.
- Listening panels and reading panels so far have one member (the author). Numbers above are
  reproducible; preference verdicts await independent raters.
- No standard benchmark suite yet — evaluation is task-native (retrieval cosines, blind pairs).

## Roadmap

1. **Task trajectories (0agi)** — the same machinery over reasoning traces instead of prose.
2. Independent listening/reading panels; publish the zab protocol for replication.
3. Dialogue memory across asks; interactive session mode.
4. Multimodal targets already stubbed in the manifest schema (music/image/video).

## Repository layout

```
granular_text_field.py   core: granulation, clusters, caches, navigator, ride engine
audio_bridge.py          CLI: --ask Q --ride [--trace] [--no-z-audio] → text + WAV + manifests
train_field_v15.py       trains the navigator (v15_field) on the grain pool
build_corpus_big.py      shard harvesting → corpus (md5 dedup, topic gates)
build_corpus_topics.py   targeted density harvesting (prog/resume/general buckets)
build_corpus_glm.py      alternative corpus builder variant
rebuild_caches.py        rebuild derived caches when the pool changes
docs/                    architecture notes
```

Data artifacts (corpus, pools, checkpoints) are intentionally **not** committed; see below.

## Quickstart

```bash
pip install -r requirements.txt

# 1) build corpora from your own document shards (see scripts' headers)
python build_corpus_big.py --help

# 2) granulate + caches + train the navigator
python train_field_v15.py --help

# 3) ask — traced, coherent, and audible
python audio_bridge.py --ask "How does climate change affect the ocean?" \
                       --ride --trace --seed 7 --steps 24 --arc 4
# → answer text, sonify_qa_*.json manifest, WAV, _readalong.txt, _trace.json
```

Audio flags: `--no-z-audio` reverts transitions to feats-only; `--z-audio` additionally
renders a randomized blind A/B pair (`zab_*_A.wav/_B.wav` + `_key.json`) for listening tests.

## Heritage

The audio half descends from the author's earlier oscillator work:
[0MGE](https://huggingface.co/0penAGI/0MGE) — text rendered through frequency-band dynamics.
0GLM keeps that physical intuition but replaces band-mixing with *navigation over learned grain
trajectories*.

## Status

Research prototype. Two days of disciplined iteration produced the current architecture; the
design log (decisions, dead ends, negative results) is kept deliberately — including what did
*not* work (runway-pref entry policy: rejected for losing sharp late entries).
