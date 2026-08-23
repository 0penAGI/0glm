# 0GLM — Granular Language Model

> **On the name:** *Granular Language Model* describes what the system does today.
> The same letters also read **Granular Living Model** — the class of system being built:
> grains as living states of a navigable field, language just one readout among possible
> projections. One name, current fact and declared direction.

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

**Landing page with verbatim examples and a real trace excerpt:
[0penagi.github.io/0glm](https://0penagi.github.io/0glm/)**

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

### Ride is not "retrieve chunks and glue them"

Retrieval exists in 0GLM — but only to choose **where to enter** and **when to leave**. No text
is recombined at answer time:

| | classic RAG | 0GLM ride |
|---|---|---|
| unit of selection | a chunk picked by similarity | a document trajectory, entered at a chosen grain |
| order of material | chunks re-ranked and glued by score | the **author's own sentence order**, verbatim |
| between-piece transitions | every boundary is a stitch the decoder must survive | boundaries exist only between phases; inside a phase there are **zero** join decisions |
| where coherence comes from | hope: prompt instructions + decoder skill | inheritance: humans already wrote it in order |
| failure mode | plausible-looking collage of fragments | visible, traceable: wrong document, wrong entry point |

The consequence is practical, not just aesthetic: when a ride answer goes wrong, it is wrong in
exactly one inspectable place — a bad anchor, a premature drift-gate exit, a foreign family —
each of which is logged by `--trace`. A collage has no single place to be wrong in.

### One field, many projections

A grain is not merely a unit of text — it is a **state in a dynamic field**, with position
(stylometry), semantics, affect, and learned transition dynamics. Language is the projection
we currently read out: stitching selected grains renders *text*; mapping the same transitions
through an audio bridge renders *sound*. One trajectory, multiple readouts:

```
corpus → grains (states) → navigator (trajectory dynamics) ─┬─► text   (stitch_narrative)
                                                            └─► audio  (M_text_to_audio)
```

This is why the z-result matters beyond audio: once a predictive head is used for its
*direction* rather than its magnitude (magnitude regresses to the mean; direction stayed
informative at cos .607, 19/22 boundaries positive), any projection can be steered by the
same signal. Whether this framing extends beyond text and sound — to planning traces,
gesture, music — is the open promise of the approach.

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

### Evidence inventory

Everything we claim above, tied to a test that actually ran:

| # | evidence | protocol | outcome |
|---|---|---|---|
| 1 | Ride coherence beats legacy stack | paired blind reading, 2 questions | ride preferred 2/2 **despite worse retrieval scores** |
| 2 | Out-of-domain battery | 8 topics outside climate family, blind reading | 5 clean, 2 soft within-family drift, 2 polysemy misses |
| 3 | Polysemy root cause | targeted harvest +4,067 docs, retrain | python-dict .319→.490, resume .317→.454; arctic byte-identical |
| 4 | Mechanism-opening retrieval | mean/max cosine of answer to question | arctic .421/.725 (ice-albedo), ocean .440/.762 (ENSO experts) |
| 5 | Family gate safety | surgical A/B, identical seed | spam family eliminated; arctic output unchanged byte-for-byte |
| 6 | z knows document dynamics | predicted vs actual style direction | cos .607 mean, 19/22 boundaries positive |
| 7 | z-audio preference | randomized blind pairs, 2 singles + 5-topic series | 7/7 preferred z-steered render (single-listener panel) |
| 8 | Negative results kept | entry-policy and blend-mode experiments | runway-pref rejected (lost sharp late anchors); value-blend rejected (muted dynamics) |

## The dataset is a first-class component

In most language systems the corpus is fuel you burn once during training. In the **current
build** of 0GLM it stays load-bearing — retrieval runs at answer time over every document.
This is a stage of the project, not an article of faith: nothing in the architecture forbids
baking the field into weights (see the open-questions table below), and that is exactly the
next frontier. Until then:

- **Retrieval runs over every document at answer time.** Answer quality is bounded by corpus
  coverage *per question*, not by parameter count. The doc-first index scans all 16,535 docs;
  nothing about the mechanism changes at 165k or 1.65M.
- **Density is correctness.** Polysemous queries fail not because the model is weak but because
  thin clusters let a wrong-family document win. We demonstrated cause and cure: harvest more
  of the right material and the miss flips (.319→.490) while control topics stay byte-identical.
- **Grains inherit author structure.** Trajectories are read from how humans actually ordered
  sentences — so a richer, better-edited corpus makes generation measurably more coherent.

The current corpus is deliberately small: it is a complete, honest end-to-end demonstration,
not a scaled system waiting to be evaluated. All evidence in the table above was collected on
it — which means the evaluation methodology, not just the checkpoint, is reproducible on a
laptop today.

**In the current build, what the system answers depends on what you feed it.** With our corpus
it speaks climate mechanics, Python documentation, resume practice, sleep science and
black-hole thermodynamics — because that is what it has read. Point the same pipeline at
medicine, law or your private notes and the answers come from there, with the same
traceability. And once the field is packed into weights, the live corpus dependency will
loosen — the evidence and selection machinery, however, stay the same either way.

### Scaling properties

| component | cost as corpus grows | bottleneck? |
|---|---|---|
| granulation + features | linear in text volume | no |
| grain clustering (MiniBatchKMeans) | single pass, online | no |
| MiniLM semantics | one-time embedding, cached | disk/speed only |
| navigator training | linear in observed transitions | minutes→hours range |
| doc-first retrieval | one vector scan over docs, per question | no |
| ride inference | bounded by steps × phase size | no |

Nothing in the design requires rethinking at two orders of magnitude more text; the honest
statement is simply that beyond-16k behavior is projected from mechanisms, not yet measured.

### What we do *not* claim

- No dialogue memory yet; each `--ask` is independent.
- The attractor-mode stack is retained as archive/ab-baseline, not a competitor claim.
- Listening panels and reading panels so far have one member (the author). Numbers above are
  reproducible; preference verdicts await independent raters.
- No standard benchmark suite yet — evaluation is task-native (retrieval cosines, blind pairs).

## Open questions, closed

Every known hole in the current build, stated and dispositioned — so nobody has to dig for them:

| question | status | detail |
|---|---|---|
| Is coherence real or metric-gamed? | **closed** (text) | blind reading preferred ride 2/2 at *worse* cosine; metrics alone were blind to it |
| Polysemy misses (python/resume) | **closed** | root cause = corpus density; fixed by harvest, controls byte-identical |
| Why did the navigator's z look "dead"? | **closed** | ride is retrieval-path by design; z repurposed as audio-direction signal, 7/7 blind preference |
| Does prediction regress to the mean? | **closed** | yes — use heads' *direction*, never magnitude; documented as a general field lesson |
| Independent listening panel | **open** | all audio verdicts are single-listener; protocol (`zab_*_A/B.wav` + key) published for replication |
| Dialogue memory across asks | **open** | each `--ask` independent; session state is next front |
| Interactive chat mode | **open** | CLI cycle only today |
| Task trajectories (reasoning, not prose) | **open** | same machinery targeted at reasoning traces ("0agi") |
| Family gate inherits anchor-1 error | **stated** | if anchor #1 misses by polysemy, arc follows its family; planned: verify candidate family against the question independently |
| Music / image / video targets | **stubbed** | manifest schema carries them; only sonification implemented |
| Legacy attractor stack | **archival** | kept for A/B baselines, not a competing claim |
| Baking the field into weights | **planned, not done** | today answers route through the live corpus at answer time; packing trajectories + selection policy into parameters is the declared next stage — until it ships, dataset-dependence is a build-stage property, not an architectural promise |
| Beyond-16k behavior | **projected** | mechanisms scale per table above; not yet measured at larger corpus |

## Roadmap

1. **Task trajectories (0agi)** — the same machinery over reasoning traces instead of prose.
2. **Bake the field into weights** — amortize document trajectories and the selection policy
   into parameters, so coherence survives without a live corpus lookup at answer time.
   The testable hypothesis, stated plainly: *can explicitly observable navigation dynamics
   be turned into a parametric system without losing provenance and controllability?*
   If yes, 0GLM stops being a text generator with good logging and becomes something more
   general: a navigable field whose readouts include language.
3. Independent listening/reading panels; the zab protocol is published for replication.
4. Dialogue memory across asks; interactive session mode.
5. Corpus growth along the density axis that already proved causal (polysemy cure).
6. Multimodal targets already stubbed in the manifest schema (music/image/video).

Contributions welcome on all of them. And a practical note for anyone trying the pipeline
today: in the pre-baking build, **the dataset you bring is the answers you get** — which is
also the cleanest way to see the mechanism work.

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

#    or talk to it — session mode with dialogue memory:
python audio_bridge.py --chat --ride --seed 7 --steps 24 --arc 4
#    follow-ups keep context ("why do scientists watch it so closely?" resolves
#    pronouns from the previous answer); /new resets, /exit saves a transcript

# 3) ask — traced, coherent, and audible
python audio_bridge.py --ask "How does climate change affect the ocean?" \
                       --ride --trace --seed 7 --steps 24 --arc 4
# → answer text, sonify_qa_*.json manifest, WAV, _readalong.txt, _trace.json
```

Audio flags: `--no-z-audio` reverts transitions to feats-only; `--z-audio` additionally
renders a randomized blind A/B pair (`zab_*_A.wav/_B.wav` + `_key.json`) for listening tests.

## Heritage

The audio half descends from the author's earlier oscillator work:
[HuggingFace](https://huggingface.co/0penAGI/0MGE) ([GitHub](https://github.com/0penAGI/0MGE)) —
text rendered through frequency-band dynamics. 0GLM keeps that physical intuition but replaces
band-mixing with *navigation over learned grain trajectories*.

## Status

Research prototype. Two days of disciplined iteration produced the current architecture; the
design log (decisions, dead ends, negative results) is kept deliberately — including what did
*not* work (runway-pref entry policy: rejected for losing sharp late entries).

A landing page with verbatim example outputs and a real trace excerpt is live at
[0penagi.github.io/0glm](https://0penagi.github.io/0glm/) (served from
[`index.html`](index.html) in this repo).

## License

MIT — see [LICENSE](LICENSE).
