# Granular Brain — one-file semantic navigator

A small language model with **all knowledge baked into a single 1.9 GB file** (`brain.pt`).
At inference: no retrieval, no corpus, no external indexes — the field navigates baked granules
directly, and every sentence stays traceable to `(document, position)` in the source corpus.

**Weights on Hugging Face: [0penAGI/0glm](https://huggingface.co/0penAGI/0glm)**

> ⚠️ **Scope.** Trained on a deliberately small research corpus (~16K documents). Within that
> material it speaks coherently; outside it has nothing to navigate. The full bake pipeline is
> here — rebuild the artifact from your own texts.

## Pipeline

```
granular_text_field.py   core library: pool, semantics, oscillatory field, canonization
gen_demos.py             teacher demonstration triples (correction moments)
train_brain.py           distills the teacher into the planner (+ doc-discriminative negatives)
bake_brain.py            packs weights + granule bank + masks → single brain.pt
brain_chat.py            inference: --ask Q or interactive --chat
```

```bash
pip install -r ../requirements.txt

python brain_chat.py --brain brain.pt --ask "What causes El Nino?"
python brain_chat.py --brain brain.pt --chat        # session mode
```

### Rebuild from your own corpus

```bash
python gen_demos.py      # teacher demos from your granulated pool
python train_brain.py    # train the planner
python bake_brain.py     # bake brain.pt
```

## Known architectural limit

The planner picks the next granule locally (context + question). Coherent long single-passage
answers are where a document-trajectory teacher still wins; a document-level plan token is the
declared next step.
