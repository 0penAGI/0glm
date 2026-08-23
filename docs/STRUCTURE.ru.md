# 0glm — structure.md

Карта проекта. Что где лежит и зачем.

## Дерево

```
0glm/
├── readme.md                  # фрейм, закон, архитектура v15–v18b, честность
├── PLAN.md                    # история решений и отрицательных результатов
├── structure.md               # ← этот файл
│
├── granular_text_field.py     # ЯДРО ПОЛЯ: scan_text / extract_feat_from_text
│                              #   (32 dim стилометрия) / extract_all /
│                              #   build_clusters / _pool_junk (чистка пула:
│                              #   FAQ+промо по всему зерну) / build_doc_embs
│                              #   (v18: эмбеддинги всех доков → doc_embs_v1.npz)/
│                              #   Navigator, MultiNavigator (+z_head, feat_head,
│                              #    аттракторное поле без изменений) /
│                              #   TextGrainEngine (weave) / stitch_narrative /
│                              #   compute_hub_scores / CITE_RE, JUNK_RE
├── train_field_v15.py         # тренировка прод-поля (FD=66, 12000 шагов)
├── rebuild_caches.py          # 6 шагов: sems(force=False)→positions→affect→
│                              #   PCA→hub→doc_embs(force=True)
├── audio_bridge.py            # ИНФЕРЕНС+ЗВУК: --ask QA-режим, --ride (v16),
│                              #   конвейер якорей v17–v18b (_tail_vec, гейт
│                              #   дрейфа, гейт семьи), _build_traj_lookup,
│                              #   _ride_answer; аудио: M_text_to_audio,
│                              #   AudioRetriever (565k зёрен 0MGE),
│                              #   sonify_stream + render_manifest → WAV+readalong
├── build_corpus_glm.py        # corpus-zero (uint16 токены) → текст
├── build_corpus_big.py        # бинарные шарды 0agi/corpus-zero → corpus_0agi_big/
│
├── model.py                   # копия 0agi/model.py — для этапа 5 (z0-integration)
├── z_mechanics.py             # копия 0agi/z_mechanics.py — анализ z (этап 5)
│
├── corpus/                    # 2005 доков (doc_000001.txt ...)
├── corpus_extra/              # +1472 дока
├── corpus_0agi_big/           # +8991 доков из шардов (big_XXXXXX.txt)
├── pool/                      # кэши пайплайна:
│   ├── filelist.json          # md5-dedup список файлов
│   ├── text_pool_v1.npz       # фичи гранул + trajectories (~923k)
│   ├── text_pool_v1.jsonl     # тексты гранул (выравнены с npz по уровням)
│   ├── text_sem_mini_v1.npz   # MiniLM-эмбеддинги гранул (384d)
│   ├── text_affect_v1.npz     # VADER аффект
│   ├── sem_proj_v1.npz        # PCA семантики
│   ├── text_pos_v1.npz        # позиции в доке
│   ├── text_hub_v1.npz        # хаб-скоры
│   ├── doc_embs_v1.npz        # эмбеддинги 12 130 доков (v18)
│   ├── audio_index_v1.npz     # переходы 565k зёрен 0MGE для аудио-ретрива
│   └── clusters_v1.json       # метки кластеров micro/meso/macro
├── pool_backup_v1_20260823/   # старое поле целиком (архив)
├── backups_20260822/          # бэкапы кода перед большими правками
├── checkpoints/
│   ├── text_navigator_v15_field.pt  # ПРОД поле v15 (loss 6.40)
│   └── ...                           # v1–v12 архив (см. readme «Чекпоинты»)
└── output/
    ├── glm_<ts>.md/.npz       # архивная генерация attract-стека
    ├── blind_ab_v16.txt(+key) # слепое A/B attract vs ride (прочитано)
    ├── blind_ab_v17.txt(+key) # слепое A/B v16-ride vs v17 (прочитано)
    ├── blind_v18_battery.txt  # слепая батарея 8 посторонних тем (прочитана)
    └── sonify_qa_<ts>.json/.wav/_readalong.txt  # каждый ответ: манифест,
                                  # звук той же z-траектории, текст с таймкодами
```

## Команды

```bash
# датасет из corpus-zero (0agi): N документов
./venv/bin/python 0glm/build_corpus_glm.py --docs 300 --shards 1

# полный цикл: пул → кластеры → обучение → генерация
./venv/bin/python 0glm/granular_text_field.py --refresh --max-docs 300 \
    --clusters 256 --train-steps 500 --gen-steps 12 --seed 42

# только генерация по готовой модели
./venv/bin/python 0glm/granular_text_field.py --generate-only --gen-steps 24 --temperature 0.8

# условление целью (4-dim cond, зеркало target_stats 0MGE)
--target-diversity 0.6 --target-punct 0.3 --target-sentlen 0.5 --target-stopword 0.4
```

## Соответствие констант

| смысл | 0MGE | 0GLM |
|---|---|---|
| фича гранулы | FEAT_DIM=22 спектр | FEAT_DIM=32 стилометрия |
| уровни | 5/26/259 кадров STFT | фраза 3–24 сл / 1–2 предл. / абзац |
| кластеры | MiniBatchKMeans 1024, seed 42 | тот же |
| контекст | CONTEXT_LEN=12 | тот же |
| backbone | TransformerEncoder 3×192/4h | тот же |
| стримы | 6 частотных полос | 6 регистров |
| аттракторы | EMA 0.9/0.1, gate/update | БЕЗ ИЗМЕНЕНИЙ |
| z | — | Z_DIM=64, feat_head: z → стилометрия след. гранулы |

## Первый прогон (2026-08-21)

- датасет: 300 доков corpus-zero (fineweb-edu), shard 0
- пул: μ=15,293 σ=8,288 Ω=300 гранул · 256 кластеров · 20,286 пар
- MultiNavigator 1.6M параметров, MPS, 500 шагов: loss 5.36→5.13 (c=4.88)
- генерация 12 шагов × 6 стримов: текст связен внутри гранул, между гранулами
  пока скачет (ожидаемо при 500 шагах и CE 4.9) · z-траектория 12×64 сохранена
