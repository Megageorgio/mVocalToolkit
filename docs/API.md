# API

Base URL: `http://127.0.0.1:8765`. Interactive docs with all fields: `/docs` (OpenAPI schema at `/openapi.json`,
usable to generate clients for Kotlin, C#, TypeScript...).

Authentication: requests from the same machine are allowed without a token. Other machines send
`Authorization: Bearer <token>` (or `X-MVT-Token`, or `?token=` for WebSockets).

## Choosing a model in a GUI

Each model belongs to a **task**: `transcribe`, `align`, `segment`, `midi`, `tempo`, `pitch`, `separate`, `refine`.

1. `GET /tasks` — tasks, their engines, and whether the models depend on the language.
2. `GET /languages?task=align` — languages with their models:

   ```json
   {"task": "align", "languages": [
     {"code": "ru", "name": "Russian", "native_name": "Русский", "models": [
       {"id": "sofa-ru-hhskt-v0.0.1", "name": "Russian SOFA (hhskt)", "engine": "sofa", "installed": false, ...},
       {"id": "person1-ru", "name": "Person 1", "engine": "sofa", "installed": true, "pack": "my-pack", ...}
     ]},
     {"code": "ja", ...},
     {"code": "*", "name": "Any language", "models": [...]}
   ]}
   ```

   Models without a language (`"*"` or none) are added to every language and listed in the `*` group.
   `GET /models?task=align&language=ru` returns the same as a flat list with all fields.
3. Start the operation with the chosen id: `POST /align {"model": "person1-ru", "language": "ru", ...}`.
   A model that isn't installed is downloaded by the job itself (stage `download` in the progress), the engine
   environment is created on first use too. `POST /models/{id}/download` downloads ahead of time.

The command line has the same: `mvt languages --task align`, `mvt models list --task align --lang ru`.

## Persisted bank workflows

`GET /training/workflows` returns persisted states, requests, completed stages, run IDs, results and errors.
`POST /training/workflows/check` validates the request and training configuration without installing engines or
running models. `POST /training/workflows` accepts the same request and starts the chain:

```json
{
  "project_dir": "D:/voices/example", "name": "example",
  "sources": [{"folder": "D:/recordings", "speaker": "voice", "language": "ru", "role": "main"}],
  "prototype": true, "prototype_steps": 200, "prepare": true,
  "preparation": {"label_model": "auto", "label_mode": "missing", "extra_languages": ["en"],
    "segment": true, "max_length": 20, "breath_padding": 0, "cache": "use", "f0_method": "auto"},
  "phonemizer_profiles": {"ru": "ru-hhskt"},
  "pipeline": {}, "checks": {"detailed_audio": false},
  "acoustic": true, "variance": true, "training": {}, "acoustic_settings": {}, "variance_settings": {},
  "device": "auto", "source": "main", "export": true, "review_before_training": false,
  "render_ds": "", "vocoder": "", "render_options": {}
}
```

The check response retains `ok`, `stages`, `prototype` and `warning`, and adds `plan`, `notices`, `device` and
`source`. `plan` is an ordered list of `{id, kind, ...details}`: per-source preparation/input, dataset, optional
manual review, vocoder/G2P training and exports, acoustic/variance training and exports, assembly, phrase creation
and rendering. Training rows contain their effective step caps (G2P `null` means epoch-limited); preparation rows
contain source/speaker/language and enabled `actions`. Assembly lists reused local exports in `components`.
`notices` contains stable codes for contextual GUI explanations, including review pauses and disabled bank export.
`legacy_g2p_stage` identifies an old auxiliary G2P stage: new desktop clients train G2P separately, while existing
persisted workflows and explicit older API recipes remain supported.
This is a configuration/path preflight: it does not enumerate/decode/hash audio, probe GPU memory, load checkpoint
objects, create outputs, install engines or download weights. Timing, actual VRAM use, phoneme coverage and graph
compatibility are not established by this check.

Render requests require assembly plus a local ONNX vocoder (`vocoder.yaml`) or a vocoder trained in the chain.
A generated render requires phrase generation or an explicit existing DS. Reused acoustic/variance folders must
contain `dsconfig.yaml`; empty paths are rejected for required acoustic exports. Generated phrase parameters and
note syntax are checked before starting, while trained-vocabulary compatibility is checked after training.
An explicit DS takes precedence over generated syllables; disabled generation settings are ignored.

Use the exact profile IDs from `GET /training/phonemizers`; an empty profile ID explicitly disables the preferred
fallback. `auto` WFL prefers installed compatible weights and otherwise requests the bundled multi13 model on
first use. Model/environment downloads follow toolkit settings. `f0_method:auto` preserves a matching source DS
curve, otherwise extracts RMVPE f0. Existing notes are preserved unless `replace_notes:true`; missing notes use GAME
when pitch training needs them. SOME weights are not used. `liquid_rules` accepts boolean rules or preceding-phone
lists, optionally with language-qualified keys. `breath_padding` adds silence only at exposed AP boundaries and
shifts phone/word/note/f0 timing consistently. Separation and alignment are disabled by default. Bank requests
remove legacy preparation/alignment `refine` options; independent labelling APIs remain compatible.

`pipeline.separation` is a list of `{model, stem, stems?, options?}` passes, e.g. vocals then dry. `pipeline.alignment`
accepts AlignRequest options except managed input/output/language; `replace` controls alignment of existing labels.
Optional transcription uses the aligner's normal settings; interactive transcription review belongs in the
labelling tool, while this chain offers `review_before_training`. `pipeline.cache` is use/refresh/off. Final audio
and labels are separate from intermediate stems. `exclusions` lists absolute original audio paths.

`prototype:true` always caps `overrides.max_updates` at `prototype_steps`, even with a saved full recipe. Full mode
uses normal presets; `training` and per-model settings accept the normal DiffSinger configuration fields. Dataset,
workspace and base-config paths are managed by the workflow and cannot be replaced through overrides. Variance
note/group requirements follow the selected prediction heads. Disable a trainer and set `acoustic_export` or
`variance_export` to reuse compatible exports; a variance export is optional for an acoustic-only bank.

`POST /training/workflows/{id}/stop` stops the chain and requests a checkpoint from its active trainer.
`POST /training/workflows/{id}/resume` continues stopped/failed/interrupted states from completed stages. A `review`
state is approved only by this explicit resume; its editable dataset copy is rechecked into a new snapshot.
Restarted toolkits mark active chains interrupted and require manual resume. Training processes follow their normal
recovery rules. Source/setting changes require a new workflow; completed stages retain their original snapshots.
Active background work is exposed as `GET /health.background_active` so clients can detach without shutting down
the shared toolkit. Banks go under `<project_dir>/exports/<name>-<workflow-id>`.

`GET /training/preparation/cache` lists namespaced cache keys, bytes, timestamps and categories. `POST
/training/preparation/cache/remove {"keys":["recording-preparation/<hash>","source-pipeline/<hash>"]}` removes
selected allowlisted cache entries only, while no jobs/workflows/trainers are active. Sources, completed outputs
and checkpoints are retained. Cache restoration verifies output hashes and copies files independently. Keys include
content/settings, model file metadata and implementation fingerprints. Refresh recomputes outputs; existing cache
entries are immutable and can be explicitly removed to repair them.

## Migration, conversion inspection and exported-bank preview

- `POST /training/import/check {"folder":"D:/DiffTrainer"}` inspects YAML/checkpoint metadata without unpickling.
- `POST /training/import {"folder":"D:/DiffTrainer","project_dir":"D:/voices/imported","copy_data":true,
  "source":"benchmark"}` returns a job. Optional `configs` selects discovered config paths. Copies resolved configs,
  maps, checkpoints and optionally data/binaries into stopped runs; originals are preserved. False `copy_data`
  retains absolute data references. Review matching source versions and external asset paths before continuing.
- `POST /training/render/external` returns a job and accepts checkpoint, config, ds, kind (default acoustic),
  source (main/benchmark), device, steps, language and speaker. It creates a separate checkpoint workspace.
- `POST /training/voicebank/render` returns a job and accepts bank, ds, optional vocoder folder, speaker, language,
  steps (1–1000), depth (0–1], gender, velocity, expression, predict (dur/pitch/variance) and explicit inputs tensors.
  The isolated CPU worker executes the bank's exported graphs and validates vocabularies and input contracts.
  DS must contain timed phones and f0 or notes. Missing pitch/expressions are predicted when bank heads exist;
  duration prediction is explicit and keeps word anchors. A local compatible vocoder.yaml+ONNX is required. Results
  include WAV paths and render-report.json. This is DS inference, without OpenUtau's text frontend/plugin runtime.
- `POST /training/phonemizers/preview` accepts path (audio with labels or a label), language, profiles, alphabet
  and mappings; returns up to 2000 timed source/target examples and conversion findings without modifying sources.
- `POST /training/phonemizers/clip` accepts audio, start and end; returns a job creating a short WAV for listening.
- `POST /training/phonemizers/merge-suggestions` accepts entries with phone, ipa and optional feature mappings.
  Only exact IPA/feature matches across languages are suggested. No merge is applied automatically.

Imports/workflows are fixture-tested. Exported-bank sessions are tested with synthetic ONNX graphs, including all
five model components and a vocoder. Real pretrained GPU chains, arbitrary upstream export variants, phonemizer
plugins and listening quality require separate validation. Detailed audio metrics are advisory noise/bandwidth/tail
heuristics rather than a reliable reverb diagnosis.

## Compute settings and ONNX preview resources

DiffSinger's native `overrides` accepts `pl_trainer_precision` (guided choices: `16-mixed`, `bf16-mixed`, `32`),
`accumulate_grad_batches` (1–1024), `ds_workers` (0–64), `binarization_args.num_workers` (0–64),
`num_ckpt_keep` (1–1000, -1 keeps all, 0 disables regular saves) and `permanent_ckpt_interval` (0 disables new periodic permanent saves, otherwise at
least 10 steps). Existing preset schedules, permanent-checkpoint start and protected checkpoints still apply.
Saving settings does not remove old files. Effective values are checked after override merging; CPU uses FP32,
and data-loader prefetch is normalized for zero/nonzero workers. BF16 requires compatible hardware/trainer support.
Gradient accumulation changes effective batch size; it does not automatically reduce the chosen micro-batch.
Training precision does not convert exported ONNX weights or create a compact architecture.

`POST /training/voicebank/render` and workflow `render_options` accept `cpu_threads` (integer 1–32, default 2)
and `cache_models` (boolean, default true). With caching, each auxiliary graph is loaded once per request;
disabling it lowers retained graph memory at the cost of repeated loading. No sessions are reused across requests.
The result/report includes `resources` with thread/cache settings and `session_loads`/`session_reuses` counters.
Invalid resource settings fail before job submission. Preview still uses the CPU execution provider.

Custom vocoder assembly uses the `model` reference from `vocoder.yaml`, preserves nested relative paths and rejects
escaping/missing references before writing a bank. Several ONNX files require an explicit selection. Bank linting
reports incompatible declared mel metadata, and preview rejects it before loading graphs. Shared checks cover
sample rate, hop, FFT/window size, mel bins/scale and frequency range. Natural/base-10 log conversion is supported;
other mel conversions are not implicit. Unspecified legacy metadata does not establish full compatibility.

Boundary-refiner training is excluded from the offered recipes and rejected by new-job creation, including direct
manager calls. Old bank preparation/alignment `refine` settings are removed on request loading. This does not delete
existing weights, runs or reports, nor remove the independent low-level toolkit `/refine` and labelling APIs.

## Recording plans from dataset coverage

`POST /training/dataset/recording-plan` accepts an existing `dataset.json` report and returns phoneme exercises:

```json
{
  "report": "D:/voice/workflows/example/dataset/dataset.json",
  "language": "ru", "speaker": "Singer", "min_count": 5,
  "patterns": ["cv", "vc", "v"], "repetitions": 3, "limit": 32,
  "include_profile": true
}
```

`speaker:""` uses combined language counts; selecting a singer requires that singer/language's measured counts,
with no fallback to donors or other singers. Targets include observed/dictionary symbols and, when enabled, the
recorded or explicitly selected alphabet. Optional `targets` replaces the target set; known symbols may still
anchor syllables. `profile` selects a bundled alphabet (empty disables it); explicit `vowels`/`consonants` support
custom types, including empty lists. Silence/control symbols are excluded. There is no word input or G2P inference.

The result includes deterministic `rows` with phone arrays/pattern/repetition counts, observed `counts`, `needed`
deficits, `planned_occurrences`, `remaining`, `unclassified`, selected `profile`, `profile_inferred` and plain `text`.
CV/VC/V/CVC exercises prioritise missing/rare sounds. Unknown types can appear as individual-phone exercises.
Old reports without alphabet snapshots may infer types by vocabulary fit, but never silently add inferred alphabet
symbols. Remaining deficits are reported if rows or repetitions are insufficient. Counts are planning arithmetic,
not a claim that new audio was recorded or that the exercises are valid words/natural syllables.

The endpoint reads only the report/dictionary, writes nothing, queues no job and starts no engine. Rebuild the
dataset after recording/labelling to measure real coverage. Limits: 1–128 rows, 1–20 repetitions, target count 1–100,
at most 2048 target symbols. The desktop can copy the result or save separate TXT/JSON files without overwriting.

## Inputs and outputs

### Phoneme audition and optional training automation

`POST /training/test-phrase/inventory` and `/training/test-phrase/create` accept `model_dir` (bank/prepared run),
or `run_id`; an optional `config` points to a matching native config. `language` selects language-qualified phones.
Creation accepts `mode: cv|vc|v|cvc|manual`, `phonemes` (`p a | t a`, or lines per syllable), optional explicit
`vowels`/`consonants` lists, `notes` (e.g. `C4 E4 G4` or MIDI 1–127), `limit` (1–64), `vowel_seconds`,
`consonant_seconds`, `pause_seconds`. Uses actual trained vocabulary and dictionary/profile types; unknown symbols
and missing custom classifications fail explicitly. Output contains a managed DS path, aligned phones/groups/notes/f0,
syllables and inventory. A short phrase can render without variance because fixed f0 is supplied. No words or
OpenUtau phonemizer runtime are involved. Phrases are limited to two minutes/512 manually entered phones.
Explicit empty vowel/consonant lists remain empty; omitted lists enable automatic classification. Language-specific
dictionary types take priority over generic dictionaries, and an explicitly disabled alphabet profile is not
silently replaced. Malformed vocabularies, duplicate IDs/symbols and non-object metadata fail with validation errors.
Relative vocabulary paths resolve beside the selected configuration; relative run reports resolve beside run.json.

Workflow options: `auto_test_phrase` (default true), `render_test_phrase` (default false) and `test_phrase` settings.
`render_ds` still selects an existing phrase. `vocoder_training` enables an optional SingingVocoders stage and
native+ONNX export; requires `accept_license:true` and clean 44.1 kHz main-source data. The new checkpoint supplies
acoustic's vocoder asset, and ONNX supplies bank assembly/preview. `g2p_training` lists dictionary-training settings;
missing dictionary uses the checked project's dictionary for that entry's language. Outputs and run IDs persist.
These stages use the normal queue, checkpoints and stage retry; prototype caps their steps too. G2P `max_steps:0`
uses the epoch limit. No speech/phonemizer runtime is installed by optional G2P export.
The `g2p_training` workflow field is retained for persisted chains and explicit legacy API clients. New desktop
bank recipes omit it: standalone `g2p_rnnt` runs under preparation-tool training are the preferred workflow.

DiffSinger configuration supports `resources: {auto_batch:false, oom_retries:0, memory_fraction:0.7,
min_batch_frames:1000, min_batch_size:1}`. Opt-in auto_batch measures selected free CUDA memory and caps preset
limits; CPU/MPS retains existing limits. This is an estimate, not a dry-run fit guarantee. OOM retries wait for
process exit, halve frame/recording and validation limits down to the minima, and queue resume using prepared
binaries/checkpoints. `resource_retries` persists; 0 disables, at most 10 retries. Unsaved updates may be repeated.
Auto sizing also caps validation batches. Nonpositive/unset native validation limits are treated as unlimited and
replaced by finite caps; explicit smaller validation limits are preserved. Recovery persists training and validation
caps in config.yaml and the saved overlay/overrides. Resource flags require booleans, integer limits require actual
integers, and memory_fraction must be a finite number; strings such as `"false"` do not enable features implicitly.
`early_stopping: {patience:0, min_delta:0, min_steps:0}` counts completed finite val_loss checks after warmup.
Zero disables; otherwise the persisted counter and final/best checkpoints survive normal continuation.

`POST /training/phonemizers/mapping-suggestions` accepts source `entries` and `targets`, each with phone, ipa,
optional features. Returns unambiguous exact mappings, ambiguous alternatives and missing matches, without applying
them or making approximate phonetic guesses. Existing project mappings take priority in the desktop workbench.

`POST /training/speech/render` returns a job for a local native `model_dir`, `kind: qwen_tts_voice|voxcpm_voice`,
`text` (1–4000 characters), optional speaker, language, device, seed, steps, max_tokens, temperature and guidance.
VoxCPM accepts an explicit reference WAV or uses `speaker-references.json`; LoRA requires a compatible local
`base_model`. No speech base is downloaded by this preview. Outputs are managed WAV/report paths; shared heavy
processing slots and cancellation apply. This is a separate TTS feature, not singing text input.
The API validates bounds, device, blank text and local package/reference/LoRA-base paths before submitting a job.
Relative local input paths are normalized before crossing into the isolated engine; an empty LoRA base never
means the current directory. Unknown output-path kwargs are ignored and outputs stay managed. Existing output
folders are rejected by the worker before importing Torch or loading weights.

Speech training partitions each speaker/language separately and uses training-only references. Qwen exports distinct
speaker embedding IDs excluding reserved codec IDs. VoxCPM packages named references; a single shared model is
conditioned on the chosen voice reference. Full pretrained training/convergence and perceptual quality still need
real-data validation. Small package tests and substituted native-call contracts do not establish voice quality.

Every operation takes an `input`:

```json
{
  "items": [
    {"path": "D:/voice/a.wav", "text": "lyrics of the song"},
    {"path": "D:/voice/b.wav", "words": ["privet", "mir"]},
    {"path": "D:/voice/c.wav", "phonemes": ["p", "rj", "i", "vj", "e", "t"]},
    {"file_id": "4f3c...", "name": "uploaded"}
  ],
  "folder": "D:/voice/corpus",
  "recursive": true,
  "patterns": ["*.wav", "*.flac", "*.mp3", "*.ogg"],
  "sidecar_text": [".txt", ".lab"]
}
```

- `phonemes` > `words` > `text`: the most specific one is used.
- For folder inputs and items without text, `<name>.txt` / `<name>.lab` next to the audio is used as text
  (time-aligned `.lab` files are ignored).
- `file_id` comes from `POST /files` (multipart upload) when the server runs on another machine.

Operations that produce labels take an `output`:

```json
{
  "formats": ["htk", "textgrid", "ds_csv", "json", "audacity"],
  "dir": "D:/voice/labels",
  "layout": "subfolders",
  "overwrite": true,
  "return_labels": true
}
```

`layout`: `subfolders` → `<dir>/htk/a.lab`, `<dir>/textgrid/a.TextGrid`; `beside` → `<dir>/a.lab`.
Without `dir`, files go next to the audio (or to `<home>/outputs/<job>` for uploaded files).
`ds_csv` writes one `transcriptions.csv` per output folder. Results contain the paths; remote clients
download them with `GET /files/download?path=...`.

## Jobs

`POST /align` (and other operations) returns a job:

```json
{"id": "a1b2c3d4e5f6", "kind": "align", "status": "queued", "progress": 0.0, ...}
```

- `GET /jobs/{id}?wait=10` — current state, waits up to 10 s for completion.
- `WS /jobs/{id}/events` — `state`, `progress` (`progress`, `stage`, `message`, `detail`), `log`, `item_error`,
  `paused`, `resumed`, `finished`.

`detail` (also in `GET /jobs/{id}`) holds the numbers of the current step, for a progress display: while a model
downloads `file`, `file_index`, `files`, `done` and `total` (bytes, `total` null when unknown), `speed` (bytes/s),
`eta` (seconds or null); while files are transcribed or aligned `items_done`, `items_total`. Empty otherwise.
- `POST /jobs/{id}/cancel`
- `POST /jobs/{id}/resume` — continue a paused job.

Statuses: `queued`, `running`, `paused`, `done`, `failed`, `cancelled`. A failed file doesn't fail the whole
job: see `item_errors` and `ok`/`error` of each item in `result.items`.

## Transcription

`POST /transcribe`

```json
{
  "input": {"folder": "D:/voice/corpus"},
  "model": "whisper-large-v3-turbo",
  "language": "ja",
  "batch_size": 8,
  "compute_type": "auto",
  "vad": {"enabled": true, "method": "silero", "onset": 0.5, "offset": 0.363, "chunk_size": 30},
  "initial_prompt": "known words or lyrics to guide recognition",
  "suppress_numerals": true,
  "frontend": true,
  "save_txt": false
}
```

Result items: `text` (as recognized), `tokens` (normalized for alignment: words, kana phonemes for Japanese,
pinyin for Chinese...), `data.transcription.segments` with timings and `low_confidence` (nothing or too
little was recognized — worth a manual check).

If the GPU runs out of memory, the batch size is halved automatically.

## Alignment

`POST /align` (`POST /pipelines/label` is the same with all steps). The aligner is chosen by the model:
SOFA, HubertFA or TIFA.

```json
{
  "input": {"folder": "D:/voice/corpus"},
  "model": "sofa-ru-hhskt-v0.0.1",
  "language": "ru",
  "mode": "force",
  "g2p": "auto",
  "ap_detector": "loudness_spectral_centroid",
  "non_lexical_phonemes": ["AP"],
  "pad_times": 1,
  "pad_length": 5.0,
  "dictionary": null,
  "transcribe": {"model": "whisper-large-v3-turbo", "batch_size": 8},
  "review_transcription": false,
  "skip_unknown_words": false,
  "postprocess": {"rule_sets": [], "rules": []},
  "output": {"formats": ["htk", "textgrid"]}
}
```

- `model`: catalog/installed id, or `path:D:/models/my_sofa` for a model folder that isn't installed.
  Catalog models are downloaded automatically on first use.
- `language`: language of the texts. Multilingual models (HubertFA) use the dictionary of this language.
- `extra_languages` (TIFA): other languages that may occur in the texts, in priority order, e.g. `["en"]` for
  English words in Chinese lyrics. Phonemes of the main language come without a prefix, the others with it
  (`en/s`).
- `optional_breaths` (TIFA, default `false`): an optional breath at the start of each item, at `SP` marks of
  `words`/`phonemes` and after punctuation of a text. TIFA places `AP` where it hears a breath; the ones it
  doesn't hear (squeezed to a frame or so, shorter than 60 ms) are left out. Needs an `AP` phoneme in the model.
- `split_silence` (TIFA and SOFA, default `false`): files longer than `split_max_length` (25 s) are aligned once as a
  whole, then cut only at pauses (the middle of a silence of at least `split_min_silence` = 0.3 s, at least
  30 dB below the loud parts, where the first pass has no sound; or the start of a breath) into segments of up
  to 25 s when the pauses allow it, and each segment is aligned again with its own words (with the pronunciation
  chosen by the first pass). A cut never falls inside a word. `data.diagnosis.segments` tells how many segments
  were used; if a segment fails, the whole-file alignment is kept.
- TIFA takes the text as it is and does its own G2P (dictionaries, MeCab for Japanese, an LSTM model for
  unknown English words); when a word has several readings it picks the one heard in the audio.
  `words` (fixed word boundaries) and `phonemes` (known phonemes) items work too; `extra_words` fixes the
  phonemes of given words. Each item gets `data.diagnosis` with self-check scores (`agreement` 0..1: how well
  the text fits the audio; `confidence` −1..1: how well each phoneme's span fits it; `determinacy`,
  `monotonicity`; `skipped_phonemes`: phonemes the model found no room for, kept as 1 ms intervals). Low values
  mark files worth checking. The label has a `texts` tier (written words, e.g. 猫) when it differs from the
  `words` tier (readings, e.g. ne ko).
- `mode` (SOFA): `force` uses every phoneme; `match` allows the alignment to skip phonemes that aren't sung.
- `g2p`: `auto` = dictionary, then the model's G2P model (`g2p/` folder, SOFA) for unknown words;
  `dictionary` = dictionary only; `none` = tokens are already phonemes.
- `ap_detector` (SOFA) / `non_lexical_phonemes` (HubertFA: `AP` breath, `EP` other sounds): breath detection.
  `ap_detector: "none"` disables it for both.
- `pad_times`, `pad_length` (HubertFA): several passes with different padding, averaged — steadier boundaries.
- `dictionary`: a custom dictionary file instead of the model's one.
- `transcribe: null` disables transcription: items without text fail.
- `skip_unknown_words`: drop words that can't be converted instead of failing the file.
  Unknown words are reported in `unknown_words` of each item. `POST /text/validate` checks texts beforehand.
- `review_transcription: true`: the job pauses after transcription with
  `pause_data = {"stage": "review_transcription", "items": [{"name", "audio", "text"}, ...]}`.
  Continue with `POST /jobs/{id}/resume`:

  ```json
  {"items": [{"name": "a", "text": "corrected lyrics"}, {"name": "b", "skip": true}]}
  ```

### Post-processing rules

```json
{"rule_sets": ["en_fixes"], "rules": [{"op": "replace", "from": "pau", "to": "SP"}]}
```

| op | fields | effect |
|---|---|---|
| `replace` | `from`, `to` | rename a phoneme |
| `merge_pair` | `first`, `second`, `into` | `uh r` → `er` |
| `merge_duplicates` | `phones` (optional) | merge equal neighbours |
| `merge_short` | `phones`, `max_dur` | merge short phonemes into the previous one |
| `contextual` | `phones`, `prev`, `next`, `max_dur`, `to` | `t`/`d` between vowels → `dx`; `prev`/`next`: `vowel`, `consonant`, a phoneme or a list |
| `min_duration` | `min_dur` | merge too short phonemes into neighbours |

Built-in sets: `en_fixes` (common English fixes), `cleanup` (`pau`/`sil` → `SP`, `br` → `AP`),
`merge_duplicates`. A model can set default rule sets in its catalog entry (`defaults.rule_sets`).

## Segmentation without text

`POST /segment` — WFL-ASR or MCFL (an exported MCFL package: model.onnx, labels.txt, langs.txt…; it uses `language`, the silence options and its own Viterbi bias, the other options are for WFL-ASR). `phonemes` of an item, if given, are used as the expected phoneme list.

```json
{"input": {"folder": "D:/voice"}, "model": "wfl-asr-ft-en-ja", "lang_id": null, "confidence_threshold": 0.3}
```

## Boundary refinement

A refiner model (mRefinerModel) moves the boundaries between touching phonemes to where they are in the sound;
the phonemes, their order and a minimum length stay. Off by default: add `"refine"` to `/align` or `/segment`
(applied before the rules, so the refiner sees the model's phoneme names), or refine ready labels:

```json
{"input": {"items": [{"path": "a.wav", "segments": [[0, 0.31, "SP"], [0.31, 0.52, "k"], [0.52, 1.0, "a"]]}]},
 "model": "mrefiner-ru-v0.1.0", "mode": "auto", "output": {"formats": [], "return_labels": true}}
```

Without `segments` the label file next to the audio (`.lab`, `.TextGrid`, `.json`) is used. `mode`: `normal` (the
model's own language and labelling style), `safe` (only confident boundaries, not far: other languages or styles),
`auto` (safe when most phonemes are unknown to the model). `phone_map` renames other phoneme names to the model's;
`min_confidence` and `max_shift_ms` override the mode. `data.refine` of an item: `mode`, `boundaries`, `moved`.
Any refiner model works: `POST /models/import` with `engine: "refiner"` and a folder with `config.yaml`,
`phonemes.txt` and `model.pt`.

## Notes

`POST /midi/extract` — GAME. `tempo: "auto"` estimates the BPM with DeepRhythm first.

```json
{"input": {"items": [{"path": "song.wav"}]}, "model": "game-large", "tempo": "auto",
 "output_formats": ["mid", "csv"], "pitch_format": "name"}
```

`POST /tempo` — BPM and confidence per file.

## Pitch

`POST /pitch` — f0 curve for piano rolls / pitch editing.

```json
{"input": {"items": [{"path": "a.wav"}]}, "model": "rmvpe", "hop": 0.01, "f0_min": 50, "f0_max": 1100,
 "output_formats": ["csv"], "return_curve": true}
```

- `model`: `rmvpe` (robust on singing, ~180 MB model), `fcpe` (fast, no download), `parselmouth` (Praat, CPU),
  or `path:` to an RMVPE checkpoint.
- Result items: `data.f0` (Hz per frame, 0 = unvoiced, frame `i` is at `i * hop` seconds), `data.voiced_ratio`,
  `files` (`csv`: `time,f0`; `json`: `{"hop", "f0"}`; `txt`: one value per line).

## Vocal separation

`POST /separate` — only when the user asks for it: no other operation separates audio by itself, because
separation can make clean recordings worse.

```json
{"input": {"items": [{"path": "song.mp3"}]}, "model": "separation-vocals-bs-roformer", "stems": ["vocals"],
 "output_format": "wav", "output_dir": "D:/voice/stems"}
```

- Built-in models: `separation-vocals-bs-roformer`, `separation-vocals-melband-roformer` (vocals /
  instrumental), `separation-karaoke-melband-roformer` (lead / backing vocals, run on extracted vocals),
  `separation-dereverb-echo`. Any other audio-separator model file name works as `model` too.
- `stems`: keep only these (`vocals`, `instrumental`, `no_reverb`...); default: all stems of the model.
- Result items: `files` = `{"vocals": ".../song_vocals.wav", ...}`. Feed the vocals to `/align`, `/pitch`, etc.

## Text

- `POST /text/normalize` `{"texts": [...], "language": "ja"}` → tokens.
- `POST /text/g2p` `{"texts": [...], "model": "sofa-ru-hhskt-v0.0.1"}` → tokens, phonemes per token,
  `unknown_words`, `guessed` (phonemes from the G2P model). A model without a G2P model of its own (HubertFA,
  SOFA without `g2p/`) borrows the G2P of another installed SOFA model of the language; a guess is used only when
  the model knows every phoneme of it.
  With a TIFA model: its own G2P; tokens are the words it found, phonemes are the first reading, and
  `candidates` lists all readings of words that have several (alignment picks one by the audio).
- `POST /text/validate` — like g2p, only the unknown words (with `guessed` when a G2P can spell them).
- `GET /models/{id}/words`, `PUT /models/{id}/words` `{"word": ["ph", ...]}` — the user's own words of an aligner
  model, kept in `<home>/dictionaries/<id>.txt` and used by every request with that model (`/align`, `/text/*`).
  Alignment also adds G2P guesses for words the dictionary lacks (with `g2p: "auto"`), so a word fails only
  when no G2P can spell it. CLI: `mvt g2p "text" -m <model> [--validate]`, `mvt words list|add|remove <model> [word phonemes…]`.
- `POST /convert` `{"content": "...", "from_format": "textgrid", "to_format": "htk"}` or `{"path": ...}`.

## Engines and models

- `GET /engines` — installed/outdated/running, size.
- `POST /engines/{name}/install?force=false` — job; `DELETE /engines/{name}`; `POST /engines/{name}/stop`
  (frees GPU memory; idle engines stop automatically after `engine_idle_timeout` seconds).
- `GET /engines/{name}/info` — Python, torch, CUDA, GPU of the engine's environment.
- `GET /models?task=align&language=ru` (also `engine=sofa,hubertfa`) — catalog + installed.
- `POST /models/{id}/download` — job. `DELETE /models/{id}`.
- `POST /models/import` `{"engine": "sofa", "path": "D:/models/my_model.zip"}` — folder or archive,
  several models inside are imported separately.
- `GET /catalogs`, `POST /catalogs {"ref": "https://.../catalog.json"}`, `POST /catalogs/refresh`.

### Disk space

- `GET /storage` (`mvt storage`) — bytes of each model (`models`), each engine's environment with its sources
  (`engines`), files the engines downloaded themselves such as Whisper models (`engine_downloads`) and the
  leftovers of jobs (`temporary`: `uploads`, `outputs`, `jobs`, `cache/work` with unfinished downloads).
- `POST /storage/cleanup {"older_than_days": 7, "parts": ["uploads"]}` (`mvt storage cleanup --days 7`) — removes
  leftovers; without `older_than_days` all of them, without `parts` every temporary part. While a job runs only
  entries older than a day go. Leftovers older than `keep_files_days` are removed when the server starts.

### Catalog format

```json
{
  "format": 1,
  "name": "my-catalog",
  "models": [
    {
      "id": "sofa-xx-mymodel-1.0",
      "engine": "sofa",
      "name": "My model",
      "version": "1.0",
      "languages": ["xx"],
      "text_frontend": "xx",
      "source": {"type": "github_release", "repo": "me/models", "tag": "v1", "asset": "*.zip"},
      "defaults": {"rule_sets": []}
    },
    {
      "id": "my-pack",
      "type": "pack",
      "engine": "sofa",
      "prefix": "mypack-",
      "source": {"type": "url", "url": "https://example.com/models.zip", "sha256": "..."},
      "models": [
        {"id": "person1-ru", "folder": "person1_ru", "name": "Person 1", "languages": ["ru"], "text_frontend": "ru"}
      ]
    }
  ]
}
```

`engine`: `sofa`, `hubertfa`, `tifa`, `whisperx`, `wfl_asr`, `mcfl`, `game`, `tempo`, `pitch`, `separation`. The task comes from
the engine; `"tasks": [...]` overrides it. Pack `models` are optional: listed members are offered by language
before the pack is downloaded (`folder` = folder name inside the archive); unlisted folders are registered as
`<prefix><folder>` after the download.

Source types: `url`, `github_release` (`asset` is a file name pattern or a list of them; `tag` can be
`latest`), `huggingface` (`repo`, `files`, `revision`), `engine` (the engine downloads the model itself).

## Training

Model training runs as **training runs**: not ordinary jobs but trainer processes of their own,
started by the toolkit, which keep running when the server stops and are followed again when it starts. One run
trains at a time; the others wait in a queue. Everything about a run is in its folder (`run.json`, `events.jsonl`,
`train.log`, `config.yaml`, checkpoints, `samples/<step>/`), by default `<home>/training/<id>`, or the folder the
spec names (a project folder). Used by SingForge; the same through `mvt train` and `mvt dataset`.

- `GET /training/environment?engine=diffsinger` — torch, CUDA, GPUs with free memory, free disk, as the engine
  sees them; `{"installed": false}` before the engine's first use (the first run installs it, ~3 GB with CUDA).
- `POST /training/dataset/build` (`mvt dataset build <folders> --out --lang`) — job: folders of labelled recordings
  (wav/flac + lab, TextGrid, Audacity, json or a `transcriptions.csv`; labels in a sibling folder are found by name)
  → `<out>/<speaker>/wavs/ + transcriptions.csv`, a `dictionary-<lang>.txt` of the phonemes that occur when none is
  given, `dataset.json` (files, minutes, phonemes, problems) and a ready run spec (`result.spec`).
- `POST /training/overlay` — the config a DiffSinger spec resolves to (presets applied), to show before starting.
- `POST /training/runs` (`mvt train start spec.json [--wait]`) — queues a run. The spec:
  ```json
  {"kind": "diffsinger_acoustic", "name": "acoustic-1", "dir": "D:/voices/masha/runs/acoustic-1", "device": "auto",
   "config": {"datasets": [{"raw_data_dir": "…/prepared/masha", "speaker": "masha", "language": "ru",
                            "test_prefixes": ["a01", "b07", "c12"]}],
              "dictionaries": {"ru": "…/dictionary-ru.txt"}, "extra_phonemes": [], "merged_phoneme_groups": [],
              "preset": {"goal": "full", "gpu_gb": 16, "size": "medium"},
              "variance": {"dur": true, "pitch": true, "breathiness": true},
              "finetune": {"ckpt": "…/model_ckpt_steps_100000.ckpt"}, "overrides": {}}}
  ```
  `kind`: `diffsinger_acoustic` or `diffsinger_variance`. `preset.goal`: `full`, `check` (200 steps, proves data,
  config and export), `preview` (a short fine-tune), `finetune`; `gpu_gb` picks the batch; `size` (small < 1 h,
  medium, large) the length. The engine merges the overlay over openvpi's defaults; `overrides` go last. The
  training assets (vocoder, RMVPE, hnsep) are catalog models of the `diffsinger` engine, downloaded when missing.
- `GET /training/runs[?active=true]`, `GET /training/runs/{id}` — the numbers: status (`queued`, `preparing`,
  `training`, `stopping`, `stopped`, `done`, `failed`), step, max_steps, speed, eta, losses, lr, gpu_mb, val_loss,
  best_val_step, checkpoints, samples (step → files), warnings, error.
- `POST /training/queue` with `{"ids": ["queued-run-2", "queued-run-1"]}` — persist the execution order of all
  currently queued runs. Every queued ID must appear once; stale or duplicate lists return HTTP 409.
  Run info includes `queue_order`. A resumed run is placed at the end of the queue.
- `GET /training/runs/{id}/events?since=0&wait=10`, `WS /training/runs/{id}/events` — everything that happened:
  `queued`, `prepare` {stage, progress, message}, `launched`, `start`, `step` {step, losses, lr, gpu_mb}, `validation`
  {val_loss}, `checkpoint` {step, path}, `checkpoint_removed`, `samples` {step, files}, `warning` {kind: hang | oom…},
  `stopping`, `end` {status, error, error_kind}. `samples` are the validation files DiffSinger renders with the
  vocoder at every validation (`gt_i`, `aux_i`, `diff_i`), as WAV files in the run folder (`GET /files/download`).
- `POST /training/runs/{id}/stop` — the trainer saves a checkpoint and ends (killed after a few minutes if not);
  `POST .../resume` — continues from the last checkpoint (also after a crash or a reboot); `DELETE .../{id}?files=`.
- `GET .../log?lines=200&name=train.log`, `GET .../checkpoints`, `GET .../samples`.
- `POST /training/runs/{id}/export` (`mvt train export <id> --step N`) — job: ONNX with DiffSinger's exporter
  (`acoustic.onnx`, `.phonemes.json`, `.languages.json`, `dsconfig.yaml`, the dictionaries), into
  `<run>/export/<step>/`.
  When a run has `config.data_report`, its snapshot is copied into the ONNX export as `dataset-report.json`.
  If donor speakers are recorded, an empty `export_spk` defaults to the main speakers; an explicit list overrides
  that selection. Voicebank assembly carries licenses from these reports into its readme and `report.json`.
- `POST /training/runs/{id}/render` (`mvt train render <id> --ds song.ds`) — job: a `.ds` sung with a checkpoint.
- `POST /training/voicebank` (`mvt voicebank <out> --name N --acoustic <export> --variance <export>`) — job: an
  OpenUtau DiffSinger voicebank from export folders: `character.txt`, `character.yaml` (subbanks per speaker),
  `dsconfig.yaml`, `dsmain/` (acoustic, phonemes, languages), `embeds/`, `dsdur/`, `dspitch/`,
  `dsvariance/` with their own `dsconfig.yaml`, linguistic encoder, vocabulary and embeds, `dsdict.yaml` (+ `dsdict-<lang>.yaml` for multilingual
  banks, with language-tagged symbols and replacements), `dsvocoder/` for a custom vocoder (else the standard
  `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` is named), `readme.txt`, `report.json` (source export folders,
  models, speakers, languages, licenses, credits, notes and linter findings). Existing banks require explicit
  `overwrite: true`; invalid folder names and output folders overlapping source exports are rejected.
  The result lists the files and the
  linter's `problems` (missing files, dangling references, speakers or phoneme sets that differ between the
  models, `use_lang_id` mismatches).

### Dataset checks and reproducibility

`POST /training/dataset/build` also accepts `role` (`main`/`donor`) and `license` per source. Its report includes
`recordings` (source/prepared paths, SHA-256, speaker/language/role/license and available audio metadata),
`coverage` (language -> phoneme occurrence counts), and findings with `code`, `severity` (`error`, `warning`,
`note`) and optional `excluded`. Invalid recordings and exact duplicate files are excluded. Blocking findings
remaining in the data prevent new report-backed runs from starting; pitch runs also require labelled notes.
`spec.data_report` points at the saved `dataset.json`; the desktop makes a separate snapshot per build.

Basic signal inspection streams PCM WAV blocks without loading ML libraries or using the GPU. FLAC and non-PCM
WAV signal checks are explicitly marked unavailable. This is a screening report, not a denoiser or repair tool.
Custom text dictionaries (tab/whitespace separated) and OpenUtau dsdict YAML entries are converted into canonical
text dictionaries in the prepared-data folder. A multilingual YAML dictionary is filtered to the selected
language and that language's phoneme prefix is removed. Same-speaker sources in several languages have
separate raw-data folders but share a speaker ID. The validation selection always leaves a training recording.

## Datasets

Public singing datasets for training (more voices, other languages) come from the same catalogs (key
`datasets`). Every dataset carries its license, and **only datasets whose license allows training are offered**;
one without a license or with `"training": false` is left out.

- `GET /datasets?language=ja` (`mvt datasets list --lang ja`) — each with `license` (`name`, `url`, `summary`,
  `training`, `publish_models`, `commercial`, `redistribution`, `attribution`), languages, hours, singers, labels,
  `installed` and `path`.
- `GET /datasets/{id}/license` (`mvt datasets info <id>`) — the license as fields and as text to show.
- `POST /datasets/{id}/download {"accept_license": true}` (`mvt datasets download <id> [--accept-license]`) — job.
  Without `accept_license` the answer is 403 with the license: a GUI shows it and asks the user first. A dataset
  that has to be downloaded by hand (a form, a login) answers 409 with `url`. The files go to
  `<home>/datasets/<id>/` together with `LICENSE.txt` (what is allowed and the credit text) and `dataset.json`.
- `DELETE /datasets/{id}` (`mvt datasets remove <id>`).

```json
{
  "datasets": [
    {
      "id": "my-dataset",
      "name": "My dataset",
      "author": "Who recorded it",
      "languages": ["ja"],
      "hours": 1.5,
      "singers": 1,
      "labels": ["phonemes", "notes"],
      "label_format": "lab",
      "sample_rate": 44100,
      "homepage": "https://...",
      "license": {
        "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/",
        "summary": "Any use with credit.",
        "training": true, "publish_models": true, "commercial": true, "redistribution": true,
        "attribution": "My dataset by Someone"
      },
      "sources": [
        {"type": "github_release", "repo": "me/data", "tag": "v1", "asset": "data.zip"},
        {"type": "url", "url": "https://.../labels.zip", "into": "labels", "part": "labels"}
      ]
    }
  ]
}
```

`sources` are downloaded in order into the dataset folder (`into`: a sub-folder, e.g. labels someone else made for
the same audio). `manual_download`: a page where the user downloads the files by hand (no `sources` then).

## Settings

`GET /settings`, `PATCH /settings` (also `POST`; changing `device` restarts the idle engines) (or `mvt config show|set`):

| key | default | |
|---|---|---|
| `torch_backend` | `auto` | `auto`, `cpu`, `cu126`, `cu128`... used when engines are installed |
| `device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:1` |
| `engine_idle_timeout` | 600 | seconds before an unused engine is stopped |
| `auto_install_engines` | true | install engines when a job needs them |
| `auto_download_models` | true | download catalog models when a job needs them |
| `catalog_url` | raw `catalog.json` of github.com/Megageorgio/mVocalToolkit-catalog | the main catalog, read at every start and update check (offline: the last copy, else the built-in one); `MVT_CATALOG_URL` overrides it, empty = built-in only |
| `catalogs` | [] | extra catalogs |
| `github_token`, `hf_token` | | optional |
| `max_parallel_jobs` | 2 | |
| `keep_files_days` | 14 | uploads, results in `<home>/outputs` and job history older than this are removed at start; 0 = kept |

## Checkpoint retention and own phrases

`POST /training/runs/{id}/checkpoints/{step}/retain` with `{"starred": true}` protects a saved checkpoint;
`false` removes its manual retention mark without deleting the trainer's file. Supported by DiffSinger and
SingingVocoders. Protected links/copies survive rotation and server restart. New detached trainers also skip
upstream deletion of manually retained checkpoints and keep a rotating backup of the best validation checkpoint.
Retention cannot recover a file already removed before the mark was set. Whole-run deletion still removes all
run files when explicitly requested with `files=true`.

`POST /training/runs/{id}/render` accepts `ds`, optional `step`, `spk`, `lang`, `seed`, `steps` (1–1000 sampling
steps for acoustic inference), and `depth` (0–1). The `.ds` must exist on the toolkit computer and contain a JSON
list of segments. The default output is a unique `<run>/renders/<id>` folder. The input is copied there for
reproducibility; the result is appended to `RunInfo.renders`. Acoustic inference returns audio; variance inference
returns a `.ds` for a subsequent acoustic render. This uses checkpoint inference, which can differ from OpenUtau.

Render and export jobs wait for an idle training resource; a request made during active training returns 409.
While such a job runs, queued training stays queued. Cancelling the job terminates the engine worker and its
subprocesses before releasing that resource. Results already completed stay in the run; no audio starts itself.

## Experimental vocoder fine-tuning

`POST /training/vocoder/check` receives `{"sources": [{"folder": "...", "speaker": "...", "license": "..."}]}`
and returns `files`, `seconds`, `recordings` with SHA-256 hashes, and coded `problems` with severity. Input is clean
44.1 kHz PCM WAV/FLAC; phoneme labels are not needed. Exact duplicate files are excluded. Separate training and
validation audio is required. Bandwidth/noise/reverb analysis is not implemented by this check.

Use `/training/overlay`, then `/training/runs` with:

```json
{
  "kind": "singingvocoders_vocoder",
  "dir": "D:/voices/myvoice/runs/vocoder-1",
  "config": {
    "sources": [{"folder": "D:/voices/myvoice/recordings", "speaker": "singer", "license": "own recordings"}],
    "preset": {"goal": "check", "gpu_gb": 8},
    "accept_license": true
  }
}
```

The user must review the OpenVPI base-weight terms before setting `accept_license`; the weights and derived model
use CC BY-NC-SA 4.0. Defaults: 100 real steps for `check`, 2000 for `full`. Optional positive integer settings:
`steps`, `validation_interval` (not longer than the run), `batch_size`, `crop_frames`; `learning_rate` defaults to
1e-5 and must not exceed 1e-3. `base_model` accepts a full SingingVocoders Lightning checkpoint or catalog ID;
the smaller DiffSinger inference asset is not a fine-tuning base.

The source is pinned in the engine manifest. Preprocessing copies mono audio, computes mel/f0 sequentially and
records a deterministic held-out split. Source hashes must still match. GAN generator+discriminator updates count
as one real step in events and the API; upstream checkpoint filenames contain twice that number. New checkpoints
and validation samples use the same queue/events/stop/resume controls as DiffSinger.

Export produces generator weights and config, then uses the `diffsinger` engine to export `vocoder.onnx` and
`vocoder.yaml`. A new export may therefore install the DiffSinger environment when it is absent. The folder can
be passed as `vocoder` to `/training/voicebank`. Recording provenance and the base-weight license are carried into
the bank report. Source imports and real preparation have been checked; full GPU fine-tuning and real conversion
of a newly trained vocoder remain to be validated.

## Project dictionary copies

`POST /training/dictionary/read` accepts `path`, optional `language`, and returns pronunciation `entries` as
`{"word": "...", "phones": ["..."]}`, unique-word count, duplicate words and phoneme counts. It reads text
dictionaries and OpenUtau dsdict YAML, with language-prefix filtering as used by dataset preparation.

`POST /training/dictionary/save` accepts `entries` and `out_dir`. Words/phonemes are normalized to Unicode NFC,
validated and written to a new UTF-8 text file. The returned `path` can be selected for that language in a voice.
Existing files and training snapshots are never overwritten. Editing/mapping is done by the client before save.

## Recording preparation and DiffSinger settings

`POST /training/dataset/prepare` is a cancellable job. Outputs must be outside the source tree and initially empty:

```json
{"folder":"D:/recordings","out_dir":"D:/prepared/corpus-1","segment":true,"max_length":20,
 "min_silence":0.25,"vowels":["a","e","i","o","u"],"liquids":["l","r","w","y"],
 "estimate_notes":true,"note_model":"game-small","language":"ja","write_ds":true,"f0_method":"rmvpe"}
```

The result contains `out_dir`, output audio `files`, `count` and coded `problems`. Preparation writes WAV, HTK,
TextGrid, transcriptions.csv and optionally ds/. Cuts are inside labelled silence and outside labelled words;
unbroken phrases can exceed the duration target. Existing word/ph_num groups take precedence. Without groups or
a vowel inventory, one phoneme per group is used with a warning. GAME is the successor of SOME; small/medium/large
models are available. `f0_method` also accepts `parselmouth`, `harvest` and `none`. Review estimated notes before
pitch training. The normal dataset builder rescues matching `.ds` sidecars from wavs/ or ds/, preserving f0/notes.

`GET /training/phonemizers` lists versioned target inventories, vowel lists, preferred profiles, OpenUtau class
names, dictionary filenames and source revisions. Expanded profiles are ru-hhskt, en-arpa-plus, de-marzipan,
fr-millefeuille and ko-g2p. Stock profiles cover Russian, English, German, Italian, Spanish and Portuguese.
These describe supported alphabets, not a promise that every WFL phone has an exact counterpart in every profile.

The same preparation request accepts:

```json
{"folder":"D:/raw-wav","out_dir":"D:/prepared/wfl-copy","language":"ru",
 "label_mode":"missing","label_model":"wfl-multi13-v0.0.1","extra_languages":["en","de"],
 "phonemizer_profiles":{"ru":"ru-hhskt","en":"en-arpa-plus","de":"de-marzipan"},
 "source_alphabet":"wfl-native","unknown_phonemes":"error","language_prefix":true,
 "phoneme_mappings":{},"segment":true,"max_length":20,"min_silence":0.25,
 "edge_silence":0.25,"fade_ms":5,"write_ds":true,"f0_method":"none"}
```

- `label_mode`: `existing` uses existing timed phoneme labels; `missing` invokes WFL only where labels are
  absent/unreadable; `replace` ignores all existing labels in the output copy. Originals are unchanged.
- Primary/additional languages must be declared by the model. Actual numeric IDs come from `langs.txt`,
  never list position. Additional languages require generation 2. WFL selects languages at phrase level.
- WFL returns **native language symbols**. `source_alphabet=target` validates existing prepared labels;
  `wfl-native` converts native output; `wfl-multi13-canonical` first restores native symbols using the bundled
  multi13-v1 merge table (its SHA-256 is stored beside the profiles). Never apply canonical restoration twice.
  For another model revision supply explicit mappings where its alphabet differs.
- Full `language/phone` override keys take precedence over unprefixed keys. Override values are lists;
  splitting divides duration equally and is reported as approximate. Empty/deleting mappings are forbidden.
  Unknown symbols block by default. `unknown_phonemes=keep` preserves them with warnings for custom dictionaries.
  Russian stock conversion folds reduced vowels and reports those losses; hhskt-ru preserves them.
- Multilingual phones retain language prefixes. Dataset construction creates separate unprefixed dictionaries
  for those languages, while keeping code-switched phonemes in the recording CSV/DS.
- `edge_silence=null` preserves edge pauses; 0–10 seconds caps only labelled SP. `fade_ms` is bounded by
  remaining labelled edge silence, never applied to voiced phones or breaths.
- `f0_method=preserve` crops an existing single-segment DS curve on its sample grid without invoking a model.
  Missing/insufficient curves are reported; no replacement curve is invented.

`POST /training/dataset/prepare/check` accepts this same request and checks paths, labels, model-language
metadata, conversions and inventories without creating output, starting workers or downloading weights.
It returns `ok`, `files`, `labelled`, `pending_auto_labels`, `phonemes` and coded `problems`. Raw WAV inventories
can only be checked after WFL inference. Jobs retain original WFL JSON under `original-labels/` and detailed
per-file mappings, losses, extensions and unknowns in `preparation-report.json`. If unknown WFL phones block a
job, that report and original labels remain available for review; use a new output folder after correcting it.

Voicebank assembly additionally accepts `phonemizer_profiles` (language -> trained alphabet) and
`default_phonemizer_profile`. It writes profile-specific dictionaries at the root and beside every variance
head. For hhskt Russian, both `dsdict-ru-hhskt.yaml` and a stock `dsdict-ru.yaml` remapper target the same trained
symbols. Stock compatibility is approximate where vowel distinctions differ. `phonemizer-report.json` lists
missing trained/G2P phones and plugin requirements. Remappers never rename the ONNX vocabulary. hhskt-ru must
be installed separately in OpenUtau; no DLL is installed by bank assembly. Other profile pairs may have partial
coverage and need custom mappings/dictionaries. Automated checks use synthetic labels/audio and fake exports;
real WFL accuracy and OpenUtau playback remain to be checked.

Dataset build accepts the same `phonemizer_profiles` map and freezes it in the dataset report/run spec.
Exported reports retain it. Bank assembly inherits that snapshot and rejects an explicitly conflicting target
alphabet, so changing project settings cannot silently reinterpret a model that was already trained.

DiffSinger run `config.options` accepts `backbone` (lynxnet2/lynxnet/wavenet), `dual_timestep`, `stretch`,
`mix_layer_norm`, `melody_encoder`, `pitch_augmentation`, `time_augmentation`, `prefer_ds`, `hnsep` (vr/world) and
`tensor_acceleration`. Backbone selection replaces architecture-specific argument dictionaries. `overrides` still
apply last. Positive schedule/batch values and final variance-module compatibility are checked. An explicit empty
`finetune.ignored_params` preserves all matching parameters. `config.prepare_only=true` ends with `stage=prepared`;
POST resume starts training using its binaries. It does not repeat binarization.

`POST /training/config/read` with `{"content":"JSON or YAML text"}` returns a JSON-compatible settings object,
without executing YAML constructors. Maximum content size is 1 MB.

## Advanced export and quick inference

Training export accepts `export_spk` (names or `alias=singer:0.7|other:0.3` mixes), `freeze_spk`,
`freeze_gender` (−1…1), `freeze_velocity`, `freeze_glide`, `freeze_expr`. Exported-speaker and frozen-speaker
selection are mutually exclusive. Freezing a speaker bypasses automatic main-speaker export selection.

`POST /training/runs/{id}/checkpoint-copy` with `{"step":200,"retain":["main"],"fill":"zeros"}` creates a
separate DiffSinger checkpoint folder. Fill can also be random/mean/cyclic. Non-retained embedding rows are
replaced, the speaker map is filtered and optimizer state is removed. Shared network weights still reflect all
training data; this operation does not remove donor influence from those weights. The original is preserved.

Voicebank assembly accepts `components={"dur":"<export>","pitch":"<different export>","variance":"<export>"}`
to select heads independently. Omit unwanted keys; each chosen export must contain the selected ONNX head.
Its own encoder, token IDs and embeddings accompany it. The older `variance` list remains supported.
`include_dictionary` defaults true; `phoneme_types` overrides dsdict symbol classes.

Acoustic render can include `variance_run` and `variance_step`. Variance inference runs first, and its `.ds`
becomes the acoustic input. `predict` chooses dur/pitch/energy/breathiness/voicing/tension to replace; enabled
parameters missing from the phrase are completed as well. `device` can be auto/cpu/cuda/cuda:N/mps; acoustic
sampling uses `steps` and `depth`. Input/output files, chosen variance run and normalized phrase SHA-256 are
recorded in render history, enabling content-matched A/B comparisons. Export/render/preparation reserve the
training queue; cancellation aborts engine children before releasing it.

`POST /training/runs/{id}/tensorboard` starts/reuses one CPU-only viewer on an ephemeral localhost port and
returns `url` and `pid`. It requires an existing DiffSinger/SingingVocoders environment. Opening a different run
replaces the viewer. `DELETE` on the same URL stops it; toolkit shutdown and deleting the viewed run also stop it.
The log is `<run>/tensorboard.log`. Remote users need their own port forwarding; the service never binds publicly.

## Pronunciation training

`POST /training/g2p/check` validates a G2P config without installing an engine. Start via `/training/runs`
(or `mvt train start spec.json --wait`):

```json
{"kind":"g2p_rnnt","device":"cpu","dir":"D:/voice/runs/g2p-1",
 "config":{"dictionary":"D:/voice/dictionary.txt","language":"ru","lowercase":true,
 "epochs":50,"batch_size":16,"hidden_size":128,"layers":2,"learning_rate":0.001,
 "validation_fraction":0.1,"patience":8,"seed":42}}
```

The dictionary is NFC-normalized and snapshotted. Alternate pronunciations of one word always stay together.
Splits, batches and initialization use the seed; resume restores optimizer and random state. Validation selects
the closest accepted pronunciation and reports phoneme edit error (PER), word error (WER) and error examples.
`val_loss` is held-out PER, not the RNN-T training loss. `validation.json` is the latest report; validations/ keeps
per-check reports. Early stopping uses PER; `patience=0` disables it. Large dictionary data stays outside polled
RunInfo JSON. Dependencies are installed only when training starts.

Fine-tune with `base_model="<folder containing cfg.yaml and model.ptsd>"`. The base architecture/token IDs are
inherited. `expand_vocabulary=true` appends missing letters/phonemes and initializes only new embedding/output
rows; otherwise incompatible dictionaries fail preflight. The starting weights are hash-checked and copied.

`POST /training/runs/{id}/predict` accepts `{"words":["example"],"step":200}` and returns raw model pronunciations
without dictionary lookup. Export writes cfg.yaml, model.ptsd, dict.txt, g2p.onnx, vocabulary.json and validation
reports. The ONNX graph is checked and compared to PyTorch at dynamic input lengths. SOFA's compatible G2P loader
can use the weights/config. An arbitrary vocabulary is not a drop-in OpenUtau language pack: its phonemizer must
use those exact grapheme/phoneme IDs and language-specific symbol types.


## RVC voice conversion training

`POST /training/rvc/check` accepts the config below, checks it without installing an engine, and returns
settings, file count/duration and coded findings. Start with `POST /training/runs` or `mvt train start`:

```json
{"kind":"rvc_voice","device":"auto","config":{
 "sources":[{"folder":"D:/voice/recordings","speaker":"voice","license":"CC0"}],
 "sample_rate":40000,"epochs":100,"batch_size":2,"save_every_epoch":5,
 "max_steps":20,"learning_rate":0.0001,"chunk_seconds":3.7,"overlap_seconds":0.3,
 "index_vectors":20000,"pretrained":true,"prepare_only":false}}
```

Only one speaker is supported; source labels are unused. 32/40/48 kHz output rates are supported, with separate
resampled copies. Exact duplicates are excluded; manifests hash the recordings and any local base weights.
Changed recordings are rejected during preparation. No engine or base weight is downloaded by preflight.
Preparation streams recordings into bounded chunks, filters/normalizes them, extracts Harvest pitch and
768-channel ContentVec features sequentially, and optionally builds an exact FAISS L2 index from a uniform,
seeded sample. `index_vectors=0` disables the index; 20,000 vectors use about 60 MB. No external mute/reference
corpus is required. At least three usable batches must survive preparation.

`max_steps=20` is a setup check, not a finished voice. Set `max_steps=0` to train all epochs. `prepare_only=true`
prepares data and finishes; resume starts the trainer. Training uses one CPU or CUDA device and float32;
explicit unavailable CUDA devices fail. ContentVec and the matching HiFi-GAN G/D base weights come from the
catalog only on start. Set `pretrained=false` for random initialization, or supply both `base_generator` and
`base_discriminator` paths to matching full training checkpoints. Compact exported inference weights cannot
replace the training pair. `embedder` may point to a local standard ContentVec folder containing config.json
and pytorch_model.bin. Base weights and recording licenses must be checked before distributing derivatives.

Stop finishes the current batch and atomically saves both networks, both optimizers, scaler, random states,
epoch and batch position. Resume restores that state and skips completed batches; it does not promise bitwise
identity across devices. Checkpoints use the shared retain/star API; the latest three unstarred checkpoints
are rotated. Each full checkpoint can use about 1.3 GB. `validation_fraction` (0–0.5, default 0.1) holds out whole
recordings before slicing and excludes their features from training and the FAISS index. At least two distinct
recordings are needed; otherwise previews use training examples. Set zero to disable. At checkpoints, up to eight
held-out chunks produce mean log-magnitude spectral L1 (`val_loss`), length error and a `heldout-*` listening pair.
The lowest spectral-error checkpoint is retained. This is a technical reconstruction metric, not perceptual voice
quality. Training chunks must still fill at least three batches. Full GPU quality validation remains pending.

The standard export job creates `voice.pth` (RVC v2, f0, HiFi-GAN, ContentVec, speaker ID 0), `voice.index` when
prepared, README.txt and provenance.json. The exported generator omits optimizer/discriminator and posterior
encoder weights and uses legacy RVC weight-norm names. It loads strictly in the pinned native Synthesizer.
Use the result in a compatible RVC client; it is not an OpenUtau DiffSinger voicebank.

### Additional model trainers

`GET /training/recipes` lists optional training families. `POST /training/recipes/{kind}/check` accepts the
configuration below and returns counts/findings without installing an engine or downloading model weights.
The normal `POST /training/runs` freezes the checked inputs again and derives the matching engine:

| kind | engine | Required data / export |
|---|---|---|
| `tifa_aligner` | `tifa_train` | Timed phonemes; native TIFA weights, vocabulary and dictionaries |
| `sofa_aligner` | `sofa_train` | Timed phonemes; SOFA inference checkpoint/vocabulary |
| `wfl_asr_teacher` | `label_train` | Timed phonemes; WFL-ASR v2 package with frozen Whisper encoder |
| `mcfl_student` | `label_train` | Timed phonemes and WFL v2 teacher; native fp32/int8 ONNX package |
| `game_notes` | `game_train` | Timed notes in CSV/DS; native GAME weights and five ONNX graphs |
| `qwen_tts_voice` | `qwen_tts` | Audio + matching `.txt`, up to 32 speakers; Qwen custom_voice package |
| `voxcpm_voice` | `voxcpm` | Audio + matching `.txt`, speaker references; VoxCPM2 full weights or LoRA |

```json
{
  "kind": "tifa_aligner",
  "device": "cuda:0",
  "config": {
    "sources": [{"folder": "D:/recordings", "speaker": "voice", "language": "ru", "license": "CC0"}],
    "steps": 100, "epochs": 20, "batch_size": 1, "accumulate": 1,
    "save_interval": 50, "validation_fraction": 0.1, "patience": 5, "seed": 42,
    "base_model": "", "prepare_only": false, "overrides": {}
  }
}
```

Use WAV/FLAC, mono/stereo, at least 8 kHz and 0.2–120 seconds per file. Timed labels use the existing lab,
TextGrid, CSV and DS readers. GAME notes use C4/F#4/Bb3/rest names within MIDI 0–127. Exact duplicate audio files
are deduplicated globally; whole recordings are split deterministically per language. Every language needs
two distinct recordings. Source audio hashes are checked again when copying/resampling into the run folder.

`learning_rate` defaults by family. `base_model` accepts a compatible local training/inference base. TIFA can
use project `dictionaries: {"ru":"D:/dict.txt"}`; SOFA supports checkpoint/safetensors bases and transfers phone
rows by name. WFL accepts `encoder` (local Whisper folder or Hub ID). MCFL needs `teacher` (WFL v2 folder with
config/maps/weights); `size` is small/medium/large. Its training base must use native unfused training weights,
rather than the folded `mcfl-v1` release format. Refiner validation uses synthetic timing shifts.

Qwen defaults to `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (compatible 1.7B Base supported), `speaker_name: "voice"`
and an optional `tokenizer` folder/Hub ID. Vox defaults to `openbmb/VoxCPM2`, `lora: true`, `lora_rank: 16`.
Speech configuration uses named fields; arbitrary native overrides are rejected. Automatic base downloads
follow the server's `auto_download_models` setting. Offline speech preparation requires a local base; WFL
also needs a local/cached encoder. Full precision TTS training needs substantially more memory than inference.

Prepare-only runs resume into training. Optimizer state, schedulers and checkpoints survive stop/resume;
augmentation order is not promised to be bitwise identical after restart. The normal export/checkpoint-star/
TensorBoard endpoints work for these engines. Exported labelling/note packages can be copied into
`POST /models/import` using engines tifa/sofa/wfl_asr/mcfl/game. TTS packages use their native inference
programs; a VoxCPM LoRA package requires its original base model. Validation/audio preview generation for Vox
follows the native ten-batch evaluation limit and is not a full-corpus quality measurement.

Native small CPU training, validation, export and inference loading passed for the six labelling/note/refiner
families. Reduced native Qwen/VoxCPM networks passed forward/backward checks. VoxCPM's native optimizer loop
also passed stop/resume with prepared-feature fixtures and a substituted audio decoder. Full pretrained TTS
preprocessing/training, GPU convergence and generated voice quality remain unverified. Large weights and new
training environments were not downloaded for validation.
