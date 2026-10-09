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

## Inputs and outputs

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
- `WS /jobs/{id}/events` — `state`, `progress` (`progress`, `stage`, `message`), `log`, `item_error`,
  `paused`, `resumed`, `finished`.
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

`POST /segment` — WFL-ASR. `phonemes` of an item, if given, are used as the expected phoneme list.

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
  `unknown_words`, `guessed` (phonemes from the G2P model).
  With a TIFA model: its own G2P; tokens are the words it found, phonemes are the first reading, and
  `candidates` lists all readings of words that have several (alignment picks one by the audio).
- `POST /text/validate` — like g2p, only the unknown words.
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

### Catalog format

```json
{
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

`engine`: `sofa`, `hubertfa`, `tifa`, `whisperx`, `wfl_asr`, `game`, `tempo`, `pitch`, `separation`. The task comes from
the engine; `"tasks": [...]` overrides it. Pack `models` are optional: listed members are offered by language
before the pack is downloaded (`folder` = folder name inside the archive); unlisted folders are registered as
`<prefix><folder>` after the download.

Source types: `url`, `github_release` (`asset` is a file name pattern or a list of them; `tag` can be
`latest`), `huggingface` (`repo`, `files`, `revision`), `engine` (the engine downloads the model itself).

## Settings

`GET /settings`, `PATCH /settings` (or `mvt config show|set`):

| key | default | |
|---|---|---|
| `torch_backend` | `auto` | `auto`, `cpu`, `cu126`, `cu128`... used when engines are installed |
| `device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:1` |
| `engine_idle_timeout` | 600 | seconds before an unused engine is stopped |
| `auto_install_engines` | true | install engines when a job needs them |
| `auto_download_models` | true | download catalog models when a job needs them |
| `catalogs` | [] | extra catalogs |
| `github_token`, `hf_token` | | optional |
| `max_parallel_jobs` | 2 | |
