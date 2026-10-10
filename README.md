# mVocalToolkit

A local API server for automatic labeling of singing voice data. One toolkit, used by any GUI
(desktop, mobile, web, scripts) through HTTP:

- **Transcription** of lyrics — WhisperX (faster-whisper, batched, Silero VAD to skip silence and noise,
  automatic batch reduction on GPU out-of-memory).
- **Forced alignment** — SOFA (dictionaries, G2P models for unknown words, breath (AP) detection, `.ckpt` and
  `.safetensors` models, SOFA model packs), HubertFA (ONNX, multilingual models, breath and other
  non-lexical sound detection) or TIFA (speech and singing, several languages in one text, picks the
  pronunciation heard in the audio, self-check scores per file). The engine is chosen by the model.
- **Phoneme segmentation without text** — WFL-ASR.
- **Boundary refinement** — mRefinerModel moves the boundaries of ready labels (from any aligner or by hand)
  to where they are in the sound; optional after alignment or segmentation, off by default.
- **Notes (MIDI)** — GAME, with automatic tempo estimation (DeepRhythm).
- **Pitch (f0)** — RMVPE, FCPE, Parselmouth: curves for piano rolls and pitch editing.
- **Vocal separation** — vocals/accompaniment, lead/backing vocals, de-reverb (Roformer, MDX, VR, Demucs
  models via audio-separator). Only on explicit request: nothing else separates audio by itself, since it can
  make clean recordings worse.
- **Label formats** — HTK `.lab`, TextGrid, DiffSinger `transcriptions.csv`, Audacity, JSON.
- **Post-processing rules** — e.g. English fixes (`dx`, `uh r → er`, merging short `hh` and duplicates).

Everything heavy is downloaded **only when needed**: each engine lives in its own isolated Python
environment (created with [uv](https://docs.astral.sh/uv/) on first use), and models are downloaded from
catalogs on first use. If you never use GAME, nothing of it is ever downloaded.

## Install

```
uv tool install git+https://github.com/Megageorgio/mVocalToolkit
mvt serve
```

Open http://127.0.0.1:8765/docs for the interactive API documentation.

### Updates

`mvt serve` keeps itself up to date: when it starts, it looks (at most every three hours) whether the source it was
installed from — a GitHub branch or a local folder — has something newer. If so, it updates itself with uv and starts
again with the same arguments; the old process exits with code 75, so a program that started it should wait for
`/health` to answer again. `--no-update`, `MVT_NO_UPDATE=1` or `auto_update: false` in `config.yaml` turn this off.
`POST /shutdown` stops a server (programs use it when they close).

Everything is stored in one folder (`~/mVocalToolkit` by default, change with `--home` or `MVT_HOME`):

```
config.yaml   models/   envs/   sources/   cache/   jobs/   uploads/   outputs/   catalogs/
```

### Remote access (phone, another PC)

```
mvt serve --host 0.0.0.0
```

A token is generated and printed on first start (`mvt config token` makes a new one). Clients send it as
`Authorization: Bearer <token>`. Requests from the same machine don't need it.

## Command line

Every operation is available both as an API method (for GUIs) and as a command (for scripts); the CLI runs the
same code in-process, without a server:

```
mvt languages --task align                                          # languages and their models
mvt models list --task align --lang ru
mvt label ./corpus -m sofa-ru-hhskt-v0.0.1 -f htk,textgrid,ds_csv
mvt label ./corpus -m hubertfa-zh-ja-en-v0.0.7 -l ja                # a multilingual HubertFA model
mvt label ./corpus -m tifa-1.0-st -l zh -L en                       # TIFA: Chinese lyrics with English words
mvt label ./corpus -m tifa-ru-hhskt-v0.0.1 --breaths --split           # TIFA Russian: breaths, long files at pauses
mvt label ./corpus -m sofa-pack-tgm_sofa_en                         # English model and its fixes
mvt transcribe ./corpus -l ja                                       # writes .txt next to the audio
mvt segment ./corpus -m wfl-asr-ft-en-ja
mvt label ./corpus -m sofa-ru-hhskt-v0.0.1 --refine mrefiner-ru-v0.1.0    # + boundary refinement
mvt refine ./corpus -m mrefiner-ru-v0.1.0 -o ./refined              # refine labels next to the audio
mvt midi song.wav --tempo auto
mvt pitch ./corpus -m rmvpe -f csv
mvt separate song.mp3 --stems vocals                                # only when the audio needs it
mvt models import --engine sofa --path my_model.zip                 # a local model (folder or zip)
mvt engines list | install <name> | remove <name> | info <name>
mvt convert a.TextGrid a.lab
mvt g2p "hello world" -m hubertfa-zh-ja-en-v0.0.7 -l en             # words -> phonemes, guesses for unknown words
mvt words add sofa-ru-hhskt-v0.0.1 мурмур m u r m u r               # own words of a model, used by every request
```

`label` transcribes files that have no text (`<name>.txt` or `<name>.lab` next to the audio is used as lyrics
if present), normalizes the text for the language, converts it to phonemes, aligns, applies rules and writes
the labels (by default into `htk/`, `textgrid/` sub-folders next to the audio).

## API overview

All long operations are **jobs**: the request returns a job, progress comes from `GET /jobs/{id}` (long
polling with `?wait=`) or `WS /jobs/{id}/events`, `POST /jobs/{id}/cancel` stops it.

| Method | What it does |
|---|---|
| `GET /health` | version, GPU — for a "Check" button |
| `GET /capabilities` | engines, formats, rule sets, languages |
| `GET /engines`, `POST /engines/{name}/install` | engines and their environments |
| `GET /languages?task=align` | languages with their models — the "language → model" choice of a GUI |
| `GET /models?task=&language=`, `POST /models/{id}/download`, `POST /models/import` | catalog, installed models |
| `POST /files` | upload audio (when the server is on another machine) |
| `POST /transcribe` | audio → text (+ tokens for alignment) |
| `POST /align`, `POST /pipelines/label` | audio + text / words / phonemes (or nothing → transcribe) → label files |
| `POST /segment` | audio → phonemes without text (WFL-ASR) |
| `POST /refine` | ready labels → the same phonemes with refined boundaries (also `"refine"` in `/align`, `/segment`) |
| `POST /midi/extract`, `POST /tempo` | notes, BPM |
| `POST /pitch` | f0 curves |
| `POST /separate` | vocal separation (explicit only) |
| `POST /text/normalize`, `/text/g2p`, `/text/validate` | text tools, unknown words check before a long run |
| `POST /convert` | label format conversion |

The three ways to align (`POST /align`):

1. Send the lyrics: `{"items": [{"path": "a.wav", "text": "..."}]}`.
2. Send nothing — the server transcribes first.
3. Transcribe with `/transcribe`, let the user correct the text in the GUI, then align with the corrected
   text. Or the same in one job: `"review_transcription": true` pauses the job after transcription; the GUI
   shows `pause_data` and continues with `POST /jobs/{id}/resume` and the corrected texts.

See [docs/API.md](docs/API.md) for details and examples.

## Models and catalogs

Models come from catalogs (JSON). The built-in one contains Whisper models, Russian SOFA, the SOFA model
packs, HubertFA, TIFA, WFL-ASR, GAME, separation and pitch models.

How a GUI uses them: it asks `GET /languages?task=align`, shows the languages, then the models of the chosen
language (e.g. `person1-ru`, `hhskt-ru` for Russian), and starts `POST /align` with `"model": "<id>"`. If the
model isn't installed yet, the job downloads it first (progress is part of the job), the engine environment is
created on first use the same way. Nothing has to be installed in advance. Add your own with `POST /catalogs` (URL or path) or by putting a JSON file into
`<home>/catalogs/`. Sources can be direct URLs, GitHub release assets (by file name pattern) or Hugging Face
files. Archives are unpacked and the model files are detected automatically; archives with several models
(packs) register every model inside.

A pack (one archive with several models) can list its models in the catalog, so they are offered by language
before the pack is downloaded; using any of them downloads the whole pack.

Model folders (detected automatically inside archives):

```
SOFA:      model.ckpt + dict.txt [+ g2p/cfg.yaml + g2p/model.ptsd]
           model.safetensors + dict.txt + vocab.yaml + train_config.yaml + global_config.yaml [+ g2p/...]
HubertFA:  model.onnx + vocab.json + config.json + VERSION + dictionaries (languages are read from vocab.json)
TIFA:      model.pt + config.yaml + vocabulary.json [+ dictionaries/ + assets/] (languages: prefixes in vocabulary.json)
```

## Development

```
uv sync --extra dev
uv run pytest
```

Engines are folders in `src/mvocaltoolkit/engines/<name>` with an `engine.toml` (Python version,
requirements, pinned upstream source) and a `worker.py` that talks JSON lines over stdin/stdout.

## License

MIT. Engines and models have their own licenses (see `engine.toml` and the model pages).
