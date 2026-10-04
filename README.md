# mVocalToolkit

A local API server for automatic labeling of singing voice data. One toolkit, used by any GUI
(desktop, mobile, web, scripts) through HTTP:

- **Transcription** of lyrics — WhisperX (faster-whisper, batched, Silero VAD to skip silence and noise,
  automatic batch reduction on GPU out-of-memory).
- **Forced alignment** — SOFA, with dictionaries, G2P models for unknown words, breath (AP) detection,
  `.ckpt` and `.safetensors` models, LabelMakr model packs.
- **Phoneme segmentation without text** — WFL-ASR.
- **Notes (MIDI)** — GAME, with automatic tempo estimation (DeepRhythm).
- **Label formats** — HTK `.lab`, TextGrid, DiffSinger `transcriptions.csv`, Audacity, JSON.
- **Post-processing rules** — e.g. LabelMakr's English fixes (`dx`, `uh r → er`, merging short `hh` and duplicates).

Everything heavy is downloaded **only when needed**: each engine lives in its own isolated Python
environment (created with [uv](https://docs.astral.sh/uv/) on first use), and models are downloaded from
catalogs on first use. If you never use GAME, nothing of it is ever downloaded.

## Install

```
uv tool install git+https://github.com/Megageorgio/mVocalToolkit
mvt serve
```

Open http://127.0.0.1:8765/docs for the interactive API documentation.

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

The CLI runs the same operations without a server:

```
mvt models list --engine sofa
mvt label ./corpus -m sofa-ru-hhskt-v0.0.1 -f htk,textgrid,ds_csv
mvt label ./corpus -m labelmakr-<model> --rules en_fixes           # English, LabelMakr fixes
mvt transcribe ./corpus -l ja                                       # writes .txt next to the audio
mvt segment ./corpus -m wfl-asr-test1
mvt midi song.wav --tempo auto
mvt models import --engine sofa --path my_model.zip                 # a local model (folder or zip)
mvt engines list | install <name> | remove <name> | info <name>
mvt convert a.TextGrid a.lab
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
| `GET /models`, `POST /models/{id}/download`, `POST /models/import` | catalog, installed models |
| `POST /files` | upload audio (when the server is on another machine) |
| `POST /transcribe` | audio → text (+ tokens for alignment) |
| `POST /align`, `POST /pipelines/label` | audio + text / words / phonemes (or nothing → transcribe) → label files |
| `POST /segment` | audio → phonemes without text (WFL-ASR) |
| `POST /midi/extract`, `POST /tempo` | notes, BPM |
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

Models come from catalogs (JSON). The built-in one contains Whisper models, Russian SOFA, the LabelMakr model
packs, WFL-ASR and GAME models. Add your own with `POST /catalogs` (URL or path) or by putting a JSON file into
`<home>/catalogs/`. Sources can be direct URLs, GitHub release assets (by file name pattern) or Hugging Face
files. Archives are unpacked and the model files are detected automatically; archives with several models
(packs) register every model inside.

SOFA model folder (any of these layouts works):

```
model.ckpt + dict.txt [+ g2p/cfg.yaml + g2p/model.ptsd]
model.safetensors + dict.txt + vocab.yaml + train_config.yaml + global_config.yaml [+ g2p/...]
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
