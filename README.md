# mVocalToolkit

A local API server for automatic labeling of singing voice data. One toolkit, used by any GUI
(desktop, mobile, web, scripts) through HTTP:

- **Transcription** of lyrics — WhisperX (faster-whisper, batched, Silero VAD to skip silence and noise,
  automatic batch reduction on GPU out-of-memory).
- **Forced alignment** — SOFA (dictionaries, G2P models for unknown words, breath (AP) detection, `.ckpt` and
  `.safetensors` models, SOFA model packs), HubertFA (ONNX, multilingual models, breath and other
  non-lexical sound detection) or TIFA (speech and singing, several languages in one text, picks the
  pronunciation heard in the audio, self-check scores per file). The engine is chosen by the model.
- **Phoneme segmentation without text** — WFL-ASR, MCFL (a small ONNX model, fast on a CPU).
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

Training is also available through `/training` and `mvt train`: DiffSinger acoustic/variance and experimental
PC-NSF-HiFiGAN fine-tuning with the pinned `singingvocoders` engine. Training runs share a queue, survive the window
closing, and support stop/resume, protected checkpoints, validation samples and exports. Vocoder fine-tuning needs
clean original 44.1 kHz audio and the full OpenVPI training base (CC BY-NC-SA 4.0). Its source imports and real data
preparation are checked; a full GPU training/export cycle still needs validation. See [training API](docs/API.md).

The `g2p_rnnt` trainer learns pronunciations from a dictionary, with word-disjoint validation, PER/WER reports,
early stopping, base-model fine-tuning and optional vocabulary expansion. It exports SOFA-style weights and a
greedy ONNX transducer with its vocabularies. Small CPU training/fine-tuning/resume/export checks passed; useful
pronunciation quality needs a substantial, consistent dictionary. Custom OpenUtau vocabularies need phonemizer
integration.

Training data preparation includes pause segmentation, phoneme grouping, GAME note estimation, `.ds` import and
creation, and RMVPE/Parselmouth/Harvest f0. DiffSinger supports guided model options, prepare-only runs, chained
variance/acoustic rendering, independent voicebank heads, frozen export controls, speaker-filtered checkpoint
copies and an optional managed local TensorBoard viewer. These operations are exposed through the training API.

Persisted `/training/workflows` chains optional source processing, WFL labelling/conversion, segmentation,
dataset checks, acoustic/variance training, ONNX export, OpenUtau assembly and an optional exported-bank preview.
Short prototypes cap training steps; completed stages and checkpoints are reused on explicit resume. The review
gate rechecks edited data. Sources remain separate from outputs. Per-record content caches, selected-entry cleanup,
speaker/language coverage and optional signal heuristics support repeated preparation. SOME is excluded.

SingForge bank workflows also exclude boundary refinement, including old saved `preparation.refine` and nested
alignment options. New boundary-refiner training jobs are rejected; independent labelling/refinement APIs stay
compatible for other clients. Guided compute controls expose native precision, accumulation, loader/binarizer
processes and future checkpoint rotation. ONNX previews can reuse graphs within a request and limit CPU threads.

The API also imports DiffTrainer/config/checkpoint folders without inspecting pickled checkpoint objects,
previews external checkpoints in separate workspaces and exposes pinned main/benchmark sources. The `bank_render`
engine executes actual exported ONNX graphs on CPU with a compatible local vocoder and DS; it does not execute
OpenUtau text plugins. Conversion examples support interval listening and optional exact user-supplied IPA/feature
merge suggestions. Synthetic execution checks do not establish real bank compatibility or voice quality.

Phoneme audition tools generate deterministic CV/VC/V/CVC or manually entered syllables from an actual model
vocabulary, with aligned notes/durations/f0. Workflows can include their preview, vocoder fine-tuning and G2P
training/export. DiffSinger offers opt-in free-VRAM batch caps, bounded batch reduction/retry after OOM, and
persisted validation-check early stopping. Speech training splits per speaker/language and preserves separate
speaker references; local exported speech previews use the matching native engine. RVC can hold out entire
recordings and exclude their features from the training/index, reporting spectral comparisons and listening pairs.

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
mvt label ./corpus -m tifa-ru-hhskt-v0.0.1 --breaths --split         # TIFA Russian: breaths, long files at pauses
mvt label ./corpus -m sofa-ru-hhskt-v0.0.1 --split                   # SOFA: long files aligned again by segments cut at pauses
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
mvt datasets list --lang ja                                         # public datasets for training, with licenses
mvt datasets download <id>                                          # shows the license, downloads after you accept it
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
| `GET /datasets`, `POST /datasets/{id}/download` | public singing datasets for training, each with its license |
| `POST /files` | upload audio (when the server is on another machine) |
| `POST /transcribe` | audio → text (+ tokens for alignment) |
| `POST /align`, `POST /pipelines/label` | audio + text / words / phonemes (or nothing → transcribe) → label files |
| `POST /segment` | audio → phonemes without text (WFL-ASR, MCFL) |
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

Models come from catalogs (JSON). The main one lives in
[mVocalToolkit-catalog](https://github.com/Megageorgio/mVocalToolkit-catalog) and is read again at every start, so
new models appear without a new toolkit; offline the last copy is used, else the one inside the toolkit. The built-in one contains Whisper models, Russian SOFA, the SOFA model
packs, HubertFA, TIFA, WFL-ASR, MCFL, GAME, separation and pitch models.

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

The `rvc_voice` trainer learns RVC v2 voice conversion from one speaker of unlabelled WAV/FLAC audio, using
ContentVec, Harvest and HiFi-GAN. It supports catalog/local training bases, bounded sequential preparation,
prepare-only mode, cooperative batch stop, optimizer/position resume, protected checkpoints and PTH/FAISS export.
Native CPU preparation, training, stop/resume and strict loading of the exported generator have been checked.
Full GPU convergence and production voice quality remain unverified.

Optional trainers also cover TIFA, SOFA, WFL-ASR v2, MCFL distillation, mRefinerModel, GAME, Qwen3-TTS and
VoxCPM2 (full or LoRA). They use frozen recording manifests, recording-disjoint validation, native model
networks, checkpoint/optimizer resume, cooperative stop and inference export. No training environment or base
weight is installed by preflight. See [the training API](docs/API.md#additional-model-trainers) for kinds,
input formats and current verification limits.

Recording preparation also accepts raw WAV through WFL-ASR, with primary/additional languages, explicit native
or shared-canonical multi13 conversion, per-language phonemizer inventories and conversion reports. Pause cuts
can cap edge silence and fade only within labelled silence; existing DS f0 can be preserved. Read-only preflight
starts no workers and downloads no weights. OpenUtau assembly writes expanded/stock phonemizer remappers with
coverage reports, including preferred hhskt-ru and approximate stock Russian compatibility. See
[recording preparation](docs/API.md#recording-preparation-and-diffsinger-settings) for requests and limits.

Dataset coverage can generate configurable per-singer/language phoneme recording exercises (CV/VC/V/CVC), with
repetition accounting, manual alphabets and remaining-deficit reports. This is read-only planning, without model
training or text phonemizers. G2P/labeller training remains independent of ordinary DiffSinger bank training;
explicit legacy auxiliary G2P workflow stages remain supported. See [recording plans](docs/API.md#recording-plans-from-dataset-coverage).
