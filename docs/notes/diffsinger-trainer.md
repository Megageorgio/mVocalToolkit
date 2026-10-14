# A DiffSinger trainer on top of mVocalToolkit: research and design

2026-10-14. Research and design only, no code yet. Sources: GitHub issues (DiffTrainer, openvpi/DiffSinger,
MakeDiffSinger, SOME, SingingVocoders, SOFA, the MLo7 Colab), the DiffTrainer v0.4.10 README and changelog, the OpenUtau
and MakeDiffSinger wikis, community guides, ACE Studio docs, Applio/RVC. Reddit and UtaForum can hardly be read (they
block bots), so most complaints below come from GitHub; most community support happens on Discord/QQ/Feishu, so issue
counts are a sample, not a real frequency. Earlier notes on the topic: `notes/diffsinger-research/REPORT.md`,
`notes/diffsinger-speedup-brainstorm.md`, `notes/diffsinger-next-gen-ideas.md` (project files).

## 1. What exists

| Tool | What it does | State |
|---|---|---|
| [agentasteriski/DiffTrainer](https://github.com/agentasteriski/DiffTrainer) | CustomTkinter GUI: cutting wav+lab, lab→csv, building `.ds` (RMVPE/Parselmouth/Harvest), SOME, config builder (multiple dictionaries, languages, merged phoneme groups, lynxnet2/wavenet), binarizing, acoustic and variance training, TensorBoard, ONNX export, OpenUtau packaging, quickinference | v0.4.10 (October 2026), 51 stars, alive. The local copy in `D:\Dev\repos\DiffTrainer` is an old 0.2.x |
| [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger) | The trainer itself: CLI binarize / train / export / infer | Very active, V3 in progress; config and dsconfig formats change often |
| [openvpi/MakeDiffSinger](https://github.com/openvpi/MakeDiffSinger) | Dataset scripts: MFA, build_dataset, estimate_midi, convert_ds, variance-temp-solution | Wiki from 2023 |
| [openvpi/dataset-tools](https://github.com/openvpi/dataset-tools) | AudioSlicer, MinLabel, SlurCutter, FoxBreatheLabeler (Qt) | Crashes on several systems |
| SOME / GAME / SOFA / HubertFA | Notes, alignment | Separate tools; GAME is replacing SOME in the Colabs |
| [SingingVocoders](https://github.com/openvpi/SingingVocoders) | Vocoder training and fine-tuning | No requirements.txt, little documentation |
| Colabs: [MLo7](https://github.com/MLo7Ghinsan/DiffSinger_colab_notebook_MLo7) (archived 2025-12), [DiffSinger4Colab](https://github.com/usamireko/DiffSinger4Colab), NevisDiffsingerColab | Training in Colab | Multiple dictionaries unsupported or TODO; config cells break after upstream updates |
| LabelMakr, labbu, cmu2dsdict, nnsvs-db-converter, Voicebank2DiffSinger, genon2wavlab, ColorSplitter, dsinfo, SCRIBE, V2DS-Studio | Labelling, converters, utilities | Scattered |
| Commercial: ACE Studio Voice Cloning | Upload dry vocals (30–100 min), training starts by itself, no labelling; 5 model versions from "rare" to "well-done" | Closed, paid |
| Commercial "done for you" | People label and train DiffSinger for money (UtaForum, carrd commissions) | There is demand for "just train it for me" |

Conclusion: **apart from DiffTrainer, no program covers the whole path from raw recordings to an OpenUtau voicebank**, and
DiffTrainer starts from ready wav+lab, so labelling happens somewhere else. That is exactly the gap mVocalToolkit and
mLabeler already fill.

## 2. What people complain about (by theme, worst first)

1. **Installation and environment** (~16 issues, first everywhere). torch/CUDA/onnx version conflicts: DiffTrainer once
   kept two conda environments because training needed torch>2 and ONNX export torch<2; now torch ≤2.8 and CUDA
   11.8–12.9. `fbgemm.dll` fails to load (DiffTrainer #54, today), onnxruntime reinstalls itself on every export
   (DiffSinger #273), Windows long paths, no macOS support, Linux questions unanswered.
2. **"I trained it, now how do I make it sing?"** (~14 issues, the most comments: DiffSinger #33, #103, #109, #258).
   No simple inference after training; a CLI render sounds different from OpenUtau (#284).
3. **ONNX export and OpenUtau packaging** (~14). onnxsim and checkpoint size errors; wrong values in dsconfig (#186);
   multilingual models fail in OpenUtau on `languages` (DiffSinger #231, DiffTrainer #46); speakers sound the same
   because of embedding paths in character.yaml (#158); unclear where the vocoder goes (#271); upstream keeps changing
   dsconfig keys (#340, #343, #345).
4. **Labelling and dataset preparation** (~18). MFA finds no files, MakeDiffSinger scripts have bugs, SlurCutter and
   FoxBreatheLabeler crash, unclear how to write `ph_seq`, `ph_num`, notes, slurs, SP/AP. A 2025 thesis estimate: 30
   minutes of singing take up to 20 hours of manual labelling.
5. **Config, dictionaries, phonemes: cryptic tracebacks** (~12). `KeyError: 'ph2word'`, `'tension'`; all `predict_*`
   switches off gives a vague Lightning error (#197, a DiffTrainer user); YAML reads the language code `no` as `false`
   (#291); 0 train/valid items because of test_prefixes (#105); reserved AP/SP and the characters `/ - +`.
6. **Hangs and silent failures** (~7). Stuck at "Sanity Checking" with no error (#226); in DiffTrainer "the buttons do
   nothing" and the cause is only in the terminal (#49); out of memory in the middle of training (upstream added a probe
   epoch in 2026, PR #327).
7. **Pretrained bases and the vocoder** (~10). No base models for one's own dictionary; vocoder fine-tuning does not
   converge on band-limited data (SingingVocoders #12); unclear what to do with generator/discriminator weights.
8. **Multiple speakers and languages** (~6). Speakers bleed into each other, small speakers drift towards big ones
   (#261); multi-style and multi-speaker can't be combined (#269).
9. **Documentation** (~8). Dead links, the main docs are in Chinese on Feishu, knowledge is spread over Google Docs,
   carrd, Discord and YouTube. How long to train and when to stop: **no numbers anywhere**, only "watch TensorBoard".
10. **Fragile wrappers** (~5, the main maintenance cost of DiffTrainer). DiffTrainer patches DiffSinger line by line at
    run time (#45); the auto-updater broke twice; the move to multiple dictionaries and v3→v4 needed a clean install.
11. **Data quality** (~5). Does denoising help (#190), band-limited recordings, low notes come out breathy (#73), hum
    in SP/AP (#179).
12. **Colab**. Disconnects, the Google Drive trash fills up and checkpoints stop saving, usage limits, folder layout
    errors only surface in the middle of a run.

How much data is needed (community consensus): **30 minutes per language at least, about an hour for a good result**;
under 10 minutes only with a public corpus mixed in. ACE Studio asks for 30–100 minutes.

## 3. What is missing and what can be done better

Grouped by step. **[have]** means mVocalToolkit or mLabeler already does it and it only has to be wired in.

### Data
- From raw songs: vocal separation, de-reverb **[have]**, cutting at pauses **[have, --split]**, transcription
  **[have]**, SOFA/HubertFA/TIFA alignment **[have]**, mRefiner boundary refinement **[have]**, GAME notes **[have]**,
  RMVPE f0 **[have]**, `transcriptions.csv` export **[have]**.
- Missing: automatic `ph_num` for any dictionary (MakeDiffSinger #11), building `.ds` with notes and slurs, breaths
  (AP) and their check, labelling voice colours/styles.
- **Dataset doctor** before training: minutes per language and per speaker, phoneme coverage of the dictionary (which
  are missing and which donors have them), sample rate, clipping, level, narrow band (matters for the vocoder),
  noise/reverb, wav/lab name mismatches, empty and too short phonemes. mLabeler already has some of these checks
  ("problems that break DiffSinger training"); move them into the toolkit so the CLI has them too.
- A list of doubtful places by aligner confidence (`"confidence": true` exists) that opens mLabeler at that spot.
  Thresholds per aligner: the SOFA and HubertFA scales are not comparable.

### Dictionaries and languages
- A dictionary wizard: from G2P/CMU/a model's own dictionary, language prefixes (`ru/a`), reserved name checks,
  `merged_phoneme_groups` with suggestions (the same IPA in different languages), quoted language codes in YAML.
- A phoneme coverage report per language (idea from `diffsinger-next-gen-ideas.md`).

### Config
- Presets instead of raw YAML: "little / much data", "8 / 12 / 24 GB", "fast / quality"; variance switches with an
  explanation of which shouldn't be on together (energy + breathiness + voicing) and that tension is unstable.
- Config check before binarizing: at least one `predict_*`, test_prefixes exist, every phoneme is in the dictionary,
  paths without spaces and non-ASCII characters.
- **Base models in the catalog** (mVocalToolkit-catalog is already read at every start): fine-tuning by default instead
  of training from scratch, bases per language, choosing a base by phonemes.

### Training
- Environment check before starting: GPU, VRAM, driver, CUDA; subprocess errors shown in the window, not the terminal.
- Batch size from VRAM, a probe epoch on the longest samples (out of memory at once, not after an hour), AMP/bf16.
- A queue: acoustic → variance (duration + pitch + parameters) → vocoder, continuing after a crash or a reboot.
  Training on one PC, watching from a phone (the toolkit already works over the network with a token).
- **A checkpoint ladder like ACE Studio's**: every N steps the same test phrase (in the bank's language) is sung, with
  A/B listening in the program. That answers "when do I stop" without TensorBoard.
- Overfitting hints: validation loss rises while training loss falls → suggest a checkpoint. Own loss graphs in the
  window, TensorBoard as an option.
- Time and step guidelines for typical cases (nobody publishes them; collect them from our own runs and show "about N
  hours left").
- Vocoder fine-tuning with a band check of the data.
- Later (from earlier notes): LoRA adapters on a shared base, rehearsal with base data against forgetting languages,
  training on 8 GB.

### Export and checking the bank
- ONNX export and the OpenUtau folder in one step, with a **linter**: character.yaml + character.txt, dsconfig with
  every key of the current version, UTF-8, phonemizer name, the `languages` input for multilingual banks, speaker
  embedding paths, the vocoder (a reference to the standard one or the bank's own next to it).
- **A test render of the finished bank without OpenUtau**: sing a `.ds`/`.ust`/MIDI right from mLabeler (the toolkit
  already has an NSF-HiFiGAN ONNX engine, mLabeler a piano roll). This answers complaint 2 and proves the export works.
- A bank report (languages, speakers, parameters it supports), like dsinfo.
- A bank archive for sharing; the vocoder licence (openvpi NSF-HiFiGAN weights are CC BY-NC-SA) in the report.

### Surviving upstream changes
- Don't patch DiffSinger at run time: an engine with a **pinned commit** of a fork (as the other engines do in
  `engine.toml`); updating the engine is a separate step with a test "train 200 steps on a tiny dataset and export".
- An own project format (what, from where, which engine versions), so a project can be opened and trained further a
  year later.

### Where to train
- Own PC (Windows/Linux, NVIDIA). macOS: data preparation and rendering only.
- A remote PC or a rented GPU: the same `mvt serve --host 0.0.0.0` on the server, the GUI connects with the token.
- Colab/Kaggle: the toolkit writes a notebook that installs the same engine at the same version, frequent checkpoints
  on Drive, a warning about the Drive trash, continuing after a disconnect.

### Very fast preview training
Three tiers, so the user hears something before committing hours. Times are estimates for a 12 GB card, not
measurements.

1. **Check (1–2 min).** Binarize a few files, 100–200 steps, export, render one phrase. It doesn't sound like the
   singer; it proves the data, dictionary, config and export work before a long run.
2. **Preview (~5–15 min).** Fine-tune only from a multi-speaker base of the same language/phoneme set:
   - acoustic model only; durations and pitch come from the base variance model;
   - a new speaker embedding plus a LoRA (or only the last layers), everything else frozen (DiffSinger already has
     `finetune_enabled` and `frozen_params`);
   - 3–5 minutes of the cleanest clips, short ones, features cached during labelling (RMVPE f0 is the slow part of
     binarizing);
   - optionally train only the auxiliary decoder (shallow diffusion's fast part) first: blurry but already the
     singer's timbre.
   Result: "this is roughly how the bank will sound", enough to judge data and labelling before the full run.
3. **Full.** The normal acoustic + variance training, starting from the preview weights so nothing is wasted.

The catch: tiers 2–3 need a good base per language whose phoneme set covers the singer's dictionary (bases are tied to
their dictionary). That is the one-time job from `diffsinger-research/REPORT.md` (multilingual base, shared phoneme
alphabet). Without training at all, a zero-shot model (SoulX-Singer class) could sing from a 10 s reference, but it's a
different model, so it previews the voice, not the future bank.

## 4. How it fits into mVocalToolkit and mLabeler

The main principle: **all the work is in the toolkit, a GUI only shows it**. Then everything is available from the CLI,
mLabeler, a new window and a phone, and installation is the one that already works (uv, separate environments, downloads
on demand). That alone removes complaint 1: the user never sees conda.

### Toolkit (new)
- An engine `diffsinger` in `src/mvocaltoolkit/engines/diffsinger`: `engine.toml` with Python, CUDA torch and a pinned
  fork commit; a JSON-lines `worker.py` like the others. A separate light engine `diffsinger-onnx` (onnxruntime only)
  for rendering and checking banks, enough without a graphics card.
- An engine `vocoder-train` (SingingVocoders), later.
- API (everything long is a job, as now):

  | Method | What it does |
  |---|---|
  | `POST /train/projects` | A training project: data folders, speakers, languages, base |
  | `POST /dataset/build` | Raw recordings → dataset (a chain of the existing steps + `ph_num`, `.ds`) |
  | `POST /dataset/check` | Dataset doctor: a report of problems with links to files and places |
  | `POST /train/config` | Config from a preset + check |
  | `POST /train` | Binarizing and training; events: step, losses, VRAM, time estimate, new checkpoints and their test renders |
  | `POST /train/{id}/pause`, `/resume` | Pause and continue |
  | `POST /export/openutau` | ONNX + the bank folder + linter |
  | `POST /render` | Sing a `.ds`/`.ust`/MIDI with a finished bank (ONNX) |

- CLI: `mvt dataset build`, `mvt dataset check`, `mvt train`, `mvt export`, `mvt render`.
- Catalog: model types `ds_base` (bases for fine-tuning) and `ds_vocoder`.

### mLabeler
- Already there: labelling, DiffSinger checks, dataset export, piano roll, vocoder. To add: opening a problem from the
  doctor's report at its place, editing `.ds` (notes, slurs) and trying the bank on one's own `.ds`.

### The trainer's own GUI
Recommended: **a separate app from the same codebase as mLabeler** (Compose Multiplatform, the shared `core` module): a
wizard "project → data → check → training → listening → export", graphs, the checkpoint ladder. Reasons: the trainer
has its own rhythm (hours of waiting, watching from a phone) and mLabeler is already big, while the toolkit client,
players, themes and translations are shared, and one can jump between the programs ("fix in mLabeler"). If it turns out
cheap, it can be a mode inside mLabeler instead; the decision can wait, the toolkit doesn't depend on it.

## 5. Name

The family is descriptive (mLabeler, mVocalToolkit, mVBConvert), so a descriptive name fits best:

- **mTrainer**: recommended. Short, obvious, sits next to mLabeler. Downside: very generic for searching.
- **mVBTrainer**: pairs with mVBConvert, says it trains voicebanks.
- **mSingerForge** / **mVoiceForge**: more of a brand, about building a whole voice, not just training.
- **mBankMaker**: honestly describes the whole path from recordings to a bank.

Better avoided: "mDiffTrainer" (confused with DiffTrainer), "mForge" (associated with Stable Diffusion Forge), "mBank"
(the local folder `D:\Dev\repos\mbank` is another project).

## 6. Where to start (proposed order)

1. The `diffsinger` engine with a pinned version and `mvt train` on a tiny dataset: binarize → 200 steps → export. This
   tests the main thing: installing without conda on Windows.
2. `mvt dataset check` (the doctor) and `mvt export openutau` with the linter: the most visible benefit even for people
   who train in DiffTrainer or Colab.
3. `mvt render` (ONNX rendering of a bank) and test renders of checkpoints.
4. `mvt dataset build` from raw songs as a chain of existing steps.
5. The GUI wizard.
6. Bases in the catalog, the vocoder, LoRA and the rest from the earlier notes.

## Main issues referenced

- Installation: https://github.com/agentasteriski/DiffTrainer/issues/54, https://github.com/agentasteriski/DiffTrainer/issues/49, https://github.com/openvpi/DiffSinger/issues/273
- Inference after training: https://github.com/openvpi/DiffSinger/issues/103, https://github.com/openvpi/DiffSinger/issues/258
- Export and OpenUtau: https://github.com/openvpi/DiffSinger/issues/231, https://github.com/agentasteriski/DiffTrainer/issues/46, https://github.com/openvpi/DiffSinger/issues/186
- Labelling: https://github.com/openvpi/MakeDiffSinger/issues/5, https://github.com/qiuqiao/SOFA/issues/24
- Config: https://github.com/openvpi/DiffSinger/issues/197, https://github.com/openvpi/DiffSinger/issues/291
- Hangs: https://github.com/openvpi/DiffSinger/issues/226
- Vocoder: https://github.com/openvpi/SingingVocoders/issues/12
- Multiple speakers: https://github.com/openvpi/DiffSinger/issues/261
- Fragile wrapper: https://github.com/agentasteriski/DiffTrainer/issues/45
- Hardware requirements: https://github.com/openvpi/MakeDiffSinger/wiki/System-Requirements
- ACE Studio: https://docs.acestudio.ai/voice-cloning/cloning-a-voice-for-vocal-synth
