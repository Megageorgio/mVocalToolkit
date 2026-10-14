# DiffTrainer v0.4.10: what it is inside

2026-10-14. Read from a fresh clone of https://github.com/agentasteriski/DiffTrainer (`D:\Dev\repos\DiffTrainer-latest`,
last commit 8d630e1, 2026-10-05). Not run: no environment was set up. Companion to
[diffsinger-trainer.md](diffsinger-trainer.md).

## Size and shape

- ~4000 lines of Python: `difftrainer.py` 1713 (the whole GUI and most logic in one class), `quickinference.py` 376,
  `dt_modules/` 1650 (export, `.ds` creation, lab→csv, corpus segmenter, pitch), `auto_torch.py`, `check_update.py`.
- One main author, 670 commits, ~105 of them in 2026. No tests, no CI, no releases (updates come from the main branch).
- UI: CustomTkinter, six tabs: About, Data Preparation, Configuration, Binarize & Train, Export (basic), Export
  (advanced). Translations: en, de (others were dropped in 0.4.0 as outdated).

## What it does, step by step

1. **Install.** User installs the CUDA Toolkit (11.8–12.9) and Miniconda by hand, then `reconfigure_conda_env.bat`
   (removes and recreates the `difftrainer` env from `assets/environment.yml`) and `auto_torch.py`, which picks a torch
   wheel from `nvcc --version`: torch 2.4 for CUDA 11.8–12.4, 2.8 for 12.6–12.9, 2.11 for 13.0, unpinned for 13.2.
   Python 3.10, numpy<2, lightning~=2.3, librosa<0.10.
2. **Update Tools.** Downloads zips of the author's DiffSinger fork (`main` or `benchmark` branch, not a pinned commit),
   PC-NSF-HiFiGAN 2025.02, RMVPE, the hnsep vocal remover, the SOME model, SOME-lite scripts and a dictionary repo.
   Before that it **deletes the whole `DiffSinger` folder**. Then it rewrites upstream configs ("secret sauce": RMVPE,
   f0_max 1600, L1 loss, UniPC, pitch-shift augmentation ±3, tension logits ±8) and regex-patches
   `utils/binarizer_utils.py`.
3. **Data preparation.** corpus_segmenter (Ghin's script) cuts long wav+lab at pauses; liteconvert turns HTK `.lab`
   into `transcriptions.csv` with `ph_num` from a vowel/liquid table (nnsvs-db-converter rules); `.ds` creation from csv
   with RMVPE, Parselmouth or Harvest f0 and notes via SOME. Labels must already exist: no alignment, no
   transcription.
4. **Configuration.** Reads speaker folders (folder name → speaker, language tag in the folder name), builds the
   acoustic or variance config: multiple dictionaries through `langloader.yaml` + `merged.yaml` (with an editor
   window), backbone (lynxnet2 / wavenet), mix-LN, dual timestep, stretch embed, variance switches, batch size, save
   interval, augmentation. Writes it into `DiffSinger/configs/`.
5. **Binarize & Train.** Runs `scripts/binarize.py` and `scripts/train.py --reset` as subprocesses in a thread; the GUI
   shows only "running / done / error", all output stays in the console. Stop = `terminate()`. TensorBoard through a
   separate `.bat`. A "tensor patch" inserts `torch.set_float32_matmul_precision('medium')` into the training files.
6. **Export.** ONNX export (optionally dropping speakers), then an OpenUtau folder: dsmain/dsdur/dspitch/dsvariance with
   embeds, character.yaml, dsconfig, an auto-generated `dsdict.yaml` (phoneme types from a Latin-letter table, OpenUtau
   language codes zh/en/de/it/ja/yue/ko/pt/ru/es), the vocoder. "Advanced" lets each model come from a different
   training run.
7. **QuickInference.** A separate window that renders a `.ds` through `scripts/infer.py` with PyTorch checkpoints
   (variance, then acoustic). It does not test the exported ONNX bank.

## What is good (worth taking)

- **Covers the whole DiffSinger side in one window**, including multi-dictionary configs, merged phoneme groups and
  speaker merging by name, which the Colabs still don't support.
- **Tuned defaults from community experience**: the "secret sauce" values, lynxnet2 by default, tension limits. These
  are the cheapest thing to copy into our presets.
- **Practical data helpers**: `.ds` rescue (copies SlurCutter `.ds` from `wavs/` into `ds/`), corpus_segmenter's report,
  dropping speakers at export, the advanced export mixing models from different runs.
- **Folder conventions** users already know: one folder per speaker, language tag in the name.

## Weak spots (what we can do better)

### Install and updates
- The CUDA Toolkit and conda are manual prerequisites; torch is chosen from `nvcc`, so a machine with a driver but no
  toolkit gets the wrong wheel. Our toolkit already picks torch through uv's `--torch-backend` with no toolkit needed.
- DiffSinger is tracked by **branch, not commit**: every "Update Tools" can bring a different trainer; the update check
  compares a GitHub commit date with a local file's mtime.
- **Update Tools deletes the `DiffSinger` folder**, and the checkpoint folder dialog **opens inside
  `DiffSinger/checkpoints` by default**; configs are also written into `DiffSinger/configs/`. The dialog warns to back
  up, but the default layout puts the user's trained models and configs exactly where the update wipes them.
- Upstream code is changed at run time (config rewrites, a regex on `binarizer_utils.py`, line inserts into the
  training tasks): this is the fragility DiffTrainer #45 complains about.

### Feedback while running
- Training output only goes to the console: no step, loss, ETA or VRAM in the window, errors appear as "error, see
  console". The CUDA check only prints `nvcc --version` and doesn't stop anything (DiffTrainer #49 is this).
- Stop is `terminate()` on the parent process; on Windows dataloader workers can outlive it.
- No checkpoint listening, no "when to stop" help: TensorBoard is a separate `.bat`.

### Correctness traps
- Without "advanced" naming, a config is written **over `DiffSinger/configs/acoustic.yaml` / `variance.yaml`**, the
  templates it reads next time, so settings leak from one config into the next.
- `test_prefixes` are 3 **random** files per speaker on every config write (validation changes between runs), and a
  speaker with fewer than 3 files is an error.
- `num_spk` is `max(spk_id) + 1`, so gaps in ids create unused speakers.
- `ph_num` and dsdict phoneme types come from hardcoded Latin-letter tables: fine for romaji-like sets, wrong for
  multi-letter or non-Latin phonemes (Russian `sch`, `y`, palatalised consonants) unless the language JSON is edited.
- `--reset` is always passed to training, so resuming silently takes the current config file, not the one the run
  started with.

### Missing
- No alignment, transcription, separation, refinement or label editing: you arrive with finished wav+lab.
- No dataset checks (minutes, phoneme coverage, clipping, band), no config validation before binarizing.
- No base-model fine-tuning workflow, no vocoder fine-tuning.
- No render test of the exported ONNX bank, no linter for the OpenUtau folder (the multi-language `languages` failures,
  DiffTrainer #46).
- No remote/headless mode: the GUI and training are one process on one machine; no CLI.
- macOS is "on your own"; Linux needs manual steps.

## What this means for our trainer

Take: the tuned defaults, multi-dictionary/merged-group config logic, speaker/language folder conventions, `.ds`
rescue, speaker dropping and mixed-run export. Do differently: pinned engine instead of branch zips and patches,
user data outside the engine folder, events with steps/losses/ETA instead of the console, a fixed validation set,
dictionary-driven `ph_num` and phoneme types, ONNX render test and an export linter, alignment from the toolkit in
front of it all. Everything in the "Missing" list above is already in `diffsinger-trainer.md`, sections 3–4.
