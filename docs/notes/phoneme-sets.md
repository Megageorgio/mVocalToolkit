# Phoneme sets: MCFL, DiffTrainer and cross-language merges

2026-10-14. Compared: MCFL multi13 (`model.json` of MCFL_multi13_s_v0.0.1: `lang_phonemes`, `phoneme_merge_map`,
`shared_labels`), DiffTrainer's default dictionaries and `merged.yaml`
(github.com/agentasteriski/difftrainer-dictfiles), the DiffSinger Best Practices and the EN+JA community guide
(github.com/nobodyP/english_japanese_diffsinger_guide). IPA values below are from general phonetics, not measured.

## What each one has

**DiffTrainer.** One "phoneme → itself" dictionary per language, in the OpenUtau phonemizer names: en (ARPAbet + dx, q),
ja (romaji), zh (opencpop-style finals), ko, es, fr, th. No Russian. `merged.yaml` has 14 consonant groups for en/ja/zh
only (b, ch/ch/q, d, g, k, m, n, p, s, sh/sh/x, t, w, y, z); the file itself says they are "not great merges", only a
format reference.

**MCFL multi13.** 13 languages (de el en es fr it ja ko nan pt ru yue zh), one unified label set, and a
`phoneme_merge_map` "unified ← native name per language". In effect it is already a full cross-language merge table,
and it already follows the community guide: en `dx` = ja `r` → `r`, en `r` → `rh`, ja `j` = en `jh`, ja/ko/fr/... `h`
→ `hh`, `y` of every language → `j`. Russian palatalised consonants share names with the Japanese ones (`by gy ky my
ny py ry hy dy ty`), so they merge for free: a good ru↔ja bridge. Shared non-lexical labels: `AP SP cl vf exh`.

## Problems found

1. **Sibilants are grouped by spelling, not sound.**
   - unified `sh` = en/fr/pt/it/de `sh` (ʃ) + ja `sh` (ɕ) + zh `x` (ɕ) + ru `sh` (ш, ʂ);
   - zh `sh` (ʂ) is separate (`zh_sh`), ru `shy` (щ, ɕː) is separate.
   Russian ш is closest to Mandarin sh, and щ to Mandarin x and Japanese sh. Better groups:
   ʃ = en/fr/pt/it/de sh; ɕ = ja sh + zh x + ru shy; ʂ = ru sh + zh sh.
2. **Affricates the same way.** unified `ch` = en/it/es ch (tʃ) + ru ch (ч, tɕ) + zh q (tɕʰ, aspirated) + ko j (tɕ,
   lax) + de tsh. DiffTrainer merges en ch / ja ch / zh q too. Better: tʃ = en/it/es ch, de tsh; tɕ = ru ch + ja ch +
   zh j + ko j; tɕʰ = zh q + ko ch.
3. **Voicing and aspiration of stops.** Mandarin `b d g` are voiceless unaspirated, `p t k` aspirated; Russian `b d g`
   are fully voiced, `p t k` unaspirated. Both MCFL (same names) and DiffTrainer merge zh b with en/ja b, etc. For a
   labeller this helps; for synthesis a Russian singer would get Mandarin-like devoiced b/d/g and vice versa. For
   DiffSinger: don't merge zh stops with ru/ja/en ones (or merge ru p/t/k with zh b/d/g, which is the closer match).
4. **ʐ vs ʒ.** unified `zh` = en zh / fr j / pt j (ʒ) + zh r (ʐ~ɻ) + ru zh (ж, ʐ). Same issue as 1: ru zh fits zh r
   better than en zh.
5. **Vowels.** MCFL merges vowels across languages a lot (a, i, u, o, e and the reduced ones). The guide advises keeping
   vowels separate because accent lives in them. DiffSinger's `use_lang_id` keeps merged phonemes language-aware, so
   merged vowels are acceptable for synthesis only with `use_lang_id: true`.
6. **Dataset-specific junk labels** in MCFL: `vtrash ctrash xxx scratchy mlem`. Must not reach a DiffSinger dataset:
   map to AP/SP or cut the phrase (consistent with the "never guess" rule of the multilang dataset).
7. **Name mismatches between MCFL output and DiffTrainer dictionaries.** Examples: zh `ian van vn En` are in MCFL but
   not in DiffTrainer's zh dictionary; MCFL en has `tr dr a e x q` and the shared labels, DiffTrainer en is plain
   ARPAbet; ja `j`/`y` differ (MCFL unified `jh`/`j`). A dataset labelled by MCFL can't go into a DiffTrainer config as
   is: binarizing fails on unknown phonemes. Needs an explicit conversion step.
8. **No Russian in DiffTrainer.** The MCFL ru set (with `ax ex x y`, palatalised pairs, `shy`) can be the Russian
   DiffSinger dictionary directly.

## Proposals

1. **One IPA table per language as the single source of truth**: every native phoneme of every language with its IPA
   (and a few features: voicing, aspiration, place). From it generate:
   - the MCFL/WFL `phoneme_merge_map` (loose: merge by near-identical sound, good for recognition), and
   - DiffSinger `merged_phoneme_groups` (strict: merge only identical sounds, never across voicing/aspiration).
   Today these two needs are mixed in one map. The table also gives the "phoneme coverage" report from the trainer
   notes: which target-language phonemes a singer has in merged form.
2. **Fix the three groups above** (ʃ/ɕ/ʂ, tʃ/tɕ/tɕʰ, ʐ/ʒ) in the next MCFL/WFL training; for labelling it costs little
   and makes the labels closer to what a DiffSinger dictionary needs. Data files don't change, only the map.
3. **Non-lexical labels as DiffSinger global phonemes**: `AP SP` reserved, `cl vf exh` as `extra_phonemes` (DiffSinger
   itself uses `EP GS VF` as examples of global ones); the junk labels are converted or cut before export.
4. **A converter "toolkit labels → DiffSinger dictionary"** in the trainer, with a report of unknown phonemes and a
   suggested mapping (from the IPA table), instead of failing at binarizing.
5. **Russian DiffSinger dictionary** from the MCFL ru set, shipped with the trainer, with suggested merges to ja (the
   palatalised pairs, `ts`, `ch`→tɕ group) and en.

Open question: whether stricter groups would hurt MCFL accuracy on small languages (needs one training run to check).

## Russian (hhskt) set with IPA

Confirmed by Georgii, 2026-10-14. Vowels: `a i u e o y` = а и у э о ы (stressed); reduced: `ax` = unstressed а/о
(вОда, ɐ~ə), `x` = unstressed и/э after a soft consonant (прИвет, ɪ), `ex` = unstressed ы/э after a hard one
(Эфир, ɨ~e). Hard consonants `b v g d z k l m n p r s t f h sh ts zh` (х = `h`, ш = `sh` ʂ, ц = `ts`, ж = `zh` ʐ);
soft `by vy gy dy zy ky ly my ny py ry sy ty fy hy`, plus `shy` = щ (ɕː), `ch` = ч (tɕ), `j` = й. Non-lexical:
`SP AP vf cl`.

This confirms the proposals above: ru `ch` belongs with ja `ch` / zh `j` (tɕ), ru `shy` with ja `sh` / zh `x` (ɕ),
ru `sh` and `zh` with zh `sh` and `r` (ʂ, ʐ).
