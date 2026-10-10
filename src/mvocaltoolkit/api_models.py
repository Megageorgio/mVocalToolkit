"""Request and response models of the API (also used by the CLI)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .labels import Label

AUDIO_PATTERNS = ["*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a", "*.aac", "*.opus"]


class InputItem(BaseModel):
    """One audio file. Text can be given in several forms, the most specific one wins:
    phonemes > words > text. Without any of them the server transcribes the audio (if allowed)."""

    path: str | None = Field(None, description="Path to the audio file on the server machine")
    file_id: str | None = Field(None, description="Id of a file uploaded with POST /files")
    name: str | None = Field(None, description="Base name of the output files (default: audio file name)")
    text: str | None = Field(None, description="Lyrics / transcription as plain text")
    words: list[str] | None = Field(None, description="Normalized tokens (words / syllables) for the dictionary")
    phonemes: list[str] | None = Field(None, description="Phoneme sequence (skips G2P)")
    language: str | None = None
    segments: list[tuple[float, float, str]] | None = Field(
        None, description="/refine: the labels to refine, [start, end, phoneme] in seconds (default: the label file next to the audio)"
    )


class InputSpec(BaseModel):
    items: list[InputItem] = Field(default_factory=list)
    folder: str | None = Field(None, description="Process all audio files in this folder")
    recursive: bool = True
    patterns: list[str] = Field(default_factory=lambda: list(AUDIO_PATTERNS))
    # text files next to the audio used as lyrics if present (SOFA corpus layout)
    sidecar_text: list[str] = Field(default_factory=lambda: [".txt", ".lab"])


class OutputSpec(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["htk"], description="htk, textgrid, ds_csv, json, audacity")
    dir: str | None = Field(
        None, description="Output folder. Default: next to the audio (path inputs) or <home>/outputs/<job>"
    )
    layout: Literal["subfolders", "beside", "flat"] = Field(
        "subfolders",
        description="subfolders: <dir>/<format>/<name>.<ext>; beside: <dir>/<name>.<ext>; flat: like beside, "
        "ignoring sub-folders of the input folder",
    )
    overwrite: bool = True
    return_labels: bool = True
    ds_csv_name: str = "transcriptions.csv"


class VadOptions(BaseModel):
    enabled: bool = True
    method: Literal["silero", "pyannote"] = "silero"
    onset: float = 0.5
    offset: float = 0.363
    chunk_size: int = 30


class TranscribeOptions(BaseModel):
    model: str = Field("whisper-large-v3-turbo", description="Catalog id or a faster-whisper model name")
    language: str | None = Field(None, description="Language code, None = auto detect")
    batch_size: int = 8
    compute_type: str = Field("auto", description="auto, float16, int8_float16, int8, float32")
    vad: VadOptions = Field(default_factory=VadOptions)
    initial_prompt: str | None = Field(None, description="Hint for the recognizer, e.g. known lyrics or words")
    hotwords: str | None = None
    suppress_numerals: bool = True
    beam_size: int = 5
    temperature: float = 0.0
    # also convert to tokens for alignment with the text frontend of the language
    frontend: bool = True


class TranscribeRequest(TranscribeOptions):
    input: InputSpec
    # write <name>.txt next to the audio (SOFA corpus style) or into output.dir
    save_txt: bool = False
    output_dir: str | None = None


class PostprocessOptions(BaseModel):
    rule_sets: list[str] = Field(default_factory=list, description="Built-in rule sets, e.g. en_fixes, cleanup")
    rules: list[dict[str, Any]] = Field(default_factory=list, description="Custom rules, see docs/API.md")
    use_model_defaults: bool = Field(
        True, description="Apply the model's default rule sets when rule_sets is empty (false: no fixes)"
    )


class RefineOptions(BaseModel):
    """Refinement of the phoneme boundaries by a boundary refiner model (mRefinerModel)."""

    model: str = Field("mrefiner-ru-v0.1.0", description="Refiner model id or path:")
    mode: Literal["auto", "normal", "safe"] = Field(
        "auto", description="safe: only confident boundaries, not far (other languages / labelling styles); auto: safe when most phonemes are unknown to the model"
    )
    phone_map: dict[str, str] = Field(default_factory=dict, description="Other phoneme names -> the model's ones")
    min_confidence: float | None = None
    max_shift_ms: float | None = None


class AlignRequest(BaseModel):
    input: InputSpec
    model: str = Field(
        ..., description="Aligner model id (SOFA, HubertFA or TIFA, catalog/installed) or path:/folder/of/model"
    )
    language: str | None = Field(
        None, description="Language of the texts (text frontend, dictionary of multilingual models); "
        "default: the model's language"
    )
    extra_languages: list[str] = Field(
        default_factory=list,
        description="TIFA: more languages that may occur in the texts, in priority order (e.g. English words in "
        "Chinese lyrics); their phonemes keep the language prefix (en/aa)",
    )
    optional_breaths: bool = Field(
        False, description="TIFA: the aligner may place a breath (AP) at the start, at SP marks and after "
        "punctuation when it hears one; breaths it doesn't hear are left out"
    )
    split_silence: bool = Field(
        False, description="TIFA and SOFA: files longer than split_max_length are aligned again in pieces cut only at "
        "pauses (clear silence or a breath), each piece with its own words; a cut never falls inside a word"
    )
    split_max_length: float = Field(25.0, gt=1, description="split_silence: longest piece in seconds when "
                                    "the pauses allow it")
    split_min_silence: float = Field(0.3, gt=0, description="split_silence: shortest silence to cut at, seconds")
    mode: Literal["force", "match"] = Field("force", description="SOFA only")
    g2p: Literal["auto", "dictionary", "none"] = Field(
        "auto", description="auto: dictionary + the model's G2P for unknown words; none: tokens are phonemes"
    )
    ap_detector: Literal["loudness_spectral_centroid", "none"] = Field(
        "loudness_spectral_centroid", description="SOFA: breath detection; none disables it"
    )
    non_lexical_phonemes: list[str] = Field(
        default_factory=lambda: ["AP"],
        description="HubertFA: non-lexical phonemes to detect (AP = breath, EP = other sounds); [] disables",
    )
    pad_times: int = Field(1, description="HubertFA: number of passes with different padding (more = steadier)")
    pad_length: float = Field(5.0, description="HubertFA: max padding in seconds for the extra passes")
    dictionary: str | None = Field(None, description="Custom dictionary file instead of the model's one")
    extra_words: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Words added to the model's dictionary for this request: {\"word\": [\"ph\", ...]}",
    )
    # transcribe items without text (None = fail for such items)
    transcribe: TranscribeOptions | None = Field(default_factory=TranscribeOptions)
    # pause after transcription so that a GUI can show and edit the texts (POST /jobs/{id}/resume)
    review_transcription: bool = False
    skip_unknown_words: bool = Field(False, description="Drop words missing in the dictionary instead of failing")
    refine: RefineOptions | None = Field(None, description="Refine the boundaries before the rules (off by default)")
    postprocess: PostprocessOptions = Field(default_factory=PostprocessOptions)
    output: OutputSpec = Field(default_factory=OutputSpec)


class SegmentRequest(BaseModel):
    input: InputSpec
    model: str = Field(..., description="WFL-ASR model id or path:")
    lang_id: int | None = None
    language: str | None = Field(None, description="Language code; turned into lang_id with the model's langs.txt")
    sample: bool = False
    top_k: int = 0
    top_p: float = 0.0
    temperature: float = 1.0
    confidence_threshold: float | None = None
    # models of WFL-ASR's refactor branch
    decoder: Literal["viterbi", "constrained"] = "viterbi"
    viterbi_bias: float = 5.0
    silence_threshold: float = 0.005
    min_silence_duration: float = 0.5
    refine: RefineOptions | None = Field(None, description="Refine the boundaries before the rules (off by default)")
    postprocess: PostprocessOptions = Field(default_factory=PostprocessOptions)
    output: OutputSpec = Field(default_factory=OutputSpec)


class RefineRequest(RefineOptions):
    """Refines ready labels: each item's segments, or the label file next to its audio (.lab, .TextGrid, .json)."""

    input: InputSpec
    output: OutputSpec = Field(default_factory=OutputSpec)


class MidiRequest(BaseModel):
    input: InputSpec
    model: str = Field("game-large", description="GAME model id or path:")
    language: str | None = None
    tempo: float | Literal["auto"] = 120.0
    seg_threshold: float | None = None
    seg_radius: float | None = None
    t0: float | None = None
    nsteps: int | None = None
    est_threshold: float | None = None
    batch_size: int | None = None
    output_formats: list[str] = Field(default_factory=lambda: ["mid", "csv"], description="mid, txt, csv")
    pitch_format: Literal["name", "number"] = "name"
    round_pitch: bool = False
    output_dir: str | None = None


class SeparateRequest(BaseModel):
    """Vocal separation. Never part of other pipelines: separation can degrade clean recordings."""

    input: InputSpec
    model: str = Field("auto",
                       description="Catalog id or an audio-separator model file name; auto: BS-Roformer with an NVIDIA GPU, "
                                   "the fast MDX-Net model without one")
    stems: list[str] | None = Field(
        None, description="Keep only these stems, e.g. [\"vocals\"]; default: all stems of the model"
    )
    output_format: Literal["wav", "flac", "mp3"] = "wav"
    sample_rate: int = 44100
    output_dir: str | None = Field(None, description="Default: next to the audio (<home>/outputs/<job> for uploads)")
    options: dict[str, Any] = Field(
        default_factory=dict, description="audio-separator parameters: mdx_params, mdxc_params, vr_params, ..."
    )


class PitchRequest(BaseModel):
    input: InputSpec
    model: str = Field("rmvpe", description="rmvpe, fcpe, parselmouth (catalog ids) or path: to an RMVPE model")
    hop: float = Field(0.01, description="Seconds between f0 values")
    f0_min: float = 50.0
    f0_max: float = 1100.0
    threshold: float | None = Field(None, description="Voicing threshold (method specific)")
    output_formats: list[str] = Field(default_factory=list, description="csv (time,f0), json, txt (f0 per line)")
    output_dir: str | None = None
    return_curve: bool = Field(True, description="Include the f0 values in the job result")


class ResynthRequest(BaseModel):
    """A recording (or a part of it) sung again with another f0."""

    input: InputSpec
    f0: list[float] = Field(description="Hz per step of `hop` from the start of the file; 0 = no value")
    hop: float = 0.01
    method: Literal["world", "nsf"] = "world"
    model: str = Field("pc-nsf-hifigan-2025.02", description="Vocoder model (catalog id) for method nsf")
    start: float = 0.0
    end: float | None = None


class TempoRequest(BaseModel):
    input: InputSpec
    model: str = "deeprhythm"


class TextRequest(BaseModel):
    texts: list[str]
    language: str | None = None
    model: str | None = Field(None, description="Aligner model (SOFA / HubertFA / TIFA) whose dictionary / G2P is used")
    g2p: Literal["auto", "dictionary"] = "auto"
    extra_words: dict[str, list[str]] = Field(default_factory=dict, description="Words added to the dictionary")


class ConvertRequest(BaseModel):
    content: str | None = None
    path: str | None = None
    from_format: str | None = None
    to_format: str
    tier: str = "phones"


class FixLabelsRequest(BaseModel):
    """Applies post-processing rules to existing label files (in place, with a backup)."""

    folder: str | None = None
    paths: list[str] = Field(default_factory=list)
    recursive: bool = True
    formats: list[str] = Field(default_factory=lambda: ["htk", "textgrid", "json"],
                               description="Which label files to process (by extension)")
    rule_sets: list[str] = Field(default_factory=list)
    rules: list[dict[str, Any]] = Field(default_factory=list)
    backup: bool = Field(True, description="Copy the original files to <folder>/_backup/<time>/ first")
    dry_run: bool = False


class ItemResult(BaseModel):
    name: str
    audio: str | None = None
    ok: bool = True
    error: str | None = None
    text: str | None = None
    tokens: list[str] | None = None
    unknown_words: list[str] | None = None
    label: Label | None = None
    files: dict[str, str] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)


class ModelDownloadRequest(BaseModel):
    force: bool = False


class ModelImportRequest(BaseModel):
    engine: str
    path: str
    id: str | None = None
    name: str | None = None
    languages: list[str] | None = None
    text_frontend: str | None = None
    copy_files: bool = True


class CatalogAddRequest(BaseModel):
    ref: str = Field(..., description="URL or path of a catalog JSON")
