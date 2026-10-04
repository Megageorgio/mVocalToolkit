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


class InputSpec(BaseModel):
    items: list[InputItem] = Field(default_factory=list)
    folder: str | None = Field(None, description="Process all audio files in this folder")
    recursive: bool = True
    patterns: list[str] = Field(default_factory=lambda: list(AUDIO_PATTERNS))
    # text files next to the audio used as lyrics if present (LabelMakr / SOFA corpus layout)
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
    # write <name>.txt next to the audio (LabelMakr corpus style) or into output.dir
    save_txt: bool = False
    output_dir: str | None = None


class PostprocessOptions(BaseModel):
    rule_sets: list[str] = Field(default_factory=list, description="Built-in rule sets, e.g. en_fixes, cleanup")
    rules: list[dict[str, Any]] = Field(default_factory=list, description="Custom rules, see docs/API.md")


class AlignRequest(BaseModel):
    input: InputSpec
    model: str = Field(..., description="SOFA model id (catalog/installed) or path:/folder/of/model")
    language: str | None = Field(None, description="Text frontend; default: the model's language")
    mode: Literal["force", "match"] = "force"
    g2p: Literal["auto", "dictionary", "none"] = Field(
        "auto", description="auto: dictionary + the model's G2P for unknown words; none: tokens are phonemes"
    )
    ap_detector: Literal["loudness_spectral_centroid", "none"] = "loudness_spectral_centroid"
    # transcribe items without text (None = fail for such items)
    transcribe: TranscribeOptions | None = Field(default_factory=TranscribeOptions)
    # pause after transcription so that a GUI can show and edit the texts (POST /jobs/{id}/resume)
    review_transcription: bool = False
    skip_unknown_words: bool = Field(False, description="Drop words missing in the dictionary instead of failing")
    postprocess: PostprocessOptions = Field(default_factory=PostprocessOptions)
    output: OutputSpec = Field(default_factory=OutputSpec)


class SegmentRequest(BaseModel):
    input: InputSpec
    model: str = Field(..., description="WFL-ASR model id or path:")
    lang_id: int | None = None
    sample: bool = False
    top_k: int = 0
    top_p: float = 0.0
    temperature: float = 1.0
    confidence_threshold: float | None = None
    postprocess: PostprocessOptions = Field(default_factory=PostprocessOptions)
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


class TempoRequest(BaseModel):
    input: InputSpec
    model: str = "deeprhythm"


class TextRequest(BaseModel):
    texts: list[str]
    language: str | None = None
    model: str | None = Field(None, description="SOFA model whose dictionary / G2P is used")
    g2p: Literal["auto", "dictionary"] = "auto"


class ConvertRequest(BaseModel):
    content: str | None = None
    path: str | None = None
    from_format: str | None = None
    to_format: str
    tier: str = "phones"


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
