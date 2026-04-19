from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CATEGORY_HUMAN = "human"
CATEGORY_AI = "ai"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a", ".aac")


@dataclass(frozen=True)
class ApprovedSource:
    source_id: str
    category: str
    name: str
    acquisition: str
    homepage: str
    license_name: str
    default_enabled: bool = True
    notes: str = ""


APPROVED_SOURCES: tuple[ApprovedSource, ...] = (
    ApprovedSource(
        source_id="human_yesno",
        category=CATEGORY_HUMAN,
        name="OpenSLR YESNO",
        acquisition="automatic",
        homepage="https://www.openslr.org/1",
        license_name="OpenSLR dataset license",
        notes="Small human speech set, useful for quick smoke tests.",
    ),
    ApprovedSource(
        source_id="human_librispeech_dev_clean",
        category=CATEGORY_HUMAN,
        name="LibriSpeech dev-clean",
        acquisition="automatic",
        homepage="https://www.openslr.org/12",
        license_name="CC BY 4.0",
        notes="Clean human speech source with official transcripts.",
    ),
    ApprovedSource(
        source_id="human_common_voice",
        category=CATEGORY_HUMAN,
        name="Mozilla Common Voice",
        acquisition="manual",
        homepage="https://commonvoice.mozilla.org/en/datasets",
        license_name="CC0 / dataset-specific",
        notes="Register a downloaded subset manually to broaden accents and mics.",
    ),
    ApprovedSource(
        source_id="human_common_voice_hi",
        category=CATEGORY_HUMAN,
        name="Mozilla Common Voice Hindi",
        acquisition="manual",
        homepage="https://commonvoice.mozilla.org/en/datasets",
        license_name="CC0 / dataset-specific",
        notes="Hindi speech subset for native Hindi pronunciation and regional accents.",
    ),
    ApprovedSource(
        source_id="human_ai4bharat_kathbath_hi",
        category=CATEGORY_HUMAN,
        name="AI4Bharat Kathbath Hindi",
        acquisition="manual",
        homepage="https://github.com/AI4Bharat/Kathbath",
        license_name="Dataset-specific research license",
        notes="Large Hindi read-speech corpus. Register downloaded clips manually.",
    ),
    ApprovedSource(
        source_id="human_hinglish_field_recordings",
        category=CATEGORY_HUMAN,
        name="Custom Hinglish Field Recordings",
        acquisition="manual",
        homepage="https://github.com/AI4Bharat/IndicVoices-R",
        license_name="Local/collection-specific",
        notes="Use for Romanized Hindi, code-mixed Hindi-English, and WhatsApp-style field recordings.",
    ),
    ApprovedSource(
        source_id="human_vctk",
        category=CATEGORY_HUMAN,
        name="VCTK Corpus",
        acquisition="manual",
        homepage="https://datashare.ed.ac.uk/handle/10283/3443",
        license_name="Open access research dataset",
        notes="Useful for multi-speaker studio-quality human voice coverage.",
    ),
    ApprovedSource(
        source_id="ai_asvspoof_2021_df",
        category=CATEGORY_AI,
        name="ASVspoof 2021 Deepfake",
        acquisition="manual",
        homepage="https://www.asvspoof.org/index2021.html",
        license_name="Challenge dataset license",
        notes="Official deepfake/spoof benchmark. Download externally, then register locally.",
    ),
    ApprovedSource(
        source_id="ai_wavefake",
        category=CATEGORY_AI,
        name="WaveFake",
        acquisition="manual",
        homepage="https://zenodo.org/records/5642694",
        license_name="CC BY 4.0",
        notes="Research corpus of generated audio samples for deepfake detection.",
    ),
    ApprovedSource(
        source_id="ai_generated_edge_tts",
        category=CATEGORY_AI,
        name="Edge TTS Synthetic Voices",
        acquisition="generated",
        homepage="https://github.com/rany2/edge-tts",
        license_name="Provider terms apply",
        notes="Created during training using many cloud voices and scam-style prompts.",
    ),
    ApprovedSource(
        source_id="ai_generated_premium_tts",
        category=CATEGORY_AI,
        name="Premium TTS Synthetic Voices",
        acquisition="generated",
        homepage="https://api.elevenlabs.io",
        license_name="Provider terms apply",
        notes="Optional premium-provider synthetic samples for harder negatives.",
    ),
)


def list_sources(category: str | None = None) -> list[ApprovedSource]:
    if category is None:
        return list(APPROVED_SOURCES)
    return [source for source in APPROVED_SOURCES if source.category == category]


def get_source(source_id: str) -> ApprovedSource:
    for source in APPROVED_SOURCES:
        if source.source_id == source_id:
            return source
    raise KeyError(f"Unknown approved source: {source_id}")


def registry_root(base_dir: Path) -> Path:
    return base_dir / "data" / "approved_sources"


def category_root(base_dir: Path, category: str) -> Path:
    return registry_root(base_dir) / category


def source_root(base_dir: Path, source_id: str) -> Path:
    source = get_source(source_id)
    return category_root(base_dir, source.category) / source.source_id


def ensure_registry_dirs(base_dir: Path) -> None:
    for source in APPROVED_SOURCES:
        source_root(base_dir, source.source_id).mkdir(parents=True, exist_ok=True)


def iter_audio_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    files: list[Path] = []
    for extension in AUDIO_EXTENSIONS:
        files.extend(directory.rglob(f"*{extension}"))
    return sorted(path for path in files if path.is_file())


def iter_registered_audio_files(
    base_dir: Path,
    category: str,
    source_ids: Iterable[str] | None = None,
) -> list[Path]:
    allowed = set(source_ids or [])
    files: list[Path] = []
    for source in list_sources(category):
        if allowed and source.source_id not in allowed:
            continue
        files.extend(iter_audio_files(source_root(base_dir, source.source_id)))
    return sorted(files)


def build_registry_index(base_dir: Path) -> dict:
    ensure_registry_dirs(base_dir)
    sources_payload = []
    for source in APPROVED_SOURCES:
        files = iter_audio_files(source_root(base_dir, source.source_id))
        sources_payload.append(
            {
                **asdict(source),
                "local_dir": str(source_root(base_dir, source.source_id)),
                "file_count": len(files),
            }
        )
    return {
        "registry_root": str(registry_root(base_dir)),
        "sources": sources_payload,
    }


def write_registry_index(base_dir: Path) -> Path:
    payload = build_registry_index(base_dir)
    out_path = registry_root(base_dir) / "source_index.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out_path
