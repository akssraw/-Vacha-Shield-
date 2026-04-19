from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torchaudio

from approved_sources import (
    CATEGORY_AI,
    CATEGORY_HUMAN,
    ApprovedSource,
    get_source,
    iter_audio_files,
    list_sources,
    source_root,
    write_registry_index,
)


def copy_audio_files(source_files: list[Path], destination_dir: Path, limit: int = 0) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    selected = source_files[:limit] if limit > 0 else source_files
    for file_path in selected:
        dest_path = destination_dir / file_path.name
        if dest_path.exists():
            continue
        shutil.copy2(file_path, dest_path)
        copied += 1
    return copied


def download_yesno(base_dir: Path, limit: int) -> int:
    dataset_root = base_dir / "data" / "internet_raw" / "yesno"
    torchaudio.datasets.YESNO(root=str(dataset_root), download=True)
    audio_dir = dataset_root / "waves_yesno"
    files = sorted(audio_dir.glob("*.wav"))
    return copy_audio_files(files, source_root(base_dir, "human_yesno"), limit=limit)


def download_librispeech_dev_clean(base_dir: Path, limit: int) -> int:
    dataset_root = base_dir / "data" / "internet_raw" / "librispeech"
    torchaudio.datasets.LIBRISPEECH(root=str(dataset_root), url="dev-clean", download=True)
    audio_dir = dataset_root / "LibriSpeech" / "dev-clean"
    files = sorted(audio_dir.rglob("*.flac"))
    return copy_audio_files(files, source_root(base_dir, "human_librispeech_dev_clean"), limit=limit)


def sync_automatic_source(base_dir: Path, source: ApprovedSource, limit: int) -> int:
    if source.source_id == "human_yesno":
        return download_yesno(base_dir, limit=limit)
    if source.source_id == "human_librispeech_dev_clean":
        return download_librispeech_dev_clean(base_dir, limit=limit)
    raise ValueError(f"Source does not support automatic sync: {source.source_id}")


def register_manual_source(base_dir: Path, source: ApprovedSource, from_dir: Path, limit: int) -> int:
    if not from_dir.exists() or not from_dir.is_dir():
        raise FileNotFoundError(f"Manual source directory not found: {from_dir}")
    audio_files = list(iter_audio_files(from_dir))
    if not audio_files:
        raise RuntimeError(f"No audio files found in {from_dir}")
    copied = copy_audio_files(audio_files, source_root(base_dir, source.source_id), limit=limit)
    metadata_path = source_root(base_dir, source.source_id) / "_import_meta.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_id": source.source_id,
                "category": source.category,
                "imported_from": str(from_dir),
                "imported_at_epoch": int(time.time()),
                "file_count_seen": len(audio_files),
                "file_count_copied": copied,
            },
            handle,
            indent=2,
        )
    return copied


def print_sources() -> None:
    for category in (CATEGORY_HUMAN, CATEGORY_AI):
        print(f"[{category.upper()}]")
        for source in list_sources(category):
            print(
                f"- {source.source_id}: {source.name} | {source.acquisition} | {source.homepage}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync approved human and AI voice datasets into the curated registry."
    )
    parser.add_argument("--list", action="store_true", help="List approved sources")
    parser.add_argument(
        "--sync",
        nargs="*",
        default=[],
        help="Automatically sync supported source ids (example: human_yesno human_librispeech_dev_clean)",
    )
    parser.add_argument("--register", type=str, help="Register a manual source id into the curated registry")
    parser.add_argument("--from-dir", type=str, help="Local directory for a manual source registration")
    parser.add_argument("--limit", type=int, default=0, help="Optional max files per source")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.list:
        print_sources()

    if args.sync:
        for source_id in args.sync:
            source = get_source(source_id)
            if source.acquisition != "automatic":
                raise ValueError(
                    f"{source_id} is not automatic. Use --register {source_id} --from-dir <path> instead."
                )
            copied = sync_automatic_source(base_dir, source, limit=args.limit)
            print(f"[+] Synced {source.source_id}: copied {copied} files")

    if args.register:
        source = get_source(args.register)
        if source.acquisition != "manual":
            raise ValueError(f"{source.source_id} is not a manual source")
        if not args.from_dir:
            raise ValueError("--from-dir is required with --register")
        copied = register_manual_source(base_dir, source, Path(args.from_dir), limit=args.limit)
        print(f"[+] Registered {source.source_id}: copied {copied} files")

    index_path = write_registry_index(base_dir)
    print(f"[+] Registry index updated: {index_path}")


if __name__ == "__main__":
    main()
