"""Safe ingestion of uploaded files, directory selections, and ZIP archives."""

from __future__ import annotations

import hashlib
import io
from pathlib import PurePosixPath
import zipfile

from .qc_models import QCFile


MAX_ARCHIVE_FILES = 2_000
MAX_EXPANDED_BYTES = 500 * 1024 * 1024
MAX_FILE_BYTES = 100 * 1024 * 1024


def _safe_path(name: str) -> str | None:
    normalized = str(name).replace("\\", "/").lstrip("/")
    path = PurePosixPath(normalized)
    if not normalized or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return str(path)


def _qc_file(path: str, content: bytes) -> QCFile:
    return QCFile(
        path=path,
        content=content,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def expand_uploaded_files(uploaded_files) -> tuple[list[QCFile], list[str]]:
    """Return normalized files and user-facing ingestion warnings."""
    files: list[QCFile] = []
    warnings: list[str] = []
    total = 0

    for uploaded in uploaded_files or []:
        name = _safe_path(getattr(uploaded, "name", ""))
        if name is None:
            warnings.append("Ignored an upload with an unsafe or empty path.")
            continue
        content = uploaded.getvalue()
        if len(content) > MAX_FILE_BYTES:
            warnings.append(f"Ignored {name}: file exceeds the 100 MB analysis limit.")
            continue

        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    members = [m for m in archive.infolist() if not m.is_dir()]
                    if len(members) > MAX_ARCHIVE_FILES:
                        warnings.append(
                            f"Ignored {name}: archive contains more than {MAX_ARCHIVE_FILES} files."
                        )
                        continue
                    for member in members:
                        member_path = _safe_path(member.filename)
                        if member_path is None:
                            warnings.append(f"Ignored an unsafe path inside {name}.")
                            continue
                        if member.file_size > MAX_FILE_BYTES:
                            warnings.append(f"Ignored {member_path}: expanded file exceeds 100 MB.")
                            continue
                        total += member.file_size
                        if total > MAX_EXPANDED_BYTES:
                            warnings.append("Stopped archive expansion at the 500 MB session limit.")
                            return files, warnings
                        files.append(_qc_file(f"{name}/{member_path}", archive.read(member)))
            except (zipfile.BadZipFile, OSError) as exc:
                warnings.append(f"Could not read {name} as a ZIP archive: {exc}")
            continue

        total += len(content)
        if total > MAX_EXPANDED_BYTES:
            warnings.append("Stopped collecting uploads at the 500 MB session limit.")
            break
        files.append(_qc_file(name, content))

    # Folder upload and additive file selection may contain exact duplicates.
    unique: dict[tuple[str, str], QCFile] = {}
    for file in files:
        unique[(file.path, file.sha256)] = file
    return list(unique.values()), warnings
