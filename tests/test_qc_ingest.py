"""Tests for raw calculation upload ingestion."""

import io
import zipfile

from iqc_dashboard.qc_ingest import expand_uploaded_files


class Upload:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


def test_directory_paths_and_duplicate_basenames_are_preserved():
    files, warnings = expand_uploaded_files(
        [Upload("run-a/job.out", b"a"), Upload("run-b/job.out", b"b")]
    )
    assert [file.path for file in files] == ["run-a/job.out", "run-b/job.out"]
    assert warnings == []


def test_zip_expansion_rejects_path_traversal_and_keeps_safe_files():
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("calc/job.out", "output")
        archive.writestr("../escape.out", "unsafe")

    files, warnings = expand_uploaded_files([Upload("calculations.zip", payload.getvalue())])

    assert [file.path for file in files] == ["calculations.zip/calc/job.out"]
    assert any("unsafe path" in warning for warning in warnings)


def test_exact_duplicate_upload_is_deduplicated():
    upload = Upload("job.out", b"same")
    files, _ = expand_uploaded_files([upload, upload])
    assert len(files) == 1
