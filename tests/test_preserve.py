"""AC-3: untouched files stay byte-identical, edits are minimal, --write is idempotent."""

import os
import shutil
import stat

from transformers_v5_migrate import scan_paths

NEEDS_FIX = (
    "import torch\n"
    "from transformers import AutoModelForCausalLM\n"
    "\n"
    "model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float16)  # note\n"
    "print('done')\n"
)


def _snapshot(directory):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(directory.iterdir())}


def test_untouched_files_byte_identical(tmp_path, fixtures):
    for name in ("no_import.py", "no_false_positives.py"):
        shutil.copy(fixtures / name, tmp_path / name)
    (tmp_path / "crlf_clean.py").write_bytes(b"import transformers\r\nx = 1\r\ny = 2\r\n")
    (tmp_path / "Dockerfile").write_text("ENV TRANSFORMERS_CACHE=/x\n")
    (tmp_path / "requirements.txt").write_text("transformers<5\n")
    before = _snapshot(tmp_path)
    old = 1_600_000_000
    for p in tmp_path.iterdir():
        os.utime(p, (old, old))
    before = _snapshot(tmp_path)

    report = scan_paths([str(tmp_path)], write=True)

    assert report.files_changed == []
    assert _snapshot(tmp_path) == before  # bytes and mtimes


def test_crlf_preserved(tmp_path):
    target = tmp_path / "train.py"
    target.write_bytes(NEEDS_FIX.replace("\n", "\r\n").encode())

    scan_paths([str(tmp_path)], write=True)

    raw = target.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") == 5
    assert raw.replace(b"\r\n", b"\n").decode() == NEEDS_FIX.replace("torch_dtype=", "dtype=")


def test_only_changed_lines_differ(tmp_path):
    target = tmp_path / "train.py"
    target.write_text(NEEDS_FIX)
    scan_paths([str(tmp_path)], write=True)
    old_lines, new_lines = NEEDS_FIX.splitlines(), target.read_text().splitlines()
    assert len(old_lines) == len(new_lines)
    assert [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b] == [3]
    assert new_lines[3].endswith("# note")  # trailing comment kept


def test_write_twice_idempotent(tmp_path, example_project):
    work = tmp_path / "project"
    shutil.copytree(example_project, work)

    first = scan_paths([str(work)], write=True)
    after_first = {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}
    second = scan_paths([str(work)], write=True)
    after_second = {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}

    assert len(first.fixes) > 0
    assert second.fixes == [] and second.files_changed == []
    assert after_first == after_second


def test_dry_run_changes_nothing(tmp_path):
    target = tmp_path / "train.py"
    target.write_text(NEEDS_FIX)
    report = scan_paths([str(tmp_path)])
    assert target.read_text() == NEEDS_FIX
    assert len(report.fixes) == 1 and report.files_changed == [str(target)]
    assert report.diffs[0].after != report.diffs[0].before


def test_utf8_bom_is_kept(tmp_path):
    target = tmp_path / "train.py"
    target.write_bytes(b"\xef\xbb\xbf" + NEEDS_FIX.encode())
    scan_paths([str(tmp_path)], write=True)
    assert target.read_bytes().startswith(b"\xef\xbb\xbf")
    assert b"dtype=torch.float16" in target.read_bytes()


def test_atomic_write_leaves_no_temp_file_and_keeps_mode(tmp_path):
    target = tmp_path / "train.py"
    target.write_text(NEEDS_FIX)
    target.chmod(0o750)
    scan_paths([str(tmp_path)], write=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train.py"]
    assert stat.S_IMODE(target.stat().st_mode) == 0o750


def test_unreadable_encoding_is_reported_and_skipped(tmp_path):
    (tmp_path / "bad.py").write_bytes(b"import transformers\nx = '\xff'\n")
    report = scan_paths([str(tmp_path)], write=True)
    assert [f.rule for f in report.findings] == ["TV5P01"]
    assert (tmp_path / "bad.py").read_bytes() == b"import transformers\nx = '\xff'\n"
