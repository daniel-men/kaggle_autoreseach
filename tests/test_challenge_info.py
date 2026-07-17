import subprocess
import zipfile

import pytest

from src import challenge_info


def test_has_local_data_detects_non_archive_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "competition.zip").write_text("archive", encoding="utf-8")
    (data_dir / "nested").mkdir()
    (data_dir / "nested" / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    assert challenge_info.has_local_data(data_dir) is True


def test_has_local_data_ignores_missing_and_archive_only_dirs(tmp_path):
    assert challenge_info.has_local_data(tmp_path / "missing") is False

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "competition.zip").write_text("archive", encoding="utf-8")
    (data_dir / "competition.7z").write_text("archive", encoding="utf-8")

    assert challenge_info.has_local_data(data_dir) is False


def test_download_competition_extracts_downloaded_zip(tmp_path, monkeypatch):
    def fake_run_cmd(cmd):
        data_dir = tmp_path / "data"
        with zipfile.ZipFile(data_dir / "payload.zip", "w") as zf:
            zf.writestr("train.csv", "x,y\n1,2\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(challenge_info, "run_cmd", fake_run_cmd)

    challenge_info.download_competition("demo", tmp_path / "data")

    assert (tmp_path / "data" / "train.csv").read_text(encoding="utf-8") == "x,y\n1,2\n"


def test_download_competition_explains_kaggle_403(tmp_path, monkeypatch):
    def fake_run_cmd(cmd):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="403 Forbidden")

    monkeypatch.setattr(challenge_info, "run_cmd", fake_run_cmd)

    with pytest.raises(RuntimeError, match="accept the competition rules"):
        challenge_info.download_competition("private-demo", tmp_path / "data")
