
from pathlib import Path
import zipfile
from src.utils import run_cmd, slug_from_url


def has_local_data(data_dir: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() not in {".zip", ".7z"} for p in data_dir.rglob("*")) if data_dir.exists() else False


def download_competition(slug: str, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    
    cp = run_cmd(["kaggle", "competitions", "download", slug, "-p", str(data_dir), "--quiet"])
    if cp.returncode != 0:
        raw = (cp.stderr or cp.stdout).strip()
        if "403" in raw or "Forbidden" in raw:
            raise RuntimeError(
                f"Kaggle denied access to `{slug}` with HTTP 403. "
                "Open the competition page in your browser, sign in with the same Kaggle account, "
                "accept the competition rules/terms, then try Start again. "
                "Also confirm your Kaggle token belongs to that account. "
                f"Raw Kaggle error: {raw}"
            )
        raise RuntimeError(raw)
    for z in data_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(data_dir)



