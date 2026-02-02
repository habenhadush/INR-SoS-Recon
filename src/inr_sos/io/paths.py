from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
print(PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DATA_CACHE = DATA_DIR / ".cache"
