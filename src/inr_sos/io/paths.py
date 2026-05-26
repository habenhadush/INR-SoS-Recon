from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = "/mnt/asgard0/data/haben/data"
# Fall back to a local cache when the server data dir is unreachable (local dev).
DATA_CACHE = (Path(DATA_DIR) / ".cache" if Path(DATA_DIR).is_dir()
              else PROJECT_ROOT / ".cache" / "joblib")
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"