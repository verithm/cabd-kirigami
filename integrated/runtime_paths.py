"""Repository-relative code and configurable, untracked experiment storage."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get('CABD_INTEGRATED_RUNS', ROOT / 'artifacts' / 'integrated')).expanduser().resolve()
os.environ.setdefault('MPLCONFIGDIR', str(RUNS / '.matplotlib'))

def reference(path):
    """Persist traces relative to RUNS, including when RUNS is outside the checkout."""
    return Path(path).resolve().relative_to(RUNS).as_posix()

def resolve_reference(value):
    """Read current references and historical workspace output/... references."""
    path = Path(str(value).replace(chr(92), "/"))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'output':
        path = Path(*path.parts[1:])
    return RUNS / path
