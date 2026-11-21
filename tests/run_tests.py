"""Minimal test runner (pytest-compatible tests, but no pytest needed).

Collects every `test_*` function from tests/test_*.py, runs it, and prints pass / fail / skip with the time taken.
A test skips itself by raising `Skip` (e.g. when the original project folder for legacy comparisons is missing).

usage: python -X utf8 tests/run_tests.py [name-substring ...]      (run from the repo root, PYTHONPATH=src)
"""
import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from _util import Skip  # noqa: E402


def main():
    filters = sys.argv[1:]
    results = []
    for path in sorted(HERE.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in [n for n in dir(mod) if n.startswith("test_")]:
            full = f"{path.stem}::{name}"
            if filters and not any(f in full for f in filters):
                continue
            t0 = time.time()
            try:
                getattr(mod, name)()
                status, msg = "PASS", ""
            except Skip as e:
                status, msg = "SKIP", str(e)
            except Exception:
                status, msg = "FAIL", traceback.format_exc()
            results.append((status, full))
            print(f"{status}  {full}  ({time.time() - t0:.1f} s) {msg if status != 'FAIL' else ''}", flush=True)
            if status == "FAIL":
                print(msg, flush=True)
    n = {s: sum(r[0] == s for r in results) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\n{n['PASS']} passed, {n['FAIL']} failed, {n['SKIP']} skipped")
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
