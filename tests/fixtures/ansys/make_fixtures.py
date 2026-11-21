"""Regenerate the Ansys-kit fixtures from the original project (run once; the tests then need no original folder).

kit_manifest.json, per historical kit:
  files   {relative path: md5} of every file except step/*.step (kit_config.json: its parsed content without kit_version)
  steps   sorted geometry ids of step/*.step
  zip     member names of the kit zip handed out (Projects/<kit>.zip), in archive order (absent for the final kit)
Sources: Projects/ansys_collect/dist/<kit>/ (verify2 P1-P3 rebuilt 2026-11-21, the others as handed out),
the final kit folder (its top-level files only: its results/ folder holds the results, compared with the raw zip).
assignment_*.csv of dist/ are listed under "assignment".
Every file is hashed as published: md5 of cabd.ansys.scrub.scrub(original bytes) (personal data removed; identity
for files without any, e.g. everything except README.md and template.wbpz).

usage: python -X utf8 tests/fixtures/ansys/make_fixtures.py <original project folder> [<final kit folder>]
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from cabd.ansys.scrub import scrub  # noqa: E402


def md5(b):
    return hashlib.md5(b).hexdigest()


def describe(folder, top_only=False):
    files, steps = {}, []
    for f in sorted(folder.rglob("*") if not top_only else folder.iterdir()):
        if not f.is_file():
            continue
        rel = f.relative_to(folder).as_posix()
        if rel.startswith("step/"):
            steps.append(f.stem)
        elif rel == "kit_config.json":
            cfg = json.loads(f.read_text(encoding="ascii"))
            cfg.pop("kit_version")
            files[rel] = cfg
        elif not rel.startswith("collect_result_"):
            files[rel] = md5(scrub(f.read_bytes()))
    if top_only:
        steps = sorted(p.stem for p in (folder / "step").glob("*.step"))
    return dict(files=files, steps=sorted(steps))


def main():
    proj = Path(sys.argv[1])
    dist = proj / "ansys_collect" / "dist"
    out = {"kits": {}, "assignment": {}}
    for k in sorted(p for p in dist.iterdir() if p.is_dir()):
        d = describe(k)
        z = proj / f"{k.name}.zip"
        if z.exists():
            d["zip"] = zipfile.ZipFile(z).namelist()
        out["kits"][k.name] = d
    for f in sorted(dist.glob("assignment_*.csv")):
        out["assignment"][f.name] = md5(f.read_bytes())
    if len(sys.argv) > 2:
        out["kits"]["CABD_final"] = describe(Path(sys.argv[2]), top_only=True)
    (HERE / "kit_manifest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print({k: len(v["files"]) for k, v in out["kits"].items()})


if __name__ == "__main__":
    main()
