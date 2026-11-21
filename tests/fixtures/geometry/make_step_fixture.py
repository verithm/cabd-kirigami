"""Regenerate step_fingerprints.json: geometry fingerprints (cabd.geometry.step.summary: solids, per-solid volume, area,
face count, bounding box) of a sample of the original STEP files (Research/sampling/step*/), per STEP set.
STEP files embed a time stamp, so the tests compare these fingerprints instead of bytes.

usage: python -X utf8 tests/fixtures/geometry/make_step_fixture.py <original project folder>   (CABD_CAD_PYTHON set)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from cabd.geometry.step import run_cad  # noqa: E402

SAMPLE = {
    "base": ("step", ["TR0000", "TR0255", "TR0511", "TE0000", "TE0255"]),
    "ext1": ("step_ext1", ["TR0512", "TR0767", "TR1023"]),
    "ext2": ("step_ext2", ["TR1024", "TR1270", "TR1536"]),
    "verify": ("step_verify", ["VG0018", "VR0090", "VC0254"]),
    "verify2": ("step_verify2", ["WG0018", "XR0090", "XC0254"]),
}


def main():
    sampling = Path(sys.argv[1]) / "Research" / "sampling"
    out = {}
    for name, (folder, ids) in SAMPLE.items():
        files = [sampling / folder / f"{g}.step" for g in ids if (sampling / folder / f"{g}.step").exists()]
        out[name] = json.loads(run_cad("fingerprint", *files).stdout.strip().splitlines()[-1])
    (HERE / "step_fingerprints.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print({k: list(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
