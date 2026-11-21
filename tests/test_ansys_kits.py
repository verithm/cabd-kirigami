"""Every historical Ansys kit rebuilt from the repository is identical to the kit that was handed out.

Reference: tests/fixtures/ansys/kit_manifest.json (md5 of every kit file except STEP, kit_config.json without
kit_version, STEP id list, member list of the zip handed out), made from the original dist folders by
tests/fixtures/ansys/make_fixtures.py. STEP files embed a time stamp and are compared by geometry (fingerprints).
"""
import hashlib
import io
import json
import os
import zipfile

import numpy as np

from _ansys_helpers import cad, raw, slow, tmp
from _util import FIXTURES, Skip, legacy_root
from cabd.ansys import kit as K
from cabd.ansys.kits import GROUPS
from cabd.config import ANSYS_KIT, BUILD, SAMPLING

MANIFEST = json.loads((FIXTURES / "ansys" / "kit_manifest.json").read_text(encoding="utf-8"))


def md5(b):
    return hashlib.md5(b).hexdigest()


def compare(folder, name):
    """Differences between a rebuilt kit folder and the manifest entry of the kit handed out."""
    ref = MANIFEST["kits"][name]
    bad = []
    files = {p.relative_to(folder).as_posix(): p for p in folder.rglob("*") if p.is_file() and p.parent.name != "step"}
    for rel, want in ref["files"].items():
        if rel not in files:
            bad.append(f"missing {rel}")
        elif rel == "kit_config.json":
            cfg = json.loads(files[rel].read_text(encoding="ascii"))
            cfg.pop("kit_version")
            if cfg != want:
                bad.append(f"kit_config {cfg} != {want}")
        elif md5(files[rel].read_bytes()) != want:
            bad.append(f"differs {rel}")
    return bad


def _inverse():
    try:
        return legacy_root() / "ml" / "results" / "inverse"
    except Skip:
        return None


def test_collection_kits_identical():
    """base PC1-3, ext1 PC4-6, ext2 PC7-9, retry1: every file except STEP, and the assignment tables."""
    out = tmp("kits")
    names = []
    for group in ("base", "ext1", "ext2", "retry1"):
        for k in K.build_group(group, out, step=False):
            names.append(k.name)
            assert not compare(out / k.name, k.name), (k.name, compare(out / k.name, k.name))
            assert sorted(k.jobs.geom_id) == MANIFEST["kits"][k.name]["steps"], k.name
    for f, want in MANIFEST["assignment"].items():
        assert md5((out / f).read_bytes()) == want, f
    print(f"{len(names)} kits identical: {names}")


def test_verification_kits_identical():
    """Kit 1 (ALL, V1-V3) and kit 2 (P1-P3) from the full-precision inverse results (original folder needed for
    those; the repository keeps only the 9-digit candidate lists), plus params_verify*.csv byte for byte."""
    inv = _inverse()
    if inv is None:
        raise Skip("needs the inverse result csvs of the original project (CABD_LEGACY)")
    out = tmp("vkits")
    build = tmp("vbuild")
    for group in ("verify", "verify_all", "verify2"):
        for k in K.build_group(group, out, step=False, build=build, inverse=inv):
            assert not compare(out / k.name, k.name), (k.name, compare(out / k.name, k.name))
            assert sorted(k.jobs.geom_id) == MANIFEST["kits"][k.name]["steps"], k.name
    for f in ("params_verify.csv", "params_verify2.csv"):
        assert (build / "sampling" / f).read_bytes() == (SAMPLING / f).read_bytes(), f


def test_final_kit_identical():
    """Final kit from the raw results of PC1-3: jobs, landmarks, collect.ps1 (patched PC1 script), overnight files,
    config; its seeded results (12 states recorded as not_converged) equal the files in the final kit's own results
    zip, and its progress.csv is the first part of the final kit's progress.csv."""
    raw()
    out = tmp("final")
    (k,) = K.build_group("final", out, step=False)
    assert not compare(out / k.name, k.name), compare(out / k.name, k.name)
    assert sorted(k.jobs.geom_id) == MANIFEST["kits"][k.name]["steps"]
    from cabd.ansys.results import Results
    final = Results("final", overlay=False).own
    for rel, data in k.extra.items():
        key = rel[len("results/"):]
        if key == "progress.csv":
            assert final.bytes(key).startswith(data)
        else:
            assert final.bytes(key) == data, key
    print(f"final kit: {len(k.jobs)} jobs, {len(k.seeded)} states seeded as not_converged, {len(k.extra)} result files identical")


def test_solver_files_are_those_of_every_kit():
    for name, ref in MANIFEST["kits"].items():
        for f in K.SOLVER_FILES:
            if f in ref["files"]:
                assert md5((ANSYS_KIT / f).read_bytes()) == ref["files"][f], (name, f)


def test_kit_zips_and_step_files():
    """Heavy: every kit with STEP files and zips (STEP sets in build/step, generated with the CAD python if missing):
    zip member lists equal the zips handed out, and the STEP files of the fixture sample match by geometry."""
    slow(); raw()
    inv = _inverse()
    if inv is None:
        raise Skip("needs the inverse result csvs of the original project (CABD_LEGACY)")
    if not all((BUILD / "step" / s).exists() for s in ("base", "ext1", "ext2", "verify", "verify2")):
        cad()
    out, zips = tmp("zkits"), tmp("zips")
    names = []
    for group in GROUPS:
        for k in K.build_group(group, out, zips, step=True, inverse=inv):
            ref = MANIFEST["kits"][k.name]
            assert sorted(p.stem for p in (out / k.name / "step").glob("*.step")) == ref["steps"], k.name
            if "zip" in ref:
                members = zipfile.ZipFile(zips / f"{k.name}.zip").namelist()
                assert sorted(members) == sorted(ref["zip"]), k.name
                if members != ref["zip"]:
                    print(f"{k.name}: same members, different order")
            names.append(k.name)
    fx = json.loads((FIXTURES / "geometry" / "step_fingerprints.json").read_text())
    from cabd.geometry.step import run_cad
    files = [BUILD / "step" / s / f"{g}.step" for s, prints in fx.items() for g in prints]
    new = json.loads(run_cad("fingerprint", *files).stdout.strip().splitlines()[-1])
    worst = max(float(np.abs(np.subtract(new[g][key], old[key])).max())
                for prints in fx.values() for g, old in prints.items() for key in ("volumes", "areas", "bbox"))
    assert worst <= 1e-9 and all(new[g]["faces"] == old["faces"] for p in fx.values() for g, old in p.items())
    print(f"{len(names)} kits with zips and STEP files; STEP sample max |diff| {worst:.2g}")
