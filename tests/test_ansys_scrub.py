"""Published files are the original files with personal data removed by cabd.ansys.scrub (and nothing else changed).

Fast: the rules on synthetic text, zip rewriting, and that no published file would still be changed by the rules.
Legacy (CABD_LEGACY): the Workbench template, the kit guide and the verification csv equal scrub(original).
Heavy (CABD_SLOW=1): every raw result zip in ANSYS_RAW equals scrub(original zip) byte for byte, and every data file in
it (curves, landmark tables, progress.csv) equals the original's. Originals are looked up in the original project
folder and, for zips it does not keep (PC1 and the final kit ran outside it), in CABD_RAW_ORIGINAL (a folder of the
unscrubbed zips) if set.
"""
import io
import os
import re
import zipfile
from pathlib import Path

from _ansys_helpers import raw, slow
from _util import legacy_root
from cabd.ansys import scrub as S
from cabd.config import ANSYS_KIT, RESULTS, ROOT

DATA_MEMBER = re.compile(r"(_curve\.csv|_landmark_disp\.csv|_landmark_nodes\.csv|(^|/)progress\.csv)$")


def _zip(members):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members:
            z.writestr(name, data)
    return b.getvalue()


def test_scrub_rules():
    text = (b"geom_id=TR0001\r\nstate=partial\r\npc=2\r\ncomputer=SOME-PC1\r\nstatus=ok\r\n"
            b"step=C:\\Users\\Some Body\\Documents\\kit\\step\\TR0001.step\r\n"
            b"r\"C:\\\\Users\\\\Some Body\\\\Documents\\\\x.stp\"\r\n"
            b"/* author */ ('Some Body'),\r\n"
            b"  Core                Machine Name   Working Directory\r\n -----------------------------\r\n"
            b"     0                    SOME-PC1   D:\\kit\\work\r\n     1                    SOME-PC1   D:\\kit\\work\r\n\r\n"
            b"source,collect_result_PCALL_SOME_PC-1_20261118_1654.zip,1.5\r\n")
    hosts = set()
    out = S.scrub_bytes(text, hosts)
    assert hosts == {b"SOME-PC1"}
    assert b"Some Body" not in out and b"SOME" not in out, out
    assert b"computer=HOST\r\n" in out and b"step=C:\\Users\\user\\Documents\\kit\\step\\TR0001.step" in out
    assert b"C:\\\\Users\\\\user\\\\Documents" in out and b"/* author */ ('')," in out
    assert b"     0                        HOST   D:\\kit\\work\r\n" in out          # column stays right-aligned
    assert b"collect_result_PCALL_20261118_1654.zip,1.5" in out
    assert S.scrub_bytes(out) == out                                               # idempotent
    kr = "  - 살짝 젖음: 홍길동 님 프로젝트의 `bi_370_13_3_12` (SYS 5)\n".encode("utf-8")
    assert S.scrub_bytes(kr).decode("utf-8") == "  - 살짝 젖음: 팀 공용 프로젝트의 `bi_370_13_3_12` (SYS 5)\n"
    # binary member: same length, UTF-8 and UTF-16-LE
    b16 = b"\x00\x01" + "C:\\Users\\Some Body\\AppData".encode("utf-16-le") + b"\x00" + b"C:/Users/Some Body/x"
    o16 = S.scrub_bytes(b16)
    assert len(o16) == len(b16) and "C:\\Users\\user_____\\AppData".encode("utf-16-le") in o16 and b"C:/Users/user_____/x" in o16
    # names
    assert S.scrub_name("collect_result_PCretry1_SOME-PC_20261117_2108.zip") == "collect_result_PCretry1_20261117_2108.zip"
    assert S.scrub_name("collect_result_PC1_20261108_1719.zip") == "collect_result_PC1_20261108_1719.zip"
    assert S.scrub_name("machine_SOME-PC.txt") == "machine_HOST.txt"
    # numbers and plain data untouched
    data = b"set,time,u_applied_mm,fx_disp_N,U_total_mJ\r\n1,0.05,0.75,0.0123,0.004\r\n"
    assert S.scrub_bytes(data) == data


def test_zip_rewrite_keeps_unchanged_members():
    z = _zip([("a/curve.csv", b"1,2,3\n" * 100), ("machine_SOME-PC.txt", b"computer    : SOME-PC\r\n"),
              ("a/status.txt", b"pc=1\r\ncomputer=SOME-PC\r\n")])
    out = S.scrub_zip(z)
    zi, zo = zipfile.ZipFile(io.BytesIO(z)), zipfile.ZipFile(io.BytesIO(out))
    assert zo.namelist() == ["a/curve.csv", "machine_HOST.txt", "a/status.txt"] and zo.testzip() is None
    assert zo.read("a/curve.csv") == zi.read("a/curve.csv") and zo.read("a/status.txt") == b"pc=1\r\ncomputer=HOST\r\n"
    second = zi.infolist()[1].header_offset
    assert out[:second] == z[:second]                                  # first member copied raw (header + data)
    assert S.scrub_zip(out) == out
    assert S.scrub_zip(z[:]) == out                                    # deterministic


def test_published_files_need_no_scrubbing():
    """Every committed text / archive file (with CABD_SLOW=1 also the Release assets in ARTIFACTS, if present) is a
    fixed point of the rules."""
    roots = [ROOT / d for d in ("ansys_kit", "data", "docs", "results", "tests/fixtures")] + [ROOT / "README.md"]
    from cabd.config import ARTIFACTS
    if ARTIFACTS.exists() and os.environ.get("CABD_SLOW") == "1":
        roots.append(ARTIFACTS)
    files = [f for r in roots for f in ([r] if r.is_file() else r.rglob("*")) if f.is_file()]
    bad = [str(f) for f in files if S.scrub_name(f.name) != f.name or S.scrub(f.read_bytes()) != f.read_bytes()]
    assert not bad, bad
    print(f"{len(files)} files checked")


def test_kit_files_are_scrubbed_originals():
    """ansys_kit/template.wbpz and README.md = scrub(original); the other solver files are the originals byte for byte;
    the verification csvs = scrub(original) (only the source zip names differ)."""
    leg = legacy_root()
    src = leg / "ansys_collect" / "src"
    changed = []
    for f in sorted(p for p in src.iterdir() if p.is_file()):
        orig, repo = f.read_bytes(), (ANSYS_KIT / f.name).read_bytes()
        assert S.scrub(orig) == repo, f.name
        if orig != repo:
            changed.append(f.name)
    assert changed == ["README.md", "template.wbpz"], changed
    inv = leg / "ml" / "results" / "inverse"
    for f in ("ansys_verify_scored.csv", "ansys_verify_summary.csv"):
        assert S.scrub((inv / f).read_bytes()) == (RESULTS / "verify" / f).read_bytes(), f
    # the template keeps every member, its order, methods and time stamps; only paths / STEP authors change
    a, b = (zipfile.ZipFile(io.BytesIO(x)) for x in ((src / "template.wbpz").read_bytes(), (ANSYS_KIT / "template.wbpz").read_bytes()))
    assert [(i.filename, i.compress_type, i.date_time, i.external_attr) for i in a.infolist()] == \
           [(i.filename, i.compress_type, i.date_time, i.external_attr) for i in b.infolist()]
    diff = [i.filename for i in a.infolist() if a.read(i) != b.read(i.filename)]
    assert all(n.endswith((".wbdp", ".mechdb", ".stp")) for n in diff), diff
    assert all(len(a.read(n)) == len(b.read(n)) for n in diff if n.endswith(".mechdb"))
    print(f"changed: {changed}; template members changed: {len(diff)}")


def _originals(leg):
    dirs = [leg] + ([Path(os.environ["CABD_RAW_ORIGINAL"])] if os.environ.get("CABD_RAW_ORIGINAL") else [])
    return {S.scrub_name(p.name): p for d in dirs for p in d.glob("collect_result_*.zip")}


def test_raw_zips_are_scrubbed_originals():
    slow()
    found = raw()
    originals = _originals(legacy_root())
    checked, missing = [], []
    for label, path in sorted(found.items()):
        orig = originals.get(path.name)
        if orig is None:
            missing.append(path.name)
            continue
        ob, pb = orig.read_bytes(), path.read_bytes()
        assert S.scrub_zip(ob) == pb, path.name
        zo, zp = zipfile.ZipFile(io.BytesIO(ob)), zipfile.ZipFile(io.BytesIO(pb))
        n_data = 0
        for i, j in zip(zo.infolist(), zp.infolist()):
            if DATA_MEMBER.search(i.filename):
                assert i.filename == j.filename and zo.read(i) == zp.read(j), (path.name, i.filename)
                n_data += 1
        checked.append(f"{path.name} ({n_data} data files identical)")
    print("checked:", checked, "| no original found for:", missing)
    assert checked
    if missing and not os.environ.get("CABD_RAW_ORIGINAL"):
        print("set CABD_RAW_ORIGINAL to the folder of the unscrubbed zips to check these too")
    elif missing:
        raise AssertionError(f"originals missing: {missing}")

