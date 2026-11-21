"""Remove personal data (user names, computer names, user-profile paths) from files that are published.

The repository and its Release assets are public, but the original files carry personal data in places no reader of
the pipeline looks at: Windows user-profile paths and STEP author fields inside the Workbench template, computer
names in the raw Ansys result zips (file name, machine_<computer>.txt, `computer=` lines of the status files, the
"Machine Name" table of the MAPDL .out logs, profile paths in the logs), and the zip file names copied into the
verification csv. Every published copy of such a file is `scrub(original)`; the legacy-equivalence tests compare
scrub(original) with the repository / Release copy, so the equivalence with the original project stays checkable.

The rules are patterns (no name appears in this file), applied to bytes:
  profile   <drive>:\\Users\\<name>\\ (any separator, single or doubled) -> <drive>:\\Users\\user\\
            In binary members (any NUL byte, e.g. the Mechanical database .mechdb) the name is replaced in place by
            "user" padded with "_" to the same length, in UTF-8 and in UTF-16-LE, so no offset or length field moves.
  author    STEP header author list (`/* author */` followed by quoted names) -> empty name ('')
  computer  `computer=<x>` / `computer    : <x>` lines -> HOST
  machine   rows of the MAPDL "Core  Machine Name  Working Directory" table: the machine name -> HOST (the column
            stays right-aligned)
  honorific Korean "<given name> 님 프로젝트" -> "팀 공용 프로젝트" (the template's origin in the kit guide)
  zip name  collect_result_PC<label>_<computer>_<yyyymmdd>_<hhmm>.zip -> collect_result_PC<label>_<yyyymmdd>_<hhmm>.zip
            (in text and as file name); zip member machine_<computer>.txt -> machine_HOST.txt
Every number and every file the pipeline reads for data (curves, landmark tables, progress.csv, the status / mesh
fields of status and log files) is unchanged. Zip archives are rewritten member by member in the original order:
an unchanged member is copied byte for byte (local header and compressed data), a changed one is recompressed with
the same method (sizes in its local header) and keeps its time stamp, attributes and extra fields; the archive
comment is kept.
After scrubbing a result zip, every computer name found by the rules is searched in all members again (any case);
a remaining occurrence is an error.

Assumption (not testable here, Ansys cannot be run): Workbench archives are relocatable and the stored absolute paths
in the template (designPoint.wbdp, *.mechdb) are refreshed when the archive is opened, so the scrubbed template behaves
like the original.

usage: python -m cabd.ansys.scrub zips <in_dir> <out_dir>     scrubbed copies of every collect_result zip (new names)
       python -m cabd.ansys.scrub file <in> <out>             one file (zip-like by content, else text/binary rules)
       python -m cabd.ansys.scrub check <path> ...            exit 1 if any rule would still change a file
"""
import argparse
import re
import struct
import sys
import zlib
from pathlib import Path

HOST = b"HOST"
USER = b"user"

_SEP = rb"(?:\\\\|\\|//|/)"
PROFILE = re.compile(rb"(?i)([A-Z]:" + _SEP + rb"Users" + _SEP + rb")([^\\/\r\n\"'<>|:*?\x00]{1,64}?)(?=[\\/])")
PROFILE16 = re.compile(rb"(?i)([A-Z]\x00:\x00(?:[\\/]\x00){1,2}U\x00s\x00e\x00r\x00s\x00(?:[\\/]\x00){1,2})"
                       rb"((?:(?![\\/]\x00)[\x20-\xff][\x00-\xff]){1,64}?)(?=[\\/]\x00)")
AUTHOR = re.compile(rb"(/\*\s*author\s*\*/\s*\()\s*'[^'\r\n]*'(?:\s*,\s*'[^'\r\n]*')*\s*(\))")
COMPUTER_LINE = re.compile(rb"(?im)^([ \t]*computer[ \t]*[=:][ \t]*)([^\r\n]*?)([ \t]*)(?=\r?$)")
MACHINE_TABLE = re.compile(rb"(Core[ \t]+Machine Name[ \t]+Working Directory[ \t]*\r?\n[ \t]*-+[ \t]*\r?\n)"
                           rb"((?:[ \t]+\d+[ \t]+\S+[ \t]+[^\r\n]*\r?\n)+)")
MACHINE_ROW = re.compile(rb"(?m)^([ \t]+\d+)([ \t]+)(\S+)(?=[ \t])")
_HANGUL = rb"(?:[\xea-\xed][\x80-\xbf][\x80-\xbf])"   # one Hangul syllable in UTF-8 (U+AC00..U+D7A3, loosely)
HONORIFIC = re.compile(_HANGUL + rb"{2,4} " + "님 프로젝트".encode("utf-8"))
HONORIFIC_TO = "팀 공용 프로젝트".encode("utf-8")
ZIP_NAME = re.compile(r"collect_result_PC([^_\s\\/]+)_(\S+?)_(\d{8})_(\d{4})\.zip")
ZIP_NAME_B = re.compile(ZIP_NAME.pattern.encode("ascii"))
MACHINE_MEMBER = re.compile(r"^((?:.*/)?machine_)(.+)(\.txt)$")


def _pad(name_len, unit=1):
    """'user' padded with '_' (or cut) to `name_len` characters."""
    s = (USER.decode() + "_" * name_len)[:name_len]
    return s.encode("utf-16-le") if unit == 2 else s.encode("ascii")


def _machine_rows(m, hosts):
    def row(r):
        if r.group(3) != HOST:
            hosts.add(r.group(3))
        ws = r.group(2)
        extra = len(r.group(3)) - len(HOST)          # keep the right edge of the name column where it was
        ws = ws + b" " * extra if extra > 0 else ws[:max(1, len(ws) + extra)]
        return r.group(1) + ws + HOST
    return m.group(1) + MACHINE_ROW.sub(row, m.group(2))


def scrub_bytes(data, hosts=None):
    """Scrubbed copy of one file's bytes (rules in the module docstring). `hosts` (a set) collects the computer names
    removed by the computer / machine rules."""
    hosts = set() if hosts is None else hosts
    if b"\x00" in data:   # binary: same-length replacements only
        data = PROFILE.sub(lambda m: m.group(1) + _pad(len(m.group(2))), data)
        return PROFILE16.sub(lambda m: m.group(1) + _pad(len(m.group(2)) // 2, 2), data)
    data = PROFILE.sub(lambda m: m.group(1) + USER, data)
    data = AUTHOR.sub(rb"\1''\2", data)

    def comp(m):
        if m.group(2) and m.group(2) != HOST:
            hosts.add(m.group(2))
        return m.group(1) + (HOST if m.group(2) else b"") + m.group(3)
    data = COMPUTER_LINE.sub(comp, data)
    data = MACHINE_TABLE.sub(lambda m: _machine_rows(m, hosts), data)
    data = HONORIFIC.sub(HONORIFIC_TO, data)
    return ZIP_NAME_B.sub(rb"collect_result_PC\1_\3_\4.zip", data)


def scrub_name(name):
    """Zip file name or zip member name without the computer name."""
    name = ZIP_NAME.sub(r"collect_result_PC\1_\3_\4.zip", name)
    return MACHINE_MEMBER.sub(lambda m: m.group(1) + HOST.decode() + m.group(3), name)


# ----------------------------------------------------------------------------------------------- zip rewriting
_LOCAL, _CENTRAL, _END, _DESCRIPTOR = 0x04034B50, 0x02014B50, 0x06054B50, 0x08074B50


def _entries(buf):
    """Central-directory entries of a (non-zip64) archive: list of dicts with the raw fields, and the comment."""
    end = buf.rfind(struct.pack("<I", _END))
    if end < 0:
        raise ValueError("not a zip archive")
    _, _, _, _, count, cd_size, cd_off, clen = struct.unpack("<IHHHHIIH", buf[end:end + 22])
    if count == 0xFFFF or cd_off == 0xFFFFFFFF:
        raise ValueError("zip64 archives are not supported")
    out, p = [], cd_off
    for _ in range(count):
        f = list(struct.unpack("<IHHHHHHIIIHHHHHII", buf[p:p + 46]))
        if f[0] != _CENTRAL:
            raise ValueError("bad central directory")
        nlen, xlen, cmlen = f[10], f[11], f[12]
        name = buf[p + 46:p + 46 + nlen]
        extra = buf[p + 46 + nlen:p + 46 + nlen + xlen]
        comment = buf[p + 46 + nlen + xlen:p + 46 + nlen + xlen + cmlen]
        out.append(dict(fields=f, name=name, extra=extra, comment=comment))
        p += 46 + nlen + xlen + cmlen
    return out, buf[end + 22:end + 22 + clen]


def rewrite_zip(buf, transform):
    """New archive bytes. transform(name: str, data: bytes) -> (new_name, new_data); a member whose name and data are
    unchanged is copied raw. Order, methods, time stamps, attributes, extra fields and comments are kept."""
    entries, comment = _entries(buf)
    out, central, changed = bytearray(), bytearray(), False
    for e in entries:
        f = e["fields"]
        flags, method, crc, csize, usize, off = f[3], f[4], f[7], f[8], f[9], f[16]
        utf8 = bool(flags & 0x800)
        lf = list(struct.unpack("<IHHHHHIIIHH", buf[off:off + 30]))
        if lf[0] != _LOCAL:
            raise ValueError("bad local header")
        lname = buf[off + 30:off + 30 + lf[9]]
        lextra = buf[off + 30 + lf[9]:off + 30 + lf[9] + lf[10]]
        start = off + 30 + lf[9] + lf[10]
        comp = buf[start:start + csize]
        end_raw = start + csize
        if flags & 0x8:   # data descriptor after the data (optional signature, crc, sizes); copied with the member
            end_raw += 16 if buf[end_raw:end_raw + 4] == struct.pack("<I", _DESCRIPTOR) else 12
        name = e["name"].decode("utf-8" if utf8 else "cp437")
        if method == 0:
            data = bytes(comp)
        elif method == 8:
            data = zlib.decompress(comp, -15)
        else:
            raise ValueError(f"compression method {method} not supported")
        new_name, new_data = transform(name, data)
        new_off = len(out)
        if new_name == name and new_data == data:
            out += buf[off:end_raw]
            nb = e["name"]
        else:
            changed = True
            nb = new_name.encode("utf-8" if utf8 or not new_name.isascii() else "cp437")
            if not new_name.isascii():
                flags |= 0x800
            flags &= ~0x8         # sizes go into the new local header
            if method == 8:
                c = zlib.compressobj(9, zlib.DEFLATED, -15)
                comp = c.compress(new_data) + c.flush()
            else:
                comp = new_data
            crc, csize, usize = zlib.crc32(new_data) & 0xFFFFFFFF, len(comp), len(new_data)
            lf[3], lf[6], lf[7], lf[8], lf[9] = flags, crc, csize, usize, len(nb)
            out += struct.pack("<IHHHHHIIIHH", *lf) + nb + lextra + comp
        f[3], f[7], f[8], f[9], f[10], f[16] = flags, crc, csize, usize, len(nb), new_off
        central += struct.pack("<IHHHHHHIIIHHHHHII", *f) + nb + e["extra"] + e["comment"]
    if not changed:   # nothing to scrub: the archive as it was (also keeps zip64 end records, e.g. torch files)
        return bytes(buf)
    cd_off = len(out)
    out += central
    out += struct.pack("<IHHHHIIH", _END, 0, 0, len(entries), len(entries), len(central), cd_off, len(comment)) + comment
    return bytes(out)


def scrub_zip(buf, hosts=None):
    """Scrubbed archive bytes (member names and contents); raises if a found computer name is still in any member."""
    hosts = set() if hosts is None else hosts
    for n in _names(buf):
        m = MACHINE_MEMBER.match(n)
        if m and m.group(2) != HOST.decode():
            hosts.add(m.group(2).encode())
    done = {}

    def t(name, data):
        new = scrub_bytes(data, hosts)
        done[name] = new
        return scrub_name(name), new
    out = rewrite_zip(buf, t)
    left = [(n, h) for h in hosts for n, d in done.items() if re.search(re.escape(h), d, re.I)]
    if left:
        raise ValueError(f"computer name still present in {len(left)} member(s), e.g. {left[0][0]}")
    return out


def _names(buf):
    entries, _ = _entries(buf)
    return [e["name"].decode("utf-8" if e["fields"][3] & 0x800 else "cp437") for e in entries]


def is_zip(buf):
    return buf[:4] == struct.pack("<I", _LOCAL)


def scrub(buf):
    """Scrubbed copy of a file's bytes: archives (zip, wbpz) member by member, other files with the byte rules."""
    return scrub_zip(buf) if is_zip(buf) else scrub_bytes(buf)


def scrub_dir(src, dst):
    """Scrubbed copies of every collect_result zip in `src` under their new names in `dst`; returns [(old, new)]."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    done = []
    for p in sorted(src.glob("collect_result_*.zip")):
        new = dst / scrub_name(p.name)
        new.write_bytes(scrub_zip(p.read_bytes()))
        done.append((p.name, new.name))
        print(f"{p.name} -> {new.name} ({new.stat().st_size} bytes)", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("zips")
    z.add_argument("src", type=Path)
    z.add_argument("dst", type=Path)
    f = sub.add_parser("file")
    f.add_argument("src", type=Path)
    f.add_argument("dst", type=Path)
    c = sub.add_parser("check")
    c.add_argument("paths", type=Path, nargs="+")
    a = ap.parse_args()
    if a.cmd == "zips":
        scrub_dir(a.src, a.dst)
    elif a.cmd == "file":
        a.dst.write_bytes(scrub(a.src.read_bytes()))
    else:
        files = [f for p in a.paths for f in ([p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file()))]
        bad = [p for p in files if scrub(p.read_bytes()) != p.read_bytes() or scrub_name(p.name) != p.name]
        print(f"{len(files)} files checked, {len(bad)} not scrubbed")
        for p in bad:
            print(f"not scrubbed: {p}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
