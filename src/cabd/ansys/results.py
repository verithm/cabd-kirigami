"""Reader of the raw Ansys results (collect_result zips) and the scalars of one solved state.

Every kit zips its results folder as collect_result_PC<label>_<computer>_<yyyymmdd>_<hhmm>.zip (make_result_zip.ps1).
These zips are the raw data of the project: they live in ANSYS_RAW (default artifacts/ansys_raw, a GitHub Release
asset; override with CABD_ANSYS_RAW) and are only read. The published copies are scrubbed of personal data
(cabd.ansys.scrub: no computer name in the file name, collect_result_PC<label>_<yyyymmdd>_<hhmm>.zip; computer names and
user-profile paths in the logs replaced; every data file unchanged); both name forms are read. For each label the
newest zip (by the time stamp in its name) is used. Labels: 1-3 (base collection), final (the single-PC kit that finished what PC1-3 left), 4-9 (extra
training batches), retry1, ALL / V1-V3 (verification kit 1), P1-Pn (verification kit 2).

The final kit re-solved states of PC1-3 geometries; its files take priority over the PC's own (same rule as the
original reader, which applied the final kit folder to every label; geometry ids of the other kits never occur in it).

Scalars of a state (`scalars`), shared by the datasets, the geometry table and the verification scoring:
  of the final and the first-attempt (autofailed) curve, the one that reached further is used;
  u1 = first downward zero crossing of the force, dE_f = U(u1); F_min = lowest force valley (the origin and a curve
  with no dip do not count); u2 = where the force returns through zero, dE_r = U(u1) - U(u2);
  class: bi (below zero and back), mono (never below zero and the curve reaches >= 12 mm), neg_end (still below zero
  at the end), unknown (never below zero, stopped before 12 mm);
  theta2 = rotation of the upper arm's A->B probe vector from its undeformed direction at u2 (landmark displacements
  interpolated linearly between the two substeps around u2).
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from cabd.config import ANSYS_RAW

FINAL_LABEL = "final"
# label, computer (absent in the scrubbed copies), date, time
NAME = re.compile(r"collect_result_PC(.+?)(?:_(.+))?_(\d{8})_(\d{4})\.zip$")


def zips(raw=None):
    """{label: newest zip path} of every collect_result zip in `raw`."""
    raw = Path(raw or ANSYS_RAW)
    found = {}
    for p in raw.glob("collect_result_PC*_*.zip"):
        m = NAME.match(p.name)
        if m and (m.group(1) not in found or found[m.group(1)][0] < m.group(3) + m.group(4)):
            found[m.group(1)] = (m.group(3) + m.group(4), p)
    return {k: v[1] for k, v in found.items()}


class Archive:
    """Read-only view of one result zip (member names with / separators)."""

    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path)
        self.names = {n.replace("\\", "/"): n for n in self.zip.namelist()}
        self.name = self.path.name

    def has(self, key):
        return key in self.names

    def bytes(self, key):
        return self.zip.read(self.names[key]) if key in self.names else None

    def listdir(self, gid, prefix):
        head = f"{gid}/{prefix}"
        return [n[len(gid) + 1:] for n in self.names if n.startswith(head)]


class MemoryArchive:
    """Archive-like view of in-memory result files {key: bytes} (e.g. the final kit's results right after it was built)."""

    def __init__(self, files, name):
        self.files, self.name = files, name

    def has(self, key):
        return key in self.files

    def bytes(self, key):
        return self.files.get(key)

    def listdir(self, gid, prefix):
        head = f"{gid}/{prefix}"
        return [n[len(gid) + 1:] for n in self.files if n.startswith(head)]


class Results:
    """Newest results of one kit label, with the final kit's files taking priority (see module docstring)."""

    def __init__(self, label, raw=None, overlay=True):
        label = str(label)
        found = zips(raw)
        if label not in found:
            raise SystemExit(f"no results found for PC{label} in {raw or ANSYS_RAW}")
        self.label, self.own = label, Archive(found[label])
        self.name = self.own.name
        if overlay is True:
            self.final = Archive(found[FINAL_LABEL]) if label != FINAL_LABEL and FINAL_LABEL in found else None
        else:   # False / None: no overlay; or an archive-like object standing in for the final kit's results
            self.final = overlay or None

    def raw(self, key):
        """Bytes of a result file (final kit first for per-geometry files), or None."""
        if self.final is not None and "/" in key and self.final.has(key):
            return self.final.bytes(key)
        return self.own.bytes(key)

    def read(self, key):
        b = self.raw(key)
        return None if b is None else pd.read_csv(io.BytesIO(b))

    def texts(self, key):
        """Every copy of a text file: the final kit's first, then this label's (wb_log.txt keeps every attempt)."""
        for arc in (self.final, self.own):
            if arc is not None and arc.has(key):
                yield arc.bytes(key).decode("utf-8-sig", "replace")


def angle(v):
    return np.degrees(np.arctan2(v[1], v[0]))


def best_curve(res, gid, state):
    """(tag, curve) of the final ('') or first-attempt ('_autofailed') curve that reached further, or None."""
    best = None
    for tag in ("", "_autofailed"):
        c = res.read(f"{gid}/{state}{tag}_curve.csv")
        if c is not None and len(c) and (best is None or c.u_applied_mm.iloc[-1] > best[1].u_applied_mm.iloc[-1]):
            best = (tag, c)
    return best


def scalars(res, gid, state):
    """Scalars of one state (see module docstring); {} if there is no curve."""
    best = best_curve(res, gid, state)
    if best is None:
        return {}
    tag, c = best
    u, f, e = (np.r_[0, c[k].to_numpy(float)] for k in ("u_applied_mm", "fx_disp_N", "U_total_mJ"))
    # lowest force valley; the origin (F = 0) and a curve with no dip must not count as a minimum
    valleys = [f[i] for i in range(2, len(f) - 1) if f[i] < f[i - 1] and f[i] <= f[i + 1]]
    d = {"curve": "first" if tag else "final", "u_end_mm": u[-1], "F_min_N": min(valleys) if valleys else np.nan}
    neg = np.where(f < 0)[0]
    if not len(neg):
        d["class"] = "mono" if u[-1] >= 12 else "unknown"
        d["F_max_N"] = f.max()
        return d
    a = neg[0]
    u1 = u[a - 1] + (u[a] - u[a - 1]) * f[a - 1] / (f[a - 1] - f[a])
    d.update({"F_max_N": f[:a].max(), "u1_mm": u1})
    d["dE_f_mJ"] = np.interp(u1, u, e)
    back = next((k for k in range(a, len(f)) if f[k] > 0), None)
    if back is None:
        d["class"] = "neg_end"
        return d
    w = f[back - 1] / (f[back - 1] - f[back])
    u2 = u[back - 1] + w * (u[back] - u[back - 1])
    d.update({"class": "bi", "u2_mm": u2, "dE_r_mJ": d["dE_f_mJ"] - np.interp(u2, u, e)})
    nodes = res.read(f"{gid}/{state}{tag}_landmark_nodes.csv").set_index("name")
    disp = res.read(f"{gid}/{state}{tag}_landmark_disp.csv").set_index(["landmark", "set"])[["ux_mm", "uz_mm"]]

    def moved(name):
        q = nodes.loc[name]
        x0 = np.array([q.x0_mm, q.z0_mm])
        d0 = disp.loc[(q.landmark, int(c.set.iloc[back - 2]))].to_numpy(float) if back >= 2 else np.zeros(2)
        return x0, x0 + (1 - w) * d0 + w * disp.loc[(q.landmark, int(c.set.iloc[back - 1]))].to_numpy(float)

    a0, a1 = moved("A_upper")
    b0, b1 = moved("B_upper")
    d["theta2_deg"] = abs(angle(b1 - a1) - angle(b0 - a0))
    return d


def load_state(res, gid, state):
    """Best curve of one state with the substep to draw: the second stable point (bi) or the last converged substep.

    Note: unlike `scalars`, the curve here has no origin row prepended, so the interpolation weight w and the landmark
    sets are those of the curve rows themselves (as in the original viewer data)."""
    best = best_curve(res, gid, state)
    if best is None:
        return None
    tag, c = best
    f, u = c.fx_disp_N.to_numpy(float), c.u_applied_mm.to_numpy(float)
    neg = np.where(f < 0)[0]
    back = next((k for k in range(neg[0], len(f)) if f[k] > 0), None) if len(neg) else None
    if back is not None:
        w = f[back - 1] / (f[back - 1] - f[back])
        at, kind = u[back - 1] + w * (u[back] - u[back - 1]), "bi"
        i0, i1 = back - 1, back
    else:
        at, w, i0, i1 = u[-1], 1.0, len(u) - 1, len(u) - 1
        kind = "neg_end" if len(neg) else ("mono" if u[-1] >= 12 else "unknown")
    nodes = res.read(f"{gid}/{state}{tag}_landmark_nodes.csv")
    disp = res.read(f"{gid}/{state}{tag}_landmark_disp.csv")
    s0, s1 = int(c.set.iloc[i0]), int(c.set.iloc[i1])
    table = disp.set_index(["landmark", "set"])[["ux_mm", "uz_mm"]]
    points = {}
    for _, q in nodes.iterrows():
        x0 = np.array([q.x0_mm, q.z0_mm])
        points[q["name"]] = (x0, x0 + (1 - w) * table.loc[(q.landmark, s0)].to_numpy(float) + w * table.loc[(q.landmark, s1)].to_numpy(float))
    return dict(points=points, u=at, kind=kind, F_min=f.min(), tag=tag, u_end=u[-1], curve=c)


def latest(progress):
    """Last progress.csv record of every (geometry, state)."""
    return progress.groupby(["geom_id", "state"]).tail(1)


def first_solve_mesh(texts, state):
    """(mesh_rule, nodes) in use when the latest failed first solve was kept: wb_log.txt keeps every attempt, and the
    first copy (final kit before the PC) that has one wins."""
    for text in texts:
        mesh = found = None
        for line in text.splitlines():
            m = re.match(rf"{state} mesh_rule=(\S+) nodes=(\d+)", line)
            if m:
                mesh = m.group(1), int(m.group(2))
            elif line.startswith(f"{state} auto_failed_dir="):
                found = mesh
        if found:
            return found
    return None
