"""Search and filter geometries in the browser, and see each one's shape, force and energy curves and values.

Two layouts:
  shared folder (build/share/CABD_viewer, made by `python -m cabd.ansys.dataset share`): this file is viewer.py next
    to CABD_dataset.csv, CABD_dataset_columns.csv, CABD_viewer_data.json.gz, viewer.html and param_geom_v3.py
    (= cabd/geometry/params.py); run `python viewer.py`, needs only numpy, pandas, matplotlib.
  repository: `python -m cabd.ansys.viewer.app` reads data/CABD_dataset.csv, data/CABD_dataset_columns.csv and
    data/curves/curves_base.json.gz; --include-test uses data/CABD_test_results.csv and data/curves/curves_test.json.gz.
The files are read again whenever they change, so a replaced dataset shows up without restarting.
Test-set results stay hidden unless --include-test is given and CABD_test_results.csv and the test curves
(CABD_viewer_test.json.gz in the shared layout) are in --test-dir.

usage: python viewer.py [--include-test] [--test-dir DIR] [--port 8765] [--no-browser]
"""
import argparse
import base64
import collections
import gzip
import io
import json
import logging
import re
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAGE = HERE / "viewer.html"
if (HERE / "CABD_dataset.csv").exists():          # shared folder: everything next to this file
    sys.path.insert(0, str(HERE))
    import param_geom_v3 as pg
    DATA, COLUMNS, CURVES = HERE / "CABD_dataset.csv", HERE / "CABD_dataset_columns.csv", HERE / "CABD_viewer_data.json.gz"
    TEST_DIR, TEST_CURVES = HERE, "CABD_viewer_test.json.gz"
else:                                              # repository layout
    from cabd.config import DATA as _DATA
    from cabd.geometry import params as pg
    DATA, COLUMNS, CURVES = _DATA / "CABD_dataset.csv", _DATA / "CABD_dataset_columns.csv", _DATA / "curves" / "curves_base.json.gz"
    TEST_DIR, TEST_CURVES = _DATA, "curves/curves_test.json.gz"
STATES = ("partial", "full")
INK, SECONDARY, RULE = "#1d1d1f", "#6e6e73", "#e5e5ea"
COLOR = {"partial": "#0071e3", "full": "#f56300"}
FILL = {"frame": "#d9d8d4", "arm": "#cde2fb", "hinge": "#f6c3ad"}
DESIGN = ["t_h", "L_h", "t_a", "L_a", "L", "t_f", "c", "rho"]
MAX_CHOICES = 12
INCLUDE_TEST = False
RENDER = threading.Lock()
_cache = {}
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # fonts missing on another OS fall back quietly
plt.rcParams.update({"font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10, "svg.fonttype": "path",
                     "axes.edgecolor": RULE, "xtick.color": SECONDARY, "ytick.color": SECONDARY, "axes.labelcolor": SECONDARY,
                     "axes.titlecolor": INK, "axes.unicode_minus": False})


def curves(path):
    """Curves file, read again only when it changes."""
    if not path.exists():
        return {}
    stamp = path.stat().st_mtime
    if _cache.get(path, (None,))[0] != stamp:
        with gzip.open(path, "rt", encoding="ascii") as f:
            _cache[path] = (stamp, json.load(f))
    return _cache[path][1]


def load_rows():
    df = pd.read_csv(DATA)
    test_file = TEST_DIR / "CABD_test_results.csv"
    if INCLUDE_TEST and test_file.exists():
        test = pd.read_csv(test_file).set_index("geom_id", drop=False)
        df = pd.concat([df[~df.geom_id.isin(test.index)], test], ignore_index=True).set_index("geom_id", drop=False).loc[df.geom_id].reset_index(drop=True)
    return df


def rows_json():
    df = load_rows()
    meta = pd.read_csv(COLUMNS).set_index("column")
    columns = []
    for c in df.columns:
        m = meta.loc[c] if c in meta.index else None
        kind = "bool" if m is not None and m.type == "0/1" else "number" if pd.api.types.is_numeric_dtype(df[c]) else "text"
        section = "Geometry" if c in DESIGN else "Partially wet" if c.startswith("partial_") else "Fully wet" if c.startswith("full_") else "Identity"
        item = dict(name=c, kind=kind, section=section, unit="" if m is None or m.unit == "-" else m.unit,
                    description="" if m is None else m.description)
        if kind == "text":
            values = sorted(df[c].dropna().unique().tolist())
            if len(values) <= MAX_CHOICES:
                item["choices"] = values
        columns.append(item)
    updated = time.strftime("%Y-%m-%d %H:%M", time.localtime(DATA.stat().st_mtime))
    return f'{{"updated":{json.dumps(updated)},"columns":{json.dumps(columns)},"rows":{df.to_json(orient="values")}}}'


def blank(ax, text):
    ax.set_axis_off()
    ax.text(.5, .5, text, ha="center", va="center", transform=ax.transAxes, color=SECONDARY, fontsize=11)


def fill_of(name):
    return FILL["frame" if name == "frame" else "arm" if name.startswith("arm") else "hinge"]


def clean(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelsize=9, colors=SECONDARY)
    ax.set_xlabel("x (mm)", fontsize=9, color=SECONDARY)
    ax.set_ylabel("z (mm)", fontsize=9, color=SECONDARY)


def angle(v):
    return np.degrees(np.arctan2(v[1], v[0]))


def draw_shape(ax, points):
    """Deformed bodies from the landmark positions, with the upper and lower A-to-B probe vectors."""
    polys = collections.defaultdict(list)
    for name, (_, xy) in points.items():
        m = re.match(r"(.+)_v\d+$", name)
        if m:
            polys[m.group(1)].append(xy)
    for name, poly in polys.items():
        ax.add_patch(Polygon(np.array(poly), closed=True, ec=SECONDARY, lw=.7, fc=fill_of(name)))
    for side in ("upper", "lower"):
        ax.annotate("", xy=points[f"B_{side}"][1], xytext=points[f"A_{side}"][1],
                    arrowprops=dict(arrowstyle="-|>", color=COLOR["partial"], lw=2, shrinkA=0, shrinkB=0))
    ax.set_aspect("equal")


def draw_angle(ax, points):
    a0, b0 = points["A_upper"][0], points["B_upper"][0]
    a1, b1 = points["A_upper"][1], points["B_upper"][1]
    v0, v1 = b0 - a0, b1 - a1
    t0, t1 = angle(v0), angle(v1)
    end = a1 + 1.25 * np.linalg.norm(v1) * v0 / np.linalg.norm(v0)
    ax.plot([a1[0], end[0]], [a1[1], end[1]], ls="--", color=COLOR["full"], lw=1.5)
    r = .55 * np.linalg.norm(v1)
    ax.add_patch(Arc(a1, 2 * r, 2 * r, theta1=min(t0, t1), theta2=max(t0, t1), color=COLOR["full"], lw=2))
    return abs(t1 - t0)


def draw_undeformed(ax, row):
    p = dict(pg.REF, **{v: float(row[v]) for v in DESIGN})
    d = pg.derived(p)
    points = []
    for name, (_, poly) in pg.bodies(p, n=16).items():
        ax.add_patch(Polygon(poly, closed=True, ec=SECONDARY, lw=.7, fc=fill_of(name)))
        points.append(poly)
    for sign in (1, -1):
        ax.annotate("", xy=(-(p["t_h"] + d["R_slot"]), sign * d["H"]), xytext=(-p["L"], sign * d["H"]),
                    arrowprops=dict(arrowstyle="-|>", color=COLOR["partial"], lw=2, shrinkA=0, shrinkB=0))
    ax.set_aspect("equal")
    ax.set_title("Undeformed", loc="left", fontweight="semibold", fontsize=12)
    clean(ax)
    return np.vstack(points)


def to_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", facecolor="white")
    plt.close(fig)
    return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def figures(gid, row, reveal, data):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.7), constrained_layout=True)
    curve_fig, curve_ax = plt.subplots(figsize=(11.4, 3.5), constrained_layout=True)
    energy_fig, energy_ax = plt.subplots(figsize=(11.4, 3.5), constrained_layout=True)
    stress_fig, stress_ax = plt.subplots(figsize=(11.4, 3.5), constrained_layout=True)
    extent = [draw_undeformed(axes[0], row)]
    visible, stressed = [], False
    for ax, state in zip(axes[1:], STATES):
        status = row[f"{state}_run_status"]
        name = "Partially wet" if state == "partial" else "Fully wet"
        if status != "finished":
            blank(ax, f"{name}: {status.replace('_', ' ')}")
            continue
        if not reveal:
            blank(ax, f"{name}: sealed")
            continue
        d = data.get(gid, {}).get(state)
        if d is None:
            blank(ax, f"{name}: curve not in this dataset")
            continue
        points = {k: (np.array(v[:2]), np.array(v[2:])) for k, v in d["points"].items()}
        draw_shape(ax, points)
        theta = draw_angle(ax, points)
        where = "second stable point" if d["kind"] == "bi" else "last converged step"
        ax.set_title(f"{name}, {where}\nu = {d['at']:.2f} mm, rotation = {theta:.1f}\N{DEGREE SIGN}", loc="left", fontweight="semibold", fontsize=12, linespacing=1.35)
        clean(ax)
        extent.append(np.array([xy for _, xy in points.values()]))
        u, f, energy = (np.r_[0, d[k]] for k in ("u", "F", "U"))
        curve_ax.plot(u, f, color=COLOR[state], lw=2, label=name, solid_capstyle="round")
        energy_ax.plot(u, energy, color=COLOR[state], lw=2, label=name, solid_capstyle="round")
        for key, marker in (("u1_mm", "v"), ("u2_mm", "o")):
            at = row[f"{state}_{key}"]
            if pd.notna(at):
                curve_ax.plot(at, 0, marker, color=COLOR[state], ms=8, mec="white", mew=1.2, zorder=4)
                energy_ax.plot(at, np.interp(at, u, energy), marker, color=COLOR[state], ms=8, mec="white", mew=1.2, zorder=4)
        visible.append((u, f, energy))
        if "S" in d:  # curves files made before 2026-11-12 have no stress
            stress = np.r_[0, d["S"]]
            k = int(stress.argmax())
            body = d["S_body"].replace("_", " ")
            stress_ax.plot(u, stress, color=COLOR[state], lw=2, solid_capstyle="round",
                           label=f"{name}: peak {stress[k]:.3g} MPa in {body} at u = {u[k]:.2f} mm")
            stress_ax.plot(u[k], stress[k], "o", color=COLOR[state], ms=7, mec="white", mew=1.2, zorder=4)
            stressed = True
    e = np.vstack(extent)
    for ax in axes:
        if ax.axison:
            ax.set_xlim(e[:, 0].min() - 1.2, e[:, 0].max() + 1.2)
            ax.set_ylim(e[:, 1].min() - 1.2, e[:, 1].max() + 1.2)
    shapes = to_uri(fig)
    if not visible:
        for f_ in (curve_fig, energy_fig, stress_fig):
            plt.close(f_)
        return shapes, None, None, None
    # the vertical axes fit the snap region: the steep end of a curve would flatten it
    second = [row[f"{s}_u2_mm"] for s in STATES if pd.notna(row[f"{s}_u2_mm"])]
    limit = max(second) + .5 if second else 12
    for ax, column, label in ((curve_ax, 1, "Reaction force (N)"), (energy_ax, 2, "Strain energy (mJ)")):
        parts = [v[column][v[0] <= limit] for v in visible]
        lo, hi = min(p.min() for p in parts), max(p.max() for p in parts)
        ax.set_ylim(lo - .15 * (hi - lo), hi + .25 * (hi - lo))
        ax.axhline(0, color=SECONDARY, lw=.8)
        ax.plot([], [], "v", color=SECONDARY, label="u1")
        ax.plot([], [], "o", color=SECONDARY, label="u2")
        ax.set_xlabel("Applied displacement (mm)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(frameon=False, ncols=4, loc="upper center", fontsize=10)
        ax.grid(axis="y", color=RULE, lw=.8)
        ax.tick_params(length=0, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    stress_uri = None
    if stressed:
        stress_ax.set_ylim(bottom=0)
        stress_ax.set_xlabel("Applied displacement (mm)", fontsize=10)
        stress_ax.set_ylabel("Max von Mises stress (MPa)", fontsize=10)
        stress_ax.legend(frameon=False, loc="upper left", fontsize=10)
        stress_ax.grid(axis="y", color=RULE, lw=.8)
        stress_ax.tick_params(length=0, labelsize=9)
        for s in stress_ax.spines.values():
            s.set_visible(False)
        stress_uri = to_uri(stress_fig)
    else:
        plt.close(stress_fig)
    return shapes, to_uri(curve_fig), to_uri(energy_fig), stress_uri


def geometry_json(gid):
    df = load_rows()
    hit = df[df.geom_id == gid]
    if hit.empty:
        return json.dumps(dict(found=False, id=gid))
    row = hit.iloc[0]
    is_test = row.group == "test"
    sealed = is_test and not (INCLUDE_TEST and (TEST_DIR / "CABD_test_results.csv").exists())
    if pd.isna(row["t_h"]):
        return json.dumps(dict(found=True, id=gid, placeholder=True, sealed=False))
    data = curves(TEST_DIR / TEST_CURVES) if is_test else curves(CURVES)
    shapes, curve, energy, stress = figures(gid, row, not sealed, data)
    return json.dumps(dict(found=True, id=gid, placeholder=False, sealed=sealed, shapes=shapes, curve=curve, energy=energy, stress=stress))


class Handler(BaseHTTPRequestHandler):
    def reply(self, body, kind, code=200):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{kind}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                self.reply(PAGE.read_text(encoding="utf-8"), "text/html")
            elif url.path == "/api/rows":
                self.reply(rows_json(), "application/json")
            elif url.path == "/api/geometry":
                with RENDER:  # matplotlib is not thread-safe
                    self.reply(geometry_json(parse_qs(url.query).get("id", [""])[0].strip().upper()), "application/json")
            else:
                self.send_error(404)
        except Exception as exc:
            self.reply(json.dumps(dict(error=repr(exc))), "application/json", 500)

    def log_message(self, *args):
        pass


def free_port(start):
    for port in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("no free port found")


def main():
    global INCLUDE_TEST, TEST_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--test-dir", type=Path, default=TEST_DIR)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    INCLUDE_TEST, TEST_DIR = args.include_test, args.test_dir.resolve()
    for need in (DATA, COLUMNS, PAGE):
        if not need.exists():
            raise SystemExit(f"missing {need.name}: keep viewer.py in the folder with the dataset files")
    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    print("Open", url, "(close this window to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
