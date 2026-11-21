"""Minimal STEP AP214 parser for extruded prismatic bodies (pure Python, no CAD kernel).
Extracts for each MANIFOLD_SOLID_BREP the planar cross-section loops at y = Y0.

usage: python -m cabd.geometry.stepparse <file.step>      print every body's section loops (lines and arcs)
"""
import re
import sys

import numpy as np


def load(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    data = txt.split("DATA;")[1]
    # join continuation lines
    ents = {}
    for m in re.finditer(r"#(\d+)=([A-Z_0-9]+)\((.*?)\);\s*(?=#\d+=|ENDSEC)", data, re.S):
        eid = int(m.group(1)); typ = m.group(2); body = m.group(3).replace("\n", "")
        ents[eid] = (typ, body)
    return ents

def refs(body):
    return [int(x) for x in re.findall(r"#(\d+)", body)]

def nums(body):
    return [float(x) for x in re.findall(r"-?\d+\.\d*(?:E[-+]?\d+)?|-?\d+(?:E[-+]?\d+)?", body)]

class Step:
    def __init__(self, path):
        self.e = load(path)
    def typ(self, i): return self.e[i][0]
    def body(self, i): return self.e[i][1]
    def point(self, i):
        t, b = self.e[i]
        assert t == "CARTESIAN_POINT", (i, t)
        return np.array(nums(b.split(",", 1)[1])[:3])
    def direction(self, i):
        t, b = self.e[i]
        assert t == "DIRECTION", (i, t)
        return np.array(nums(b.split(",", 1)[1])[:3])
    def vertex(self, i):
        return self.point(refs(self.body(i))[0])
    def bodies(self):
        return [i for i, (t, b) in self.e.items() if t == "MANIFOLD_SOLID_BREP"]
    def body_edges(self, bid):
        """return list of (type, p0, p1, extra) for all edges of the body (unique by edge id)"""
        shell = refs(self.body(bid))[0]
        faces = refs(self.body(shell))
        edges = {}
        for f in faces:
            # ADVANCED_FACE('',(bounds...),surface,bool)
            for bnd in refs(self.body(f))[:-1]:
                if self.typ(bnd) not in ("FACE_BOUND", "FACE_OUTER_BOUND"):
                    continue
                loop = refs(self.body(bnd))[0]
                for oe in refs(self.body(loop)):
                    if self.typ(oe) != "ORIENTED_EDGE":
                        continue
                    ec = refs(self.body(oe))[-1]
                    if ec in edges: continue
                    r = refs(self.body(ec))
                    v0, v1, geom = r[0], r[1], r[2]
                    p0, p1 = self.vertex(v0), self.vertex(v1)
                    gt = self.typ(geom)
                    extra = None
                    if gt == "CIRCLE":
                        rr = refs(self.body(geom))
                        ax = rr[0]
                        rad = nums(self.body(geom).split(",")[-1])[0]
                        axr = refs(self.body(ax))
                        c = self.point(axr[0]); n = self.direction(axr[1])
                        extra = dict(center=c, normal=n, r=rad)
                    edges[ec] = (gt, p0, p1, extra)
        return edges

def section(step, bid, y0=0.0, tol=1e-6):
    out = []
    for ec, (gt, p0, p1, ex) in step.body_edges(bid).items():
        if abs(p0[1] - y0) < tol and abs(p1[1] - y0) < tol:
            if gt == "LINE" or (gt == "CIRCLE" and abs(abs(ex["normal"][1]) - 1) < 1e-6):
                out.append((gt, p0, p1, ex))
    return out

def order_loop(edges, tol=1e-6):
    """chain edges into closed loop(s); returns list of list of (gt,p0,p1,ex) oriented"""
    rem = list(edges); loops = []
    while rem:
        gt, p0, p1, ex = rem.pop(0)
        loop = [(gt, p0, p1, ex)]; cur = p1; start = p0
        while np.linalg.norm(cur - start) > tol and rem:
            for k, (g, a, b, x) in enumerate(rem):
                if np.linalg.norm(a - cur) < tol:
                    loop.append((g, a, b, x)); cur = b; rem.pop(k); break
                if np.linalg.norm(b - cur) < tol:
                    loop.append((g, b, a, x)); cur = a; rem.pop(k); break
            else:
                break
        loops.append(loop)
    return loops

def arc_points(p0, p1, ex, n=12):
    c = ex["center"]; r = ex["r"]
    a0 = np.arctan2(p0[2] - c[2], p0[0] - c[0]); a1 = np.arctan2(p1[2] - c[2], p1[0] - c[0])
    # choose the shorter sweep (all fillets here are < 180 deg) -- verified for this model
    d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    ts = a0 + d * np.linspace(0, 1, n)
    return np.c_[c[0] + r * np.cos(ts), c[2] + r * np.sin(ts)]

def loop_polygon(loop, n_arc=12):
    pts = []
    for gt, p0, p1, ex in loop:
        if gt == "LINE":
            pts.append([p0[0], p0[2]])
        else:
            pts.extend(arc_points(p0, p1, ex, n_arc)[:-1].tolist())
    return np.array(pts)

def poly_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

def describe(path):
    s = Step(path)
    res = {}
    for bid in s.bodies():
        name = s.body(bid).split(",")[0].strip("'")
        edges = section(s, bid)
        loops = order_loop(edges)
        res[bid] = dict(name=name, loops=loops)
    return s, res

if __name__ == "__main__":
    s, res = describe(sys.argv[1])
    for bid, d in res.items():
        print(f"\n=== body #{bid} {d['name']}  loops={len(d['loops'])}")
        for lp in d["loops"]:
            P = loop_polygon(lp)
            print(f"  loop n_edges={len(lp)} area={poly_area(P):.4f} bbox x[{P[:,0].min():.3f},{P[:,0].max():.3f}] z[{P[:,1].min():.3f},{P[:,1].max():.3f}]")
            for gt, p0, p1, ex in lp:
                if gt == "LINE":
                    L = np.linalg.norm(p1 - p0)
                    ang = np.degrees(np.arctan2(p1[2]-p0[2], p1[0]-p0[0]))
                    print(f"    LINE ({p0[0]:8.4f},{p0[2]:8.4f}) -> ({p1[0]:8.4f},{p1[2]:8.4f})  L={L:.4f} ang={ang:7.2f}")
                else:
                    print(f"    ARC  ({p0[0]:8.4f},{p0[2]:8.4f}) -> ({p1[0]:8.4f},{p1[2]:8.4f})  r={ex['r']:.3f} c=({ex['center'][0]:.4f},{ex['center'][2]:.4f})")
