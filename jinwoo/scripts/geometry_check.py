"""Official v3 geometry checks, extracted from the original evaluation runner."""
import math


def check(batch):
    import param_geom_v3 as geom

    original = geom._dedup
    cache = {}

    def dedup(points, tol=1e-9):
        # Same sequential tolerance test, reusing each polygon's result per candidate.
        key = id(points), tol
        if key not in cache:
            keep = [0]
            coords = points.tolist()
            for i, (x, z) in enumerate(coords[1:], 1):
                a, b = coords[keep[-1]]
                if math.hypot(x - a, z - b) > tol:
                    keep.append(i)
            x, z = coords[keep[-1]]
            a, b = coords[keep[0]]
            if len(keep) > 1 and math.hypot(x - a, z - b) <= tol:
                keep.pop()
            cache[key] = (points, points[keep])
        return cache[key][1]

    def result(x):
        p = dict(zip(geom.VAR_NAMES, x))
        if not all(a <= p[k] <= b for k, (a, b) in geom.RANGES.items()):
            return False
        p['t_w'] = geom.T_W
        return all(geom.check_constraints(p).values())

    try:
        geom._dedup = dedup
        passes = []
        for x in batch:
            cache.clear()
            passes.append(result(x))
    finally:
        geom._dedup = original
    return passes
