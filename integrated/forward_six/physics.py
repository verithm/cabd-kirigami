"""Existing Seungmin ROM equations, calibrated on the current Train partition only."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from cabd.models import rom
from common import FEATURES, dump

PHYSICS_VERSION = 'seungmin-rigid-arm-springs-v1'


def fit_physics(training, curves_path, cache_dir=None, *, max_nfev=None):
    if training.empty or not training.role.eq('train').all():
        raise ValueError('ROM calibration requires Train rows only')
    curves_path = Path(curves_path)
    if not curves_path.exists():
        raise FileNotFoundError('ROM calibration requires completed_force_curves.json; scalar CSV alone is insufficient')
    curve_bytes = curves_path.read_bytes()
    curve_hash = hashlib.sha256(curve_bytes).hexdigest()
    ordered = training.sort_values('geom_id')
    fingerprint = hashlib.sha256(
        PHYSICS_VERSION.encode() + curve_hash.encode() +
        ordered[['geom_id', *FEATURES]].to_csv(index=False).encode() + str(max_nfev).encode()).hexdigest()
    cache = Path(cache_dir) / f'{fingerprint}.json' if cache_dir else None
    if cache is not None and cache.exists():
        record = json.loads(cache.read_text())
        if record['fingerprint'] != fingerprint or record['training_ids'] != ordered.geom_id.tolist():
            raise ValueError('ROM cache population mismatch')
        return np.array(record['theta']), record
    curves = json.loads(curve_bytes)
    rows = []
    for _, row in ordered.iterrows():
        geo = rom.geometry(row)
        for state in ['partial', 'full']:
            key = f'{row.geom_id}|{state}'
            if key in curves:
                curve = curves[key]
                u, f = np.asarray(curve['u'], float), np.asarray(curve['F'], float)
                if len(u) < 2 or u.shape != f.shape or not np.isfinite(u).all() or not np.isfinite(f).all():
                    raise ValueError(f'Invalid training force curve: {key}')
                rows.append((key, geo, state, u, f))
    if not rows:
        raise ValueError('No completed force curves match this Train partition')
    geo = {key: np.array([r[1][key] for r in rows])[:, None] for key in rows[0][1]}
    ef = np.array([rom.E_DRY if r[2] == 'partial' else rom.E_WET for r in rows])[:, None]
    scales = [max(np.max(np.abs(r[4])), 1e-3) for r in rows]

    def residual(theta):
        ct, ca, cf, ch = np.exp(theta[:4])
        kt = ct * rom.E_WET * rom.B * geo['t_h'] ** 3 / (12 * geo['l_tip'])
        ka = ca * rom.E_WET * rom.B * geo['t_a'] ** 3 / (12 * geo['l_apex'])
        kf = cf * 3 * ef * (rom.B * geo['t_f'] ** 3 / 12) / geo['L_fl'] ** 3
        kh = ch * rom.E_WET * rom.B * geo['t_h'] / geo['l_tip']
        kl = 1 / (1 / kf + 1 / kh)
        x0 = -(geo['L'] - geo['t_a'] / 2) * theta[4]
        z0 = geo['z_t0']
        radius = np.hypot(x0, z0)
        xa = x0 + rom.U_GRID[None, :]
        zt = np.sqrt(np.maximum(radius ** 2 - xa ** 2, 0))
        phi = np.arctan2(zt, -xa)
        dp = phi - phi[:, :1]
        energy = kl * (zt - z0) ** 2 + kt * dp ** 2 + .5 * ka * (2 * dp) ** 2
        force = np.gradient(energy, rom.U_GRID, axis=1)
        return np.concatenate([(np.interp(r[3], rom.U_GRID, f) - r[4]) / scale
                               for r, f, scale in zip(rows, force, scales)])

    fitted = least_squares(residual, [0., 0., 0., -5., 1.],
                           bounds=([-8.] * 4 + [.5], [8.] * 4 + [1.5]), max_nfev=max_nfev)
    # Ensure the vectorized calibration still matches the original physical model.
    row = rows[0]
    expected = (np.interp(row[3], rom.U_GRID, rom.curve(row[1], row[2], fitted.x)[1]) - row[4]) / scales[0]
    np.testing.assert_allclose(residual(fitted.x)[:len(expected)], expected, atol=1e-10, rtol=1e-9)
    record = {'physics_version': PHYSICS_VERSION, 'fingerprint': fingerprint,
              'curve_sha256': curve_hash, 'training_ids': ordered.geom_id.tolist(),
              'calibration_curve_keys': [r[0] for r in rows], 'theta': fitted.x.tolist(),
              'normalized_force_rmse': float(np.sqrt(np.mean(residual(fitted.x) ** 2))),
              'solver_success': bool(fitted.success), 'solver_message': str(fitted.message),
              'nfev': fitted.nfev, 'max_nfev': max_nfev}
    if cache is not None:
        dump(cache, record)
    return fitted.x, record


def physics_features(d, theta):
    f = rom.features(d, theta)
    values = np.column_stack([f.rom_partial_theta2, f.rom_partial_u2,
                              np.maximum(f.rom_partial_logE, -4.2)])
    return values, f.rom_partial_Fmin.to_numpy(), f.rom_full_Fmin.to_numpy()
