"""Extract common energy targets from explicitly selected raw collection archives."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_paths import ROOT, RUNS
sys.path.insert(0, str(ROOT / 'integrated' / 'preprocessing'))
from preprocess_forward import FEATURES, process, read_rows, write_rows
from preprocess_energy_targets import prepare

OUT = RUNS / 'unified-forward' / 'data'

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-roots', type=Path, nargs='+', required=True,
                        help='Directories or individual collect_result_PC*.zip files')
    parser.add_argument('--jobs', type=Path, required=True, help='Joined geometry dimensions and original train/test roles')
    parser.add_argument('--case-index', type=Path, help='Prior index: select exactly its archive names (recommended for historical reproduction)')
    parser.add_argument('--reference-targets', type=Path, help='Optional scalar_forward_targets.csv for exact re-extraction audit')
    args = parser.parse_args()
    jobs = {}
    for row in read_csv(args.jobs):
        if row['geom_id'] in jobs:
            raise ValueError(f'Duplicate geometry in jobs: {row["geom_id"]}')
        if row['role'] not in ('train', 'test') or not all(row.get(k) for k in FEATURES):
            raise ValueError(f'Invalid role or missing dimensions: {row["geom_id"]}')
        jobs[row['geom_id']] = row
    wanted = {r['archive'] for r in read_csv(args.case_index)} if args.case_index else None
    files = {}
    for base in args.raw_roots:
        paths = [base] if base.is_file() else sorted(base.rglob('collect_result_PC*.zip'))
        for path in paths:
            if wanted is not None and path.name not in wanted:
                continue
            if path.name in files and files[path.name].resolve() != path.resolve():
                raise ValueError(f'Ambiguous archive basename: {path.name}')
            files[path.name] = path
    if not files or (wanted is not None and set(files) != wanted):
        raise ValueError(f'Missing collection archives: {sorted((wanted or set()) - set(files))}')
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    curves = {}
    seen = set()
    manifest = []
    for name, path in sorted(files.items()):
        manifest.append(dict(archive=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        with ZipFile(path) as archive:
            members = {n.replace('\\', '/'): n for n in archive.namelist()}
            for row, _ in process(path, .01):
                gid = row['geom_id']
                key = (gid, row['state'])
                if gid not in jobs or key in seen:
                    raise ValueError(f'Unknown geometry or duplicate result: {key}; select the intended archives only')
                seen.add(key)
                row.update({k: jobs[gid][k] for k in FEATURES})
                if row['role'] != jobs[gid]['role']:
                    raise ValueError(f'Train/Test role mismatch: {gid}')
                index.append(row)
                if row['status'] == 'ok' and row['force_complete']:
                    samples = read_rows(archive, members[row['curve_file']])
                    curves[f'{gid}|{row["state"]}'] = {
                        k: [float(r[col]) for r in samples]
                        for k, col in [('u', 'u_applied_mm'), ('F', 'fx_disp_N')]
                    }
    write_rows(OUT / 'case_index.csv', index)
    rows = list(prepare(files, OUT / 'case_index.csv'))
    import numpy as np
    for row in rows:
        energy = row['reverse_barrier_mJ']
        row['log10_reverse_barrier'] = float(np.log10(energy)) if energy is not None and energy > 0 else None
    if not rows:
        raise ValueError('No ok/completed states were found')
    reference_checked = False
    if args.reference_targets:
        before = {(r['geom_id'], r['state']): r for r in read_csv(args.reference_targets)}
        if set(before) != {(r['geom_id'], r['state']) for r in rows}:
            raise ValueError('Reference target population differs')
        for row in rows:
            ref = before[row['geom_id'], row['state']]
            assert row['stability'] == ref['stability']
            for col in ['stable_angle_deg', 'deformation_applied_mm', 'reverse_barrier_mJ']:
                if row[col] is not None:
                    np.testing.assert_allclose(row[col], float(ref[col]), rtol=0, atol=1e-10)
        reference_checked = True
    write_rows(OUT / 'scalar_targets.csv', rows)
    joined = [{**{k: jobs[g][k] for k in ['geom_id', 'role', *FEATURES]}, 'feature_source': jobs[g].get('feature_source', 'jobs_csv')} for g in sorted(jobs)]
    write_rows(OUT / 'jobs.csv', joined, ['geom_id', 'role', *FEATURES, 'feature_source'])
    (OUT / 'completed_force_curves.json').write_text(json.dumps(curves))
    summary = dict(geometries=len(jobs), completed_states=len(rows),
                   labels={f'{a}/{b}/{c}': n for (a,b,c),n in Counter((r['role'],r['state'],r['stability']) for r in rows).items()},
                   sources=manifest, energy_cache_exactly_reproduced=reference_checked, stress_excluded=True)
    (OUT / 'preprocessing_audit.json').write_text(json.dumps(summary, indent=2))
    print('Prepared common energy data:', OUT, len(rows), 'states')

if __name__ == '__main__':
    main()
