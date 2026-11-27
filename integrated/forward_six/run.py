"""Build, smoke-check, fit or cross-validate the six candidates; no HPO search."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parents[1]
for folder in [BASE, BASE / 'forward', BASE.parent / 'src']:
    sys.path.insert(0, str(folder))
from runtime_paths import RUNS
from common import load_data, dump, metrics_cls, reg_report, TARGETS
from forward_six.models import NAMES, default_config, fit_candidate, predict_candidate
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def score(model, held):
    p = predict_candidate(model, held)
    c = held.label.notna().to_numpy()
    r = held.is_reg.to_numpy(bool)
    return {'stability': metrics_cls(held.loc[c, 'label'].eq('bi>mono'), p.loc[c, 'bi_mono']),
            'regression': reg_report(held.loc[r, TARGETS].to_numpy(float), p.loc[r, TARGETS].to_numpy(float))}


def describe():
    return {'status': 'construction baseline specification; HPO results are recorded separately in forward-six/tuning; no new final Test selection',
            'prediction_averaging': False, 'output_directory': 'forward-six',
            'candidates': {n: {'base': n.removeprefix('rom_'), 'rom': n.startswith('rom_'),
                               'provisional_config_not_tuned': default_config(n)} for n in NAMES},
            'origins': {'mlp': 'Seungmin single neural network, joint 3-mean/3-log-variance regression',
                        'gp': 'Seungmin ARD Matern 5/2 and two-stage classification',
                        'xgboost': 'Jinwoo XGBoost with state-conditioned classification'},
            'rom_inputs': {'regression': '8 dimensions + partial ROM angle/displacement/log10 barrier = 11',
                           'mlp_classification': '8 dimensions + 3 partial ROM outputs = 11',
                           'gp_classification': '8 dimensions + corresponding-state ROM minimum force = 9',
                           'xgboost_classification': '8 dimensions + corresponding-state ROM minimum force + state code = 10'},
            'rom_curve_source': 'unified-forward/data/completed_force_curves.json',
            'evaluation': 'Existing geometry Train/Test and Train folds; no Test used for calibration or fitting',
            'stress': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['describe', 'smoke', 'train', 'cv', 'predict'])
    parser.add_argument('--models', nargs='+', choices=NAMES, default=list(NAMES))
    parser.add_argument('--config', type=Path, help='One candidate configuration with classification/regression sections')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args(argv)
    destination = RUNS / 'forward-six'
    if args.action == 'describe':
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return
    if args.action == 'predict':
        if args.checkpoint is None or args.input is None or args.output is None:
            parser.error('predict requires --checkpoint, --input and --output')
        result = predict_candidate(joblib.load(args.checkpoint), pd.read_csv(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        return
    data = load_data()
    training = data[data.role.eq('train')].copy()
    curves = RUNS / 'unified-forward/data/completed_force_curves.json'
    supplied = json.loads(args.config.read_text()) if args.config else None
    if (supplied or (args.output and args.action in ['train', 'cv'])) and len(args.models) != 1:
        parser.error('--config or a train/cv --output requires exactly one candidate in --models')
    if args.action == 'smoke':
        # Deliberately Train-only. This is an execution check, never a performance result.
        chunks = [q.sort_values('geom_id').head(16) for _, q in training[training.label.notna()].groupby('label')]
        training = pd.concat(chunks).sort_values('geom_id').reset_index(drop=True)
    records = []
    with threadpool_limits(limits=1):
        for name in args.models:
            if args.action in ['train', 'cv']:
                suffix = 'joblib' if args.action == 'train' else 'json'
                folder = 'models' if args.action == 'train' else 'cv'
                planned = args.output or destination / folder / name / f'seed{args.seed}.{suffix}'
                if planned.exists():
                    raise FileExistsError(planned)
            start = time.perf_counter()
            cfg = supplied
            if args.action == 'smoke':
                cfg = default_config(name)
                for section in cfg.values():
                    if name.endswith('mlp'):
                        section.update(width=16, depth=1, epochs=4)
                    elif name.endswith('xgboost'):
                        section.update(n_estimators=8, max_depth=2)
                    else:
                        section.update(optimize_kernel=False)
            if args.action == 'cv':
                folds, oof = [], []
                for fold in sorted(training.fold.unique()):
                    if fold < 0:
                        raise ValueError('Training population has unassigned CV folds')
                    train = training[training.fold != fold]
                    held = training[training.fold == fold].reset_index(drop=True)
                    m = fit_candidate(name, train, cfg, args.seed, curves_path=curves,
                                      rom_cache=destination / 'rom')
                    fold_score = score(m, held)
                    pred = predict_candidate(m, held)
                    oof.append((held, pred))
                    if name.endswith('mlp'):
                        eligible = held.is_reg.to_numpy(bool)
                        sd = train.loc[train.is_reg.astype(bool), TARGETS].std(ddof=0).to_numpy()
                        fold_score['scaled_mae'] = float(np.mean(np.abs(
                            pred.loc[eligible, TARGETS].to_numpy() - held.loc[eligible, TARGETS].to_numpy()).mean(0)
                            / np.maximum(sd, 1e-8)))
                    folds.append({'fold': int(fold), **fold_score})
                held_all = pd.concat([q for q, _ in oof], ignore_index=True)
                predicted_all = pd.concat([p for _, p in oof], ignore_index=True)
                cls_mask = held_all.label.notna().to_numpy()
                reg_mask = held_all.is_reg.to_numpy(bool)
                pooled = {'stability': metrics_cls(held_all.loc[cls_mask, 'label'].eq('bi>mono'),
                                                   predicted_all.loc[cls_mask, 'bi_mono']),
                          'regression': reg_report(held_all.loc[reg_mask, TARGETS].to_numpy(float),
                                                   predicted_all.loc[reg_mask, TARGETS].to_numpy(float))}
                if name.endswith('mlp'):
                    pooled['joint_scaled_mae'] = float(np.mean([f['scaled_mae'] for f in folds]))
                record = {'candidate': name, 'seed': args.seed, 'folds': folds, 'pooled_oof': pooled,
                          'note': 'One supplied configuration only; no HPO or Test evaluation',
                          'seconds': time.perf_counter() - start}
                out = args.output or destination / 'cv' / name / f'seed{args.seed}.json'
                if out.exists():
                    raise FileExistsError(out)
                dump(out, record)
            else:
                model = fit_candidate(name, training, cfg, args.seed, curves_path=curves,
                                      rom_cache=destination / ('smoke-rom' if args.action == 'smoke' else 'rom'),
                                      rom_max_nfev=4 if args.action == 'smoke' else None)
                if args.action == 'smoke':
                    query = training[training.is_reg].head(6).reset_index(drop=True)
                    before = predict_candidate(model, query)
                    out = destination / 'smoke' / name / f'seed{args.seed}.joblib'
                    out.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, out)
                    after = predict_candidate(joblib.load(out), query)
                    np.testing.assert_allclose(before[TARGETS + ['p_bi_mono']], after[TARGETS + ['p_bi_mono']], rtol=0, atol=0)
                    pd.testing.assert_series_equal(before.bi_mono, after.bi_mono)
                    record = {'candidate': name, 'status': 'fit/predict/reload passed',
                              'training_rows': len(training), 'prediction_rows': len(query),
                              'single_networks': 2 if name.endswith('mlp') else None,
                              'prediction_averaging': False, 'tuning_started': False,
                              'test_evaluated': False, 'seconds': time.perf_counter() - start}
                else:
                    out = args.output or destination / 'models' / name / f'seed{args.seed}.joblib'
                    if out.exists():
                        raise FileExistsError(out)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, out)
                    record = {'candidate': name, 'seed': args.seed, 'training_rows': len(training),
                              'config': model['config'], 'prediction_averaging': False,
                              'test_evaluated': False, 'seconds': time.perf_counter() - start}
                    dump(out.with_suffix('.json'), record)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    if args.action == 'smoke':
        dump(args.output or destination / 'smoke' / 'validation.json',
             {'purpose': 'execution check only; NOT full training, HPO, CV or Test evaluation', 'results': records})


if __name__ == '__main__':
    main()
