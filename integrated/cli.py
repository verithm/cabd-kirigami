"""One entry point for common preprocessing, Forward and inverse comparisons."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, help='Private experiment directory; default artifacts/integrated')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Re-extract labels from local result ZIPs and joined jobs.csv')
    p.add_argument('--raw-roots', type=Path, nargs='+', required=True)
    p.add_argument('--jobs', type=Path, required=True)
    p.add_argument('--case-index', type=Path)
    p.add_argument('--reference-targets', type=Path)
    sub.add_parser('forward-data', help='Build common geometry rows and fixed Train folds')
    p = sub.add_parser('train-forward', help='Compare and refit Forward families (potentially long-running)')
    p.add_argument('--group', choices=['all', 'jinwoo', 'seungmin'], default='all')
    sub.add_parser('report-forward', help='Aggregate Test metrics, CV selection and charts')
    p = sub.add_parser('predict', help='Predict geometry CSV with saved Forward weights')
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--model', default='sm_mlp')
    p.add_argument('--seed', type=int, default=0)
    sub.add_parser('inverse', help='Train and compare five inverse methods; NO final inverse model selection')
    sub.add_parser('target-strategies', help='Compare five target-generation strategies with physical mJ acceptance')
    sub.add_parser('without-barrier', help='Rescore SAME pools without barrier acceptance; no retraining or search')
    sub.add_parser('relaxed', help='Historical slightly relaxed log-tolerance paired rescoring')
    sub.add_parser('review-barrier', help='Recover Train OOF physical errors and compare absolute barrier thresholds')
    p = sub.add_parser('audit', help='Reload and check saved experiment artifacts')
    p.add_argument('--scope', choices=['forward', 'inverse', 'barrier'], default='forward')
    args = parser.parse_args(argv)
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    if args.runs:
        env['CABD_INTEGRATED_RUNS'] = str(args.runs.expanduser().resolve())
    runs = Path(env.get('CABD_INTEGRATED_RUNS', ROOT / 'artifacts/integrated')).expanduser().resolve()
    env.setdefault('MPLCONFIGDIR', str(runs / '.matplotlib'))
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        env[name] = '1'
    def execute(folder, script, *extra):
        subprocess.run([sys.executable, str(BASE / folder / script), *map(str, extra)], env=env, check=True)
    def require(*paths):
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            parser.error('Missing local artifacts: ' + ', '.join(missing))
    if args.command == 'prepare':
        opts = ['--raw-roots', *args.raw_roots, '--jobs', args.jobs]
        if args.case_index:
            opts += ['--case-index', args.case_index]
        if args.reference_targets:
            opts += ['--reference-targets', args.reference_targets]
        execute('forward', 'prepare.py', *opts)
    elif args.command == 'forward-data':
        require(runs / 'unified-forward/data/scalar_targets.csv', runs / 'unified-forward/data/jobs.csv')
        execute('forward', 'common.py')
    elif args.command == 'train-forward':
        require(runs / 'unified-forward/data/scalar_targets.csv', runs / 'unified-forward/data/jobs.csv')
        if args.group in ['all', 'jinwoo']:
            execute('forward', 'run_sklearn.py')
        if args.group in ['all', 'seungmin']:
            execute('forward', 'run_seungmin.py')
    elif args.command == 'report-forward':
        families=['jw_mlp','jw_xgboost','jw_gp','jw_extra_trees','jw_svm','jw_mlp_xgboost','sm_mlp','sm_gp','sm_rom_gp','sm_referee','sm_referee_nogp']
        require(*[runs/'unified-forward/models'/name/'complete.json' for name in families])
        execute('forward', 'report.py')
        execute('forward', 'plot_results.py')
    elif args.command == 'predict':
        require(args.input, runs / 'unified-forward/models' / args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        execute('forward', 'predict.py', args.model, args.input, args.output, '--seed', args.seed)
    elif args.command in ['inverse', 'target-strategies', 'without-barrier', 'relaxed', 'review-barrier']:
        require(runs/'unified-forward/forward_selection.json', runs / 'unified-forward/models/sm_mlp/joint_s0.joblib', runs / 'unified-forward/models/sm_mlp/stability_s0.joblib')
        if args.command == 'inverse':
            execute('inverse', 'run_all.py')
        elif args.command == 'target-strategies':
            for method, ext in [('jw_cvae','.pt'),('sm_cvae','.pt'),('conditional_gmm','.joblib')]:
                require(runs / 'unified-inverse/models' / method / ('seed0' + ext))
            execute('inverse', 'goal_range_benchmark.py')
        elif args.command == 'without-barrier':
            require(runs / 'inverse-target-strategies/summary.csv')
            execute('inverse', 'evaluate_no_barrier.py')
        elif args.command == 'relaxed':
            require(runs / 'unified-inverse/summary.csv')
            execute('inverse', 'reevaluate_tolerances.py')
        else:
            require(runs / 'unified-inverse/summary.csv')
            execute('inverse', 'review_barrier_tolerance.py')
            execute('inverse', 'audit_barrier_tolerance_review.py')
            execute('inverse', 'make_barrier_tolerance_report.py')
    elif args.command == 'audit':
        folder, script = {'forward':('forward','audit_models.py'), 'inverse':('inverse','audit_results.py'), 'barrier':('inverse','audit_barrier_tolerance_review.py')}[args.scope]
        execute(folder, script)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
