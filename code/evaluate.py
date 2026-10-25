"""Verify the frozen release, then time fresh official prediction and scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RELEASE = '1.0'


def release_hashes():
    files = sorted([*ROOT.glob('*.py'), *(ROOT / 'selector').glob('*.py'),
                    *(ROOT / 'tests').glob('*.py'), ROOT / 'requirements.txt',
                    *(ROOT / 'checkpoint').glob('*')])
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


def verify_release(create=False):
    path = ROOT / 'release.json'
    hashes = release_hashes()
    if not path.exists():
        if not create:
            raise FileNotFoundError('Missing release.json; freeze once with --freeze before evaluation')
        version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        record = {'release': RELEASE, 'release_date': '2026-10-25',
            'frozen_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'code_version': version, 'sha256': hashes,
            'source': 'Equivalent cleanup of reviewed v22; no model or parameter selection',
            'method': 'submission.py', 'checkpoint': 'checkpoint',
            'validation_role': 'development and model selection; not independent evaluation',
            'test_role': 'previously scored; cleanup evaluation is reproduction only'}
        path.write_text(json.dumps(record, indent=2) + '\n')
    record = json.loads(path.read_text())
    if record['release'] != RELEASE or record['sha256'] != hashes:
        raise RuntimeError('Frozen source/configuration/weight hashes differ from this release')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--freeze', action='store_true', help='Create the initial immutable release manifest')
    parser.add_argument('--reproduce', action='store_true', help='Allow overwriting existing scores for unchanged code')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    os.chdir(ROOT)
    release = verify_release(args.freeze)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / f'{args.split}-predictions.json'
    score = output / f'{args.split}-score.json'
    if (score.exists() or predictions.exists()) and not args.reproduce:
        raise FileExistsError('Use --reproduce or a new --output-dir; existing evidence is preserved')
    asset_bytes = sum(p.stat().st_size for p in (ROOT / 'checkpoint').iterdir() if p.is_file())
    if asset_bytes > 16 * 2**20:
        raise RuntimeError('Inference assets exceed 16 MiB')
    env = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[key] = '1'
    env['OMP_NUM_THREADS'] = '4'
    env['OMP_DYNAMIC'] = 'FALSE'
    commands = [
        [sys.executable, 'predict.py', '--method', 'submission.py', '--checkpoint', 'checkpoint',
         '--split', args.split, '--output', str(predictions)],
        [sys.executable, 'score_predictions.py', '--split', args.split,
         '--predictions', str(predictions), '--output', str(score)],
    ]
    elapsed = []
    for command in commands:
        start = time.perf_counter()
        process = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
        elapsed.append(time.perf_counter() - start)
        print(process.stdout, end='', flush=True)
        if process.stderr:
            print(process.stderr, file=sys.stderr)
    hardware = (subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
                if sys.platform == 'darwin' else platform.processor())
    result = {'release': RELEASE, 'split': args.split, 'score': json.loads(score.read_text()),
        'code_version': release['code_version'], 'hardware': hardware,
        'platform': platform.platform(), 'python': platform.python_version(),
        'asset_bytes': asset_bytes, 'asset_mib': asset_bytes / 2**20,
        'fresh_prediction_process_seconds': elapsed[0], 'fresh_scoring_process_seconds': elapsed[1],
        'total_seconds': sum(elapsed), 'includes_startup_imports_io': True,
        'thread_policy': 'BLAS=1; OpenMP/CatBoost<=4; sequential processes and ensemble members',
        'commands': commands, 'reference_cpu_not_measured': True}
    (output / f'{args.split}-resources.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
