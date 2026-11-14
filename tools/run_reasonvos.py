"""Run the current paper-aligned ReasonVOS implementation in an isolated directory."""
import argparse
import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.reasonvos_runtime import (
    DATA_ROOT, RUN_ROOT, SOURCE_FILES, REFLECTION_BUDGET_SEMANTICS,
    atomic_json, normalize_qa, ensure_identity, file_sha256,
)
from utils.reorder_reasonvos_meta import reorder_metadata, complete_frame_metadata
from tools.reuse_reasonvos_preprocessing import reuse_preprocessing


def run_stage(name, script, arguments, env, directory):
    atomic_json({'state': 'running', 'stage': name, 'updated_at': time.time()}, directory / 'status.json')
    with (directory / (name + '.log')).open('a') as log:
        subprocess.run([sys.executable, '-u', str(ROOT / script), *arguments],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def pipeline(meta_path, directory, args):
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), REASONVOS_META=str(meta_path),
               REASONVOS_RUN=str(directory), REASONVOS_ROOT=str(DATA_ROOT),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    videos = json.loads(meta_path.read_text())['videos']
    expected = {(video, str(e['exp_id'])) for video, v in videos.items() for e in v['expressions']}
    if len(expected) != sum(len(v['expressions']) for v in videos.values()):
        raise RuntimeError('Duplicate per-video expression IDs in run metadata')
    common = ['-ng', str(args.num_gpus), '--max_query_num', str(args.max_query_num),
              '--seed', str(args.seed)]
    try:
        run_stage('01_clip', 'utils/preprocess_reasonvos.py', [], env, directory)
        run_stage('02_query_qa', 'utils/Ovis_preprocess_query_reasonvos.py', common, env, directory)
        qa = json.loads((directory / 'query_qa/reasonvos_query_qa.json').read_text())
        if set(qa) != {f'{v}/{e}' for v, e in expected}:
            raise RuntimeError('Query QA expression coverage mismatch')
        for v, info in videos.items():
            for expression in info['expressions']:
                entry = qa[f"{v}/{expression['exp_id']}"]
                if entry['exp'].strip() != expression['exp_text'].strip():
                    raise RuntimeError('Query QA text mismatch')
                normalize_qa(entry['qa_complete'])  # A missing field must not become an empty list.
        run_stage('03_grounding', 'eval/Ovis_infer_reasonvos.py',
                  [*common, '--max_existence_checks', str(args.max_existence_checks),
                   '--max_consistency_checks', str(args.max_consistency_checks)], env, directory)
        results_path = directory / 'predictions/Annotations/all_results.json'
        results = json.loads(results_path.read_text())
        actual = {(r['video_name'], str(r['exp_id'])) for r in results}
        if actual != expected or len(results) != len(expected):
            raise RuntimeError(f'Grounding coverage {len(actual)}/{len(expected)}')
        statuses = Counter(r['reflection']['status'] for r in results)
        atomic_json({'expressions': len(results), 'status_counts': dict(statuses),
                     'empty_qa': sum(not e['qa_complete'] for e in qa.values()),
                     'revision_counts': dict(Counter(str(r['reflection']['revisions']) for r in results)),
                     'budget_semantics': REFLECTION_BUDGET_SEMANTICS,
                     'max_existence_checks': args.max_existence_checks,
                     'max_consistency_checks': args.max_consistency_checks,
                     'check_usage_counts': dict(Counter(
                         str(r['reflection']['checks_used']) for r in results))},
                    directory / 'reflection_summary.json')
        run_stage('04_sam2', 'eval/post_processing.py', [
            '-ng', str(args.num_gpus), '--all_results_path', str(results_path)], env, directory)
        missing_masks = []
        expected_masks = 0
        for video, expression in expected:
            info = videos[video]
            entry = next(e for e in info['expressions'] if str(e['exp_id']) == expression)
            gt = DATA_ROOT / 'Annotations' / f"{info['source']}_{video}_{entry['obj_id']}"
            gt_frames = list(gt.glob('*.png'))
            if not gt_frames:
                raise RuntimeError(f'Missing GT frame directory: {gt}')
            expected_masks += len(gt_frames)
            for frame in gt_frames:
                prediction = directory / 'predictions/Annotations' / video / expression / frame.name
                if not prediction.is_file() or prediction.stat().st_size == 0:
                    missing_masks.append(str(prediction))
        atomic_json({'expected_expressions': len(expected), 'expected_masks': expected_masks,
                     'missing_masks': missing_masks}, directory / 'coverage.json')
        if missing_masks:
            raise RuntimeError(f'Missing {len(missing_masks)} prediction masks')
        run_stage('05_evaluation', 'eval/eval_reasonvos.py', [
            '--meta_exp_path', str(meta_path), '--mask_path', str(DATA_ROOT / 'Annotations'),
            '--pred_path', str(directory / 'predictions/Annotations'),
            '--num_workers', str(args.eval_workers)], env, directory)
        summary = json.loads((directory / 'predictions/summary.json').read_text())
        atomic_json({'state': 'complete', 'stage': 'evaluation', 'summary': summary,
                     'updated_at': time.time()}, directory / 'status.json')
        print(json.dumps({'run': str(directory), 'summary': summary}), flush=True)
    except Exception as exc:
        atomic_json({'state': 'failed', 'error': repr(exc), 'updated_at': time.time()}, directory / 'status.json')
        raise


def prepare_run(args):
    directory = args.run_dir.resolve()
    original = DATA_ROOT / 'meta_expressions.json'
    metadata, mapping = reorder_metadata(json.loads(original.read_text()))
    metadata, changes = complete_frame_metadata(
        metadata, DATA_ROOT / 'JPEGImages', DATA_ROOT / 'Annotations')
    processed_digest = hashlib.sha256(
        json.dumps(metadata, indent=2, ensure_ascii=False).encode()).hexdigest()
    parameters = {'alpha': .3, 'beta': .7, 'N': 10, 'K': 5,
                  'max_existence_checks': args.max_existence_checks,
                  'max_consistency_checks': args.max_consistency_checks,
                  'max_total_checks': args.max_existence_checks + args.max_consistency_checks,
                  'max_query_num': args.max_query_num,
                  'gpus': args.num_gpus, 'worker_seed_base': args.seed,
                  'reflection_budget_semantics': REFLECTION_BUDGET_SEMANTICS,
                  'consistency_failure': 'inconsistent_targets > 0.3 * predicted_targets',
                  'empty_qa': 'not_checked_empty_qa', 'training_free': True}
    model_manifest = {'ovis_sam2': json.loads((ROOT / '.downloads/manifest.json').read_text()),
                      'clip': json.loads((ROOT / '.downloads/clip_manifest.json').read_text())}
    manifest = {'schema': 'reasonvos-independent-checks-v1',
                'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'data_root': str(DATA_ROOT.resolve()), 'run_root': str(directory),
                'preprocessing_cache': str(args.preprocessing_cache.resolve()) if args.preprocessing_cache else None,
                'meta_sha256': file_sha256(original), 'processed_meta_sha256': processed_digest,
                'source_sha256': {p: file_sha256(ROOT / p) for p in SOURCE_FILES},
                'model_manifest_sha256': {p: file_sha256(ROOT / '.downloads' / p)
                                          for p in ('manifest.json', 'clip_manifest.json')},
                'parameters': parameters,
                'generation_config': json.loads((ROOT / 'AIDC-AI/Ovis2.5-9B/generation_config.json').read_text())}
    # Check before changing metadata or an existing manifest, including prepare-only.
    ensure_identity(directory / 'run_identity.json', manifest,
                    [directory / name for name in ('manifest.json', 'meta_expressions.json',
                                                   'predictions', 'query_qa', 'smoke')])
    atomic_json(manifest, directory / 'manifest.json')
    atomic_json(model_manifest, directory / 'model_manifest.json')
    processed = directory / 'meta_expressions.json'
    atomic_json(metadata, processed)
    atomic_json(mapping, directory / 'expression_id_mapping.json')
    atomic_json(changes, directory / 'frame_metadata_changes.json')
    # Reuse the full-run remapped IDs for the smoke subset.
    video = min(metadata['videos'], key=lambda k: len(metadata['videos'][k]['frames']))
    subset = copy.deepcopy(metadata['videos'][video])
    subset['expressions'] = subset['expressions'][:1]
    smoke_meta = directory / 'smoke_meta.json'
    atomic_json({'videos': {video: subset}}, smoke_meta)
    if not (directory / 'environment.txt').exists():
        with (directory / 'environment.txt').open('w') as log:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=log, check=True)
    if args.preprocessing_cache:
        reuse_preprocessing(args.preprocessing_cache, directory)
    return processed, smoke_meta


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, default=RUN_ROOT)
    parser.add_argument('--preprocessing-cache', type=Path,
                        help='Audited CLIP/QA inputs from a matching historical run; never imports predictions')
    parser.add_argument('--max_existence_checks', type=int, default=2, choices=(0, 1, 2))
    parser.add_argument('--max_consistency_checks', type=int, default=2, choices=(0, 1, 2))
    parser.add_argument('--max_query_num', type=int, default=3)
    parser.add_argument('--num_gpus', '-ng', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_workers', type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true', help='CPU metadata and manifest checks only')
    mode.add_argument('--smoke-only', action='store_true', help='Run one expression end to end')
    args = parser.parse_args()
    if min(args.max_existence_checks, args.max_consistency_checks) < 0 or min(args.max_query_num, args.num_gpus, args.eval_workers) < 1:
        parser.error('Budgets/workers must be positive; stage budgets may be zero')
    return args


if __name__ == '__main__':
    args = parse_args()
    processed, smoke_meta = prepare_run(args)
    if args.prepare_only:
        print(json.dumps({'state': 'prepared', 'run': str(args.run_dir.resolve()),
                          'meta': str(processed)}, ensure_ascii=False))
    else:
        pipeline(smoke_meta, args.run_dir.resolve() / 'smoke', args)
        if not args.smoke_only:
            pipeline(processed, args.run_dir.resolve(), args)
