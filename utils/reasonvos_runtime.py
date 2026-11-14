"""Shared paths and atomic checkpoints for the local ReasonVOS run."""
import json
import os
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('REASONVOS_ROOT', REPO_ROOT.parent / 'Data' / 'ReasonVOS'))
META_PATH = Path(os.environ.get('REASONVOS_META', DATA_ROOT / 'meta_expressions.json'))
RUN_ROOT = Path(os.environ.get('REASONVOS_RUN', REPO_ROOT / 'runs' / 'reasonvos_e2_c2_20261114'))
OVIS_PATH = str(REPO_ROOT / 'AIDC-AI' / 'Ovis2.5-9B')
CLIP_PATH = str(REPO_ROOT / 'openai' / 'clip-vit-base-patch32')
QA_PATH = RUN_ROOT / 'query_qa' / 'reasonvos_query_qa.json'
SCORES_PATH = RUN_ROOT / 'video_scores_reasonvos.json'

SOURCE_FILES = (
    'eval/Ovis_infer_reasonvos.py', 'utils/preprocess_reasonvos.py',
    'utils/Ovis_preprocess_query_reasonvos.py', 'eval/post_processing.py',
    'eval/eval_reasonvos.py', 'utils/concat_frames_reasonvos.py',
    'utils/reasonvos_runtime.py', 'utils/reorder_reasonvos_meta.py',
    'utils/reasonvos_pipeline.py', 'utils/reasonvos_reflection.py',
    'tools/run_reasonvos.py', 'tools/reuse_reasonvos_preprocessing.py',
)
REFLECTION_BUDGET_SEMANTICS = 'independent_existence_and_consistency_check_caps_including_initial_checks'


def normalize_qa(value):
    """Accept a single generated QA object without dropping its question."""
    if isinstance(value, dict) and 'question' in value:
        value = [value]
    if not isinstance(value, list):
        raise ValueError('QA output must be a list or a single question object')
    for item in value:
        if (not isinstance(item, dict) or not isinstance(item.get('question'), str)
                or not item['question'].strip()):
            raise ValueError('Each QA entry must be a question object')
        if not isinstance(item.get('options'), dict) or not item.get('correct_answer'):
            raise ValueError('Each QA entry must contain options and a correct answer')
        if (len(item['options']) < 2 or item['correct_answer'] not in item['options']
                or any(k not in ('A', 'B', 'C', 'D') or not isinstance(v, str) or not v.strip()
                       for k, v in item['options'].items())):
            raise ValueError('Invalid QA choices or correct answer')
    return value


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_identity(path, identity, legacy_files=()):
    """Refuse stale or unversioned caches before writing into their directory."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(f'Run inputs/code/settings changed: {path}; use a new run directory')
    elif any(Path(p).exists() for p in legacy_files):
        raise ValueError(f'Unversioned results in {path.parent}; use a new run directory')
    else:
        atomic_json(identity, path)


def inference_identity(args):
    inputs = [META_PATH, QA_PATH, SCORES_PATH]
    sources = ['eval/Ovis_infer_reasonvos.py', 'utils/reasonvos_pipeline.py',
               'utils/reasonvos_reflection.py', 'utils/reasonvos_runtime.py',
               'utils/concat_frames_reasonvos.py', 'utils/preprocess_reasonvos.py']
    model_files = [Path(OVIS_PATH) / name for name in
                   ('config.json', 'generation_config.json', 'preprocessor_config.json')]
    model_files += [REPO_ROOT / '.downloads/manifest.json', REPO_ROOT / '.downloads/clip_manifest.json']
    return {'schema': 'reasonvos-independent-checks-v1',
            'inputs': {str(p): file_sha256(p) for p in inputs},
            'sources': {p: file_sha256(REPO_ROOT / p) for p in sources},
            'models': {str(p): file_sha256(p) for p in model_files if p.exists()},
            'model_paths': {'ovis': OVIS_PATH, 'clip': CLIP_PATH},
            'max_existence_checks': args.max_existence_checks,
            'max_consistency_checks': args.max_consistency_checks,
            'reflection_budget_semantics': REFLECTION_BUDGET_SEMANTICS,
            'retries': args.max_query_num,
            'gpus': args.num_gpus, 'seed': args.seed}
