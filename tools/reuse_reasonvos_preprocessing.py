"""Reuse only audited CLIP/QA inputs, never grounding or segmentation results."""
import ast
import json
from pathlib import Path
import shutil

from utils.reasonvos_runtime import (
    REPO_ROOT, DATA_ROOT, OVIS_PATH, CLIP_PATH, atomic_json, file_sha256, normalize_qa,
)


def read(path):
    return json.loads(path.read_text())


def functions(path, names):
    tree = ast.parse(path.read_text())
    found = {n.name: ast.dump(n) for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}
    if set(found) != set(names):
        raise ValueError(f'Missing audited functions: {path}')
    return found


def reuse_preprocessing(source, run):
    source, run = Path(source).resolve(), Path(run).resolve()
    if source == run:
        raise ValueError('Use a new run directory for the new protocol')
    source_qa_id = read(source / 'qa_identity.json')
    source_clip_id = read(source / 'clip_identity.json')
    manifest = read(run / 'manifest.json')
    parameters = manifest['parameters']
    for name, expected in [('seed', parameters['worker_seed_base']),
                           ('gpus', parameters['gpus']), ('retries', parameters['max_query_num'])]:
        if source_qa_id[name] != expected:
            raise ValueError(f'QA cache configuration differs: {name}')
    if read(source / 'model_manifest.json') != read(run / 'model_manifest.json'):
        raise ValueError('Cached model revisions differ')
    if (source / 'environment.txt').read_text() != (run / 'environment.txt').read_text():
        raise ValueError('Cached model environment differs')
    snapshot = source / 'source_snapshot'
    generator = 'utils/Ovis_preprocess_query_reasonvos.py'
    if (file_sha256(REPO_ROOT / generator) != source_qa_id['source'] or
            file_sha256(snapshot / generator) != source_qa_id['source']):
        raise ValueError('QA generator source differs')
    proof = {}
    # QA imports these helpers, not the reflection controller or its budgets.
    for relative, names, identity_key in (
        ('utils/reasonvos_runtime.py', ['atomic_json', 'normalize_qa', 'ensure_identity', 'file_sha256'], 'runtime'),
        ('utils/reasonvos_reflection.py', ['record_model_response'], 'reflection'),
    ):
        if file_sha256(snapshot / relative) != source_qa_id[identity_key]:
            raise ValueError(f'Historical dependency source changed: {relative}')
        if functions(snapshot / relative, names) != functions(REPO_ROOT / relative, names):
            raise ValueError(f'QA helper behavior differs: {relative}')
        proof[relative] = {'identical_QA_helpers': names,
                           'historical_sha256': file_sha256(snapshot / relative),
                           'current_sha256': file_sha256(REPO_ROOT / relative)}
    # The recorder context variable must retain the same initialization too.
    def context_assignment(path):
        return [ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'MODEL_TRACE' for t in n.targets)]
    if context_assignment(snapshot / 'utils/reasonvos_reflection.py') != context_assignment(REPO_ROOT / 'utils/reasonvos_reflection.py'):
        raise ValueError('QA trace context differs')
    clip_source = 'utils/preprocess_reasonvos.py'
    if file_sha256(snapshot / clip_source) != source_clip_id['source']:
        raise ValueError('Historical CLIP source changed')
    if ast.dump(ast.parse((snapshot / clip_source).read_text())) != ast.dump(ast.parse((REPO_ROOT / clip_source).read_text())):
        raise ValueError('CLIP computation differs')
    if source_clip_id['model_config'] != file_sha256(Path(CLIP_PATH) / 'config.json'):
        raise ValueError('CLIP configuration differs')
    if source_clip_id['weights'] != {'clip': 1.0, 'iqa': 0.0}:
        raise ValueError('CLIP scoring weights differ')
    for name, filename in [('model_config', 'config.json'), ('generation_config', 'generation_config.json')]:
        if source_qa_id[name] != file_sha256(Path(OVIS_PATH) / filename):
            raise ValueError(f'Ovis configuration differs: {name}')
    original_meta = source / 'meta_expressions.json'
    if (file_sha256(original_meta) != source_qa_id['meta'] or
            file_sha256(original_meta) != source_clip_id['meta'] or
            file_sha256(original_meta) != file_sha256(run / 'meta_expressions.json')):
        raise ValueError('Full-run metadata differs')
    qa_all = read(source / 'query_qa/reasonvos_query_qa.json')
    clip_all = read(source / 'video_scores_reasonvos.json')
    for directory, meta_path in ((run, run / 'meta_expressions.json'),
                                 (run / 'smoke', run / 'smoke_meta.json')):
        videos = read(meta_path)['videos']
        qa, clips, copied = {}, {}, {}
        for video, info in videos.items():
            clips[video] = {'CLIP': {}, 'IQA': clip_all[video]['IQA']}
            for expression in info['expressions']:
                exp = str(expression['exp_id'])
                key = f'{video}/{exp}'
                item = qa_all[key]
                if item['exp'].strip() != expression['exp_text'].strip():
                    raise ValueError(f'Cached query differs: {key}')
                normalize_qa(item['qa_complete'])
                checkpoint = source / 'query_qa/checkpoints' / video / (exp + '.json')
                if read(checkpoint) != item:
                    raise ValueError(f'Cached QA aggregate differs: {key}')
                paths = [checkpoint]
                for name, suffix in [('rng', '.rng.pt'), ('trace', '.trace.json')]:
                    artifact = Path(str(checkpoint) + suffix)
                    if file_sha256(artifact) != item['artifact_sha256'][name]:
                        raise ValueError(f'Cached QA artifact changed: {artifact}')
                    paths.append(artifact)
                for path in paths:
                    copied[str(path.relative_to(source / 'query_qa'))] = file_sha256(path)
                qa[key] = item
                scores = clip_all[video]['CLIP'][exp]
                if any(frame + '.jpg' not in scores for frame in info['frames']):
                    raise ValueError(f'CLIP frame coverage differs: {key}')
                clips[video]['CLIP'][exp] = scores
        qa_id = {**source_qa_id, 'meta': file_sha256(meta_path),
                 'runtime': file_sha256(REPO_ROOT / 'utils/reasonvos_runtime.py'),
                 'reflection': file_sha256(REPO_ROOT / 'utils/reasonvos_reflection.py')}
        clip_id = {**source_clip_id, 'meta': file_sha256(meta_path),
                   'source': file_sha256(REPO_ROOT / clip_source), 'model_path': CLIP_PATH,
                   'image_root': str((DATA_ROOT / 'JPEGImages').resolve())}
        provenance = directory / 'preprocessing_cache_provenance.json'
        identity = {'source_run': str(source), 'meta_sha256': file_sha256(meta_path),
                    'expressions': len(qa), 'source_qa_identity': source_qa_id,
                    'source_clip_identity': source_clip_id, 'QA_helper_proof': proof,
                    'copied_QA_files_sha256': copied,
                    'scope': 'Fixed CLIP/QA inputs only; no grounding, reflection outputs or masks imported'}
        if provenance.exists():
            if (read(provenance) != identity or read(directory / 'qa_identity.json') != qa_id or
                    read(directory / 'clip_identity.json') != clip_id or
                    read(directory / 'query_qa/reasonvos_query_qa.json') != qa or
                    read(directory / 'video_scores_reasonvos.json') != clips):
                raise ValueError(f'Imported preprocessing inputs changed: {directory}')
            continue
        if (directory / 'query_qa').exists() or (directory / 'video_scores_reasonvos.json').exists():
            raise ValueError(f'Refuse to overwrite unversioned preprocessing outputs: {directory}')
        for relative, expected_hash in copied.items():
            target = directory / 'query_qa' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / 'query_qa' / relative, target)
            if file_sha256(target) != expected_hash:
                raise ValueError(f'Copy checksum differs: {target}')
        # Match the producer's JSON encoding, so restart does not change hashes.
        (directory / 'query_qa/reasonvos_query_qa.json').write_text(json.dumps(qa, indent=4))
        atomic_json(clips, directory / 'video_scores_reasonvos.json')
        atomic_json(qa_id, directory / 'qa_identity.json')
        atomic_json(clip_id, directory / 'clip_identity.json')
        atomic_json(identity, provenance)
