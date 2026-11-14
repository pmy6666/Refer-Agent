"""Import a proven equivalent local QA cache, preserving explicit provenance.

This narrowly scoped migration allows only the audited run-directory default
and reflection-budget default changes. It never imports grounding or masks.
Stop the target Slurm job before --apply; no running cache may be replaced.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.reasonvos_runtime import atomic_json, normalize_qa


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equivalent_dependencies(source, target):
    proof = {}
    for name, identity_key in [('reasonvos_runtime.py', 'runtime'),
                               ('reasonvos_reflection.py', 'reflection')]:
        original = source / 'source_snapshot/utils' / name
        current = ROOT / 'utils' / name
        if sha(original) != read(source / 'qa_identity.json')[identity_key]:
            raise ValueError(f'Historical dependency differs from recorded source: {original}')
        if sha(current) != read(target / 'qa_identity.json')[identity_key]:
            raise ValueError(f'Target dependency differs from active identity: {current}')
        old, new = original.read_text(), current.read_text()
        # These exact audited substitutions are unrelated to text QA generation.
        if name == 'reasonvos_runtime.py':
            old = old.replace('reasonvos_reflection6_20261112', 'reasonvos_paper4_20261114')
        else:
            old = old.replace('max_reflections=6, trace=None', 'max_reflections=4, trace=None')
        if ast.dump(ast.parse(old)) != ast.dump(ast.parse(new)):
            raise ValueError(f'Unaudited dependency changes: {name}')
        proof[name] = {'historical_sha256': sha(original), 'current_sha256': sha(current),
                       'allowed_change': 'run_directory_default' if identity_key == 'runtime'
                       else 'reflection_budget_default_unused_by_QA'}
    return proof


def validate(source, target):
    source_id, target_id = read(source / 'qa_identity.json'), read(target / 'qa_identity.json')
    for key in set(source_id) | set(target_id):
        if key not in ('runtime', 'reflection') and source_id.get(key) != target_id.get(key):
            raise ValueError(f'QA identity mismatch: {key}')
    if sha(ROOT / 'utils/Ovis_preprocess_query_reasonvos.py') != target_id['source']:
        raise ValueError('Current QA generator differs from target identity')
    historical_generator = source / 'source_snapshot/utils/Ovis_preprocess_query_reasonvos.py'
    if sha(historical_generator) != source_id['source']:
        raise ValueError('Historical QA generator differs from its recorded identity')
    if sha(source / 'meta_expressions.json') != sha(target / 'meta_expressions.json'):
        raise ValueError('Metadata differs')
    if sha(source / 'meta_expressions.json') != source_id['meta']:
        raise ValueError('Historical metadata differs from QA identity')
    for name in ('model_manifest.json', 'qa_generation_config_0.json'):
        if read(source / name) != read(target / name):
            raise ValueError(f'Model identity/generation configuration differs: {name}')
    dependencies = equivalent_dependencies(source, target)
    metadata = read(target / 'meta_expressions.json')['videos']
    expected = {f"{video}/{e['exp_id']}": e['exp_text'].strip()
                for video, info in metadata.items() for e in info['expressions']}
    aggregate_path = source / 'query_qa/reasonvos_query_qa.json'
    aggregate = read(aggregate_path)
    if set(aggregate) != set(expected):
        raise ValueError('Historical QA coverage differs from target metadata')
    checksums, fresh_prefix = {}, []
    for key, text in expected.items():
        video, exp = key.rsplit('/', 1)
        checkpoint = source / 'query_qa/checkpoints' / video / (exp + '.json')
        item = read(checkpoint)
        if item != aggregate[key] or item['exp'].strip() != text:
            raise ValueError(f'Historical checkpoint/aggregate/query mismatch: {key}')
        normalize_qa(item['qa_complete'])
        artifacts = [checkpoint]
        for name, suffix in [('rng', '.rng.pt'), ('trace', '.trace.json')]:
            artifact = Path(str(checkpoint) + suffix)
            if sha(artifact) != item['artifact_sha256'][name]:
                raise ValueError(f'Historical artifact checksum mismatch: {artifact}')
            artifacts.append(artifact)
        existing = target / 'query_qa/checkpoints' / video / (exp + '.json')
        if existing.exists():
            if existing.read_bytes() != checkpoint.read_bytes():
                raise ValueError(f'Fresh QA checkpoint differs: {key}')
            for suffix in ('.rng.pt', '.trace.json'):
                if sha(Path(str(existing) + suffix)) != sha(Path(str(checkpoint) + suffix)):
                    raise ValueError(f'Fresh QA artifact differs: {key}{suffix}')
            fresh_prefix.append(key)
        for artifact in artifacts:
            checksums[str(artifact.relative_to(source / 'query_qa'))] = sha(artifact)
    checksums['reasonvos_query_qa.json'] = sha(aggregate_path)
    return {'schema': 'reasonvos-audited-QA-cache-import-v1', 'source_run': str(source),
            'target_run': str(target), 'expressions': len(expected),
            'source_qa_identity': source_id, 'target_qa_identity': target_id,
            'dependencies': dependencies, 'fresh_identical_prefix': fresh_prefix,
            'files_sha256': checksums,
            'scope': 'QA only; CLIP, grounding, reflection and SAM2 are not imported',
            'state': 'validated', 'timestamp': time.time()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--target-run', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    source, target = args.source_run.resolve(), args.target_run.resolve()
    if source == target:
        parser.error('Source and target must differ')
    report = validate(source, target)
    if args.apply:
        stage, backup = target / 'qa_import_staging', target / 'query_qa_fresh_before_import'
        if stage.exists() or backup.exists() or (target / 'qa_cache_import.json').exists():
            raise ValueError('Import staging/backup/provenance already exists; inspect before retrying')
        stage.mkdir()
        for relative, expected_sha in report['files_sha256'].items():
            copied = stage / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / 'query_qa' / relative, copied)
            if sha(copied) != expected_sha:
                raise ValueError(f'Copy checksum mismatch: {relative}')
        atomic_json(report, target / 'qa_cache_import.json')
        (target / 'query_qa').rename(backup)
        stage.rename(target / 'query_qa')
        report['state'] = 'imported'
        atomic_json(report, target / 'qa_cache_import.json')
    print(json.dumps({key: report[key] for key in
                      ('state', 'expressions', 'fresh_identical_prefix', 'scope')}, ensure_ascii=False))
