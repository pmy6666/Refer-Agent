"""Audit completed ReasonVOS predictions without changing inference artifacts.

GT pixels are used only for this post-inference diagnostic. Group differences
are descriptive associations, not causal estimates of reflection improvements.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor_diagnostic(mask_path, boxes, points, cache):
    if len(boxes) != len(points):
        raise ValueError('Diagnostic input has different box/point counts')
    geometry = {'points_outside_their_boxes': sum(
        not (x1 <= x < x2 and y1 <= y < y2)
        for (x1, y1, x2, y2), (x, y) in zip(boxes, points))}
    if not mask_path.exists():
        return {'anchor_gt': 'not_annotated', **geometry}
    if mask_path not in cache:
        with Image.open(mask_path) as image:
            cache[mask_path] = np.asarray(image.convert('L')) > 0
    mask = cache[mask_path]
    pixels = int(mask.sum())
    height, width = mask.shape
    covered = np.zeros_like(mask)
    for x1, y1, x2, y2 in boxes:
        # Pixel boxes use the same half-open convention as NumPy.
        left, right = max(0, int(x1)), min(width, int(x2))
        top, bottom = max(0, int(y1)), min(height, int(y2))
        if left < right and top < bottom:
            covered[top:bottom, left:right] = True
    intersection = int((mask & covered).sum())
    return {'anchor_gt': 'visible' if pixels else 'absent',
            **geometry,
            'anchor_gt_pixels': pixels,
            'anchor_box_gt_intersection_pixels': intersection,
            'anchor_box_gt_recall': intersection / pixels if pixels else None,
            'anchor_point_gt_hits': sum(
                0 <= int(x) < width and 0 <= int(y) < height and bool(mask[int(y), int(x)])
                for x, y in points)}


def analyze(run):
    manifest_path, meta_path = run / 'manifest.json', run / 'meta_expressions.json'
    if run.name == 'smoke' and not manifest_path.exists():
        manifest_path, meta_path = run.parent / 'manifest.json', run.parent / 'smoke_meta.json'
    manifest = read(manifest_path)
    videos = read(meta_path)['videos']
    summary = read(run / 'predictions/summary.json')
    scores = read(run / 'predictions/reason_vos.json')
    qa = read(run / 'query_qa/reasonvos_query_qa.json')
    expected = sum(len(info['expressions']) for info in videos.values())
    if summary['evaluation_policy'] != 'complete_coverage':
        raise ValueError('A diagnostic/incomplete evaluation cannot support this report')
    if summary['expressions'] != expected or len(scores) != expected:
        raise ValueError('Evaluation expression count differs from run metadata')
    root = Path(manifest['data_root'])
    rows, groups = [], defaultdict(list)
    for video, info in videos.items():
        mask_cache = {}
        for expression in info['expressions']:
            exp = str(expression['exp_id'])
            key = f"{info['source']}_{video}_{expression['obj_id']}_{exp}"
            j, f = scores[key]
            directory = run / 'predictions/Annotations' / video / exp
            result = read(directory / 'grounding.json')
            trace_path = directory / 'trace.json'
            if digest(trace_path) != result['artifact_sha256']['trace']:
                raise ValueError(f'Trace checksum changed: {trace_path}')
            trace = read(trace_path)
            frames = info['frames']
            if result['frames'] != frames:
                raise ValueError(f'Grounding frame sequence mismatch: {video}/{exp}')
            reflection = result['reflection']
            row = {
                'video': video, 'exp_id': exp, 'query': expression['exp_text'],
                'source': info['source'], 'is_sent': expression['is_sent'],
                'J': 100 * j, 'F': 100 * f, 'J&F': 50 * (j + f),
                'status': reflection['status'], 'revisions': reflection['revisions'],
                'empty_qa': not qa[f'{video}/{exp}']['qa_complete'],
                'model_calls': len(trace['model_calls']),
                'key_frame': frames[result['key_frame_idx']],
                'targets': len(result['all_boxes']),
                'descriptions': result['descriptions'],
            }
            mask_directory = (root / 'Annotations' /
                              f"{info['source']}_{video}_{expression['obj_id']}")
            row.update(anchor_diagnostic(mask_directory / (row['key_frame'] + '.png'),
                                         result['all_boxes'], result['all_points'], mask_cache))
            trajectory = []
            for event in trace['rounds']:
                if 'key_frame_idx' not in event:
                    continue  # Reflection-disabled traces may only have a status.
                frame = frames[event['key_frame_idx']]
                trajectory.append({
                    'revision': event['revision'], 'action': event['action'],
                    'key_frame': frame, 'boxes': event['boxes'], 'points': event['points'],
                    'existence': event['existence']['status'],
                    'consistency': event['consistency']['status'],
                    **anchor_diagnostic(mask_directory / (frame + '.png'),
                                        event['boxes'], event['points'], mask_cache),
                })
            if trajectory and (trajectory[-1]['boxes'] != result['all_boxes'] or
                               trajectory[-1]['key_frame'] != row['key_frame']):
                raise ValueError(f'Final grounding differs from its last verified state: {video}/{exp}')
            row['trajectory'] = trajectory
            rows.append(row)
            for group, value in (
                ('source', row['source']), ('is_sent', row['is_sent']),
                ('reflection_status', row['status']), ('revisions', row['revisions']),
                ('empty_qa', row['empty_qa']), ('anchor_gt', row['anchor_gt']),
            ):
                groups[f'{group}/{value}'].append([row['J'], row['F'], row['J&F']])
    raw = np.asarray([[r['J'], r['F'], r['J&F']] for r in rows])
    means = dict(zip(('J', 'F', 'J&F'), raw.mean(axis=0).tolist()))
    for name, value in means.items():
        if not np.isclose(value, summary[name], rtol=0, atol=1e-9):
            raise ValueError(f'Saved mean does not match per-expression scores: {name}')
    visible = [r for r in rows if r['anchor_gt'] == 'visible']
    trajectories = [r for r in rows if r['trajectory']]
    def hit(state):
        return state['anchor_gt'] == 'visible' and state['anchor_box_gt_intersection_pixels'] > 0
    report = {
        'scope': ('full_dataset' if expected == 458 and len(videos) == 91
                  and digest(meta_path) == manifest['processed_meta_sha256'] else 'subset_only'),
        'expressions': expected, 'videos': len(videos), 'metrics': means,
        'paper_target': {'J': 67.0, 'F': 72.7, 'J&F': 69.8},
        'gap_to_paper': {name: means[name] - target for name, target in
                         (('J', 67.0), ('F', 72.7), ('J&F', 69.8))},
        'groups': {name: {'n': len(values), **dict(zip(
            ('J', 'F', 'J&F'), np.asarray(values).mean(axis=0).tolist()))}
                   for name, values in sorted(groups.items())},
        'counts': {
            'anchor_gt_absent': sum(r['anchor_gt'] == 'absent' for r in rows),
            'anchor_gt_unannotated': sum(r['anchor_gt'] == 'not_annotated' for r in rows),
            'visible_gt_but_box_no_overlap': sum(
                r['anchor_box_gt_intersection_pixels'] == 0 for r in visible),
            'visible_gt_but_no_positive_point_hit': sum(
                r['anchor_point_gt_hits'] == 0 for r in visible),
            'expressions_with_point_outside_its_box': sum(
                r['points_outside_their_boxes'] > 0 for r in rows),
            'whole_video_JF_below_10': sum(r['J&F'] < 10 for r in rows),
            'empty_qa': sum(r['empty_qa'] for r in rows),
            'model_calls': sum(r['model_calls'] for r in rows),
            'reflection_statuses': dict(Counter(r['status'] for r in rows)),
            'initial_anchor_gt_absent': sum(
                r['trajectory'][0]['anchor_gt'] == 'absent' for r in trajectories),
            'initial_visible_gt_but_box_no_overlap': sum(
                r['trajectory'][0]['anchor_gt'] == 'visible' and not hit(r['trajectory'][0])
                for r in trajectories),
            'initial_box_miss_recovered_by_final': sum(
                r['trajectory'][0]['anchor_gt'] == 'visible' and not hit(r['trajectory'][0])
                and hit(r) for r in trajectories),
            'initial_box_hit_lost_by_final': sum(
                hit(r['trajectory'][0]) and r['anchor_gt'] != 'not_annotated' and not hit(r)
                for r in trajectories),
            'verified_but_whole_video_JF_below_10': sum(
                r['status'] == 'verified' and r['J&F'] < 10 for r in rows),
        },
        'worst_20': sorted(rows, key=lambda r: (r['J&F'], r['video'], r['exp_id']))[:20],
        'limitations': [
            'Groups describe association; they do not isolate the causal effect of a repair.',
            'Anchor visibility means nonzero GT pixels, not sufficient causal/temporal evidence.',
            'Box/GT intersection is a localization diagnostic, not SAM2 segmentation quality.',
            'Unannotated anchor frames are excluded from visibility/localization diagnoses.',
            'A subset score must not be compared as a full-dataset reproduction result.',
        ],
        'inputs': {str(p): digest(p) for p in (
            manifest_path, meta_path,
            run / 'predictions/summary.json', run / 'predictions/reason_vos.json')},
    }
    return report, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True,
                        help='Separate diagnostic directory; inference artifacts are read-only')
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output.resolve()
    if output == run or run in output.parents:
        parser.error('Keep diagnostic output outside the evaluated run directory')
    report, rows = analyze(run)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('report.json', report), ('per_expression.json', rows)]:
        path = output / name
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
        temporary.replace(path)
    print(json.dumps({k: report[k] for k in ('scope', 'expressions', 'metrics', 'counts')},
                     ensure_ascii=False))
