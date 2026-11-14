"""Assign per-video unique expression IDs without overwriting the source dataset."""
import argparse
import copy
import json
from pathlib import Path
from utils.reasonvos_runtime import DATA_ROOT, RUN_ROOT, atomic_json


def reorder_metadata(metadata):
    reordered = copy.deepcopy(metadata)
    mapping = []
    for video, info in reordered['videos'].items():
        for index, expression in enumerate(info['expressions']):
            mapping.append({'video': video, 'expression_index': index,
                            'original_exp_id': expression['exp_id'],
                            'exp_id': index, 'obj_id': expression['obj_id']})
            expression['exp_id'] = index
    return reordered, mapping


def complete_frame_metadata(metadata, image_root, mask_root):
    """Restore required evaluation frame names without reading GT mask contents.

    Retain the declared input sequence and add only annotation frame names with
    actual RGB files. Other unannotated RGB files do not expand the benchmark.
    """
    updated = copy.deepcopy(metadata)
    changes = []
    for video, info in updated['videos'].items():
        required = set(info['frames'])
        for expression in info['expressions']:
            directory = Path(mask_root) / f"{info['source']}_{video}_{expression['obj_id']}"
            annotation_frames = {p.stem for p in directory.glob('*.png')}
            if not annotation_frames:
                raise ValueError(f'Missing annotation frames: {directory}')
            required.update(annotation_frames)
        actual = sorted(required)
        missing = {frame for frame in actual if not (Path(image_root) / video / (frame + '.jpg')).is_file()}
        if missing or not actual:
            raise ValueError(f'Missing RGB frames for {video}: {sorted(missing)}')
        if info['frames'] != actual:
            changes.append({'video': video, 'added_frames': sorted(set(actual) - set(info['frames'])),
                            'previous_count': len(info['frames']), 'new_count': len(actual)})
        info['frames'] = actual
    return updated, changes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=str(DATA_ROOT / 'meta_expressions.json'))
    parser.add_argument('--output', default=str(RUN_ROOT / 'meta_expressions.json'))
    parser.add_argument('--mapping', default=str(RUN_ROOT / 'expression_id_mapping.json'))
    parser.add_argument('--image_root', default=str(DATA_ROOT / 'JPEGImages'))
    parser.add_argument('--mask_root', default=str(DATA_ROOT / 'Annotations'))
    args = parser.parse_args()
    metadata, mapping = reorder_metadata(json.load(open(args.source)))
    metadata, changes = complete_frame_metadata(metadata, args.image_root, args.mask_root)
    atomic_json(metadata, args.output)
    atomic_json(mapping, args.mapping)
    atomic_json(changes, Path(args.output).parent / 'frame_metadata_changes.json')
