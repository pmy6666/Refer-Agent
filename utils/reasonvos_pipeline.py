"""ReasonVOS adapters retaining the public prompts, layouts and grounding models."""
import math
from pathlib import Path

import numpy as np

from utils.concat_frames_reasonvos import select_top_images, simple_concat_frames, stitch_images
from utils.reasonvos_reflection import (
    answers_match, require_answers, retry_valid, run_reflection,
    validate_descriptions, validate_scores,
)


def infer_expression(upstream, model, scorer, images, frames, image_directory,
                     query, clip_scores, qa, max_existence_checks=2, max_consistency_checks=2, retries=3, pid=0,
                     trace=None, candidate_indices=None, operators_only=False,
                     context_indices=None):
    trace = trace if trace is not None else {'calls': [], 'selection': [], 'rounds': []}

    def observe(stage, indices, rendered):
        ledger = trace.setdefault('observations', [])
        seen = {i for event in ledger for i in event['frame_indices']}
        ledger.append({'module': stage, 'frame_indices': list(indices),
                       'resolution': list(rendered.size),
                       'pixels': rendered.width * rendered.height,
                       'first_access': [i for i in indices if i not in seen],
                       'crop': None})
        return rendered

    def select(guidance, previous_descriptions):
        scores = {frame + '.jpg': dict(clip_scores[frame + '.jpg']) for frame in frames}
        if previous_descriptions and candidate_indices is None:
            trace.setdefault('clip_calls', []).append({
                'stage': 'description_guided_clip', 'frame_indices': list(range(len(frames))),
                'text_count': len(previous_descriptions),
                'image_text_evaluations': len(frames) * len(previous_descriptions)})
            description_scores = {
                name: sum(scorer.calculate_clip_score(str(Path(image_directory) / name), text)
                          for text in previous_descriptions) / len(previous_descriptions)
                for name in scores}
            description_scores = scorer.normalize_scores(description_scores)
            for name, item in scores.items():
                item['weighted_score'] = .1 * item['clip'] + .2 * description_scores[name]
        else:
            for item in scores.values():
                item['weighted_score'] = .1 * item['clip']
        candidates = (select_top_images(scores, 'weighted_score', 1, len(frames))
                      if candidate_indices is None else
                      [frames[i] + '.jpg' for i in candidate_indices])
        indices = [frames.index(Path(name).stem) for name in candidates]
        strip = simple_concat_frames([images[index] for index in indices])
        importance = retry_valid(
            lambda: upstream.score_frame_with_vlm(
                [observe('frame_scores', indices, strip)], query, guidance, model, pid, num_frames=len(indices)),
            lambda result: validate_scores(result, len(indices)), retries, trace['calls'], 'frame_scores')
        fused = [.3 * scores[name]['clip'] + .7 * importance[i]
                 for i, name in enumerate(candidates)]
        selected = sorted(sorted(range(len(indices)), key=lambda i: fused[i], reverse=True)[:5])
        evidence = [indices[i] for i in selected]
        anchor_position = max(range(len(selected)), key=lambda i: fused[selected[i]])
        anchor_index = evidence[anchor_position]
        # Public Dynamic Focus Layout: insert context around a 1-based even anchor.
        layout_indices = list(evidence)
        layout_images = [images[i] for i in evidence]
        layout_position = anchor_position
        if anchor_position % 2 == 1:
            left = (evidence[anchor_position - 1] + anchor_index + 1) // 2
            right = (anchor_index + evidence[anchor_position + 1] + 1) // 2
            if context_indices is not None:
                allowed = [i for i in context_indices if i != anchor_index]
                left = min(allowed, key=lambda i: (abs(i - left), i))
                right = min(allowed, key=lambda i: (abs(i - right), i))
            layout_images.insert(anchor_position, images[left])
            layout_images.insert(anchor_position + 2, images[right])
            layout_position += 1
            layout_indices.insert(anchor_position, left)
            layout_indices.insert(anchor_position + 2, right)
        layout = {'image': stitch_images(layout_images, layout_position),
                  'strip': simple_concat_frames([images[i] for i in evidence]),
                  'key_frame_idx': anchor_index, 'key_frame': images[anchor_index],
                  'key_evidence_position': anchor_position, 'layout_position': layout_position,
                  'layout_count': len(layout_images), 'evidence_indices': evidence,
                  'layout_indices': layout_indices}
        trace['selection'].append({'candidate_indices': indices, 'ovis_scores': importance,
                                   'fused_scores': fused, 'evidence_indices': evidence,
                                   'key_frame_idx': anchor_index, 'guidance': guidance,
                                   'description_guided_clip': bool(previous_descriptions) and candidate_indices is None,
                                   'clip_scores': [scores[name]['clip'] for name in candidates],
                                   'layout_indices': layout_indices})
        return layout

    def describe(layout, history):
        def call():
            arguments = (observe('description', layout['layout_indices'], layout['image']),
                         query, layout['layout_position'] + 1)
            if history:
                return upstream.generate_descriptions_with_vlm_cot(
                    *arguments, history, layout['layout_count'], model, pid)
            return upstream.generate_descriptions_with_vlm(
                *arguments, layout['layout_count'], model, pid)
        return retry_valid(call, validate_descriptions, retries, trace['calls'], 'description')

    def ground(layout, descriptions):
        width, height = layout['key_frame'].size
        boxes, points, accepted = [], [], []
        for description in descriptions:
            no_candidates = 0
            for attempt in range(1, retries + 1):
                raw_box = upstream.grounding_with_vlm(
                    observe('grounding', [layout['key_frame_idx']], layout['key_frame']),
                    query, description, model)
                event = {'stage': 'grounding', 'description': description, 'attempt': attempt,
                         'raw_box': raw_box}
                trace['calls'].append(event)
                if raw_box is None:
                    no_candidates += 1
                    event['status'] = 'no_candidate'
                    continue
                if (len(raw_box) != 4 or any(not math.isfinite(x) for x in raw_box)):
                    event['status'] = 'invalid_box'
                    continue
                # Ovis also encodes "no target" as a zero box. Treat it like
                # None, so mixed empty encodings reach the existing reflection
                # path rather than being mistaken for a malformed localization.
                if all(x == 0 for x in raw_box):
                    no_candidates += 1
                    event['status'] = 'no_candidate'
                    continue
                box = upstream.convert_bbox(raw_box, height, width)
                if box[0] >= box[2] or box[1] >= box[3]:
                    event['status'] = 'invalid_box'
                    continue
                labeled = upstream.visualize_box(np.asarray(layout['key_frame']), box)
                observe('point', [layout['key_frame_idx']], layout['key_frame'])
                raw_point = upstream.refine_grounding_with_vlm(labeled, query, description, model)
                event['raw_point'] = raw_point
                if (raw_point is None or len(raw_point) != 2
                        or any(not math.isfinite(x) for x in raw_point)):
                    event['status'] = 'invalid_point'
                    continue
                point = upstream.convert_point(raw_point, height, width)
                point = [min(width - 1, max(0, point[0])), min(height - 1, max(0, point[1]))]
                boxes.append(box)
                points.append(point)
                accepted.append(description)
                event.update(status='accepted', box=box, point=point)
                break
            else:
                if no_candidates != retries:
                    raise ValueError(f'Grounding exhausted {retries} calls without a valid box/point')
        return {'boxes': boxes, 'points': points, 'descriptions': accepted}

    def ask(call, questions, stage, guidance=False):
        return retry_valid(call, lambda result: require_answers(questions, result, guidance),
                           retries, trace['calls'], stage)

    def existence(state):
        questions = [{
            'question': f"Does the frame sequence contain targets matching query '{query}' but absent in the keyframe?",
            'options': {'A': 'Yes', 'B': 'No'}, 'correct_answer': 'B'}]
        # Check every predicted target, rather than only descriptions[0].
        for description in state['descriptions']:
            for text in (
                f"Does another sampled frame show '{description}' more clearly than the keyframe?",
                f"Does '{description}' suffer severe occlusion, blurring or overlap in the keyframe?",
            ):
                questions.append({'question': text, 'options': {'A': 'Yes', 'B': 'No'},
                                  'correct_answer': 'B'})
        layout = state['layout']
        result = ask(lambda: upstream.image_qa_with_vlm(
            [observe('existence_QA', layout['evidence_indices'], layout['strip'])],
            query, upstream.convert_qa_for_answer_prompt(questions),
            layout['key_evidence_position'] + 1, model,
            num_frames=len(layout['evidence_indices'])), questions, 'existence_QA', True)
        passed = answers_match(questions, result)
        return {'status': 'passed' if passed else 'failed', 'questions': questions,
                'answers': result['answers'], 'feedback': result['guidance'] or
                ' '.join(a['reason'] for a in result['answers'])}

    def consistency(state):
        if not qa:
            return {'status': 'not_checked_empty_qa', 'feedback': 'No applicable attribute questions.'}
        failures, details = 0, []
        for description, box in zip(state['descriptions'], state['boxes']):
            labeled = upstream.visualize_box(np.asarray(state['layout']['key_frame']), box)
            def answer():
                observe('consistency_QA', [state['layout']['key_frame_idx']],
                        state['layout']['key_frame'])
                return upstream.answer_qa_with_vlm(
                    labeled, upstream.convert_qa_for_answer_prompt(qa), query, model)
            result = ask(answer, qa, 'consistency_QA')
            matched = answers_match(qa, result)
            failures += not matched
            details.append({'description': description, 'box': box, 'answers': result['answers'],
                            'consistent': matched})
        failed = failures > .3 * len(state['boxes'])
        return {'status': 'failed' if failed else 'passed', 'inconsistent_targets': failures,
                'target_count': len(state['boxes']), 'threshold': .3, 'questions': qa,
                'targets': details, 'feedback': ' '.join(
                    a['reason'] for target in details if not target['consistent'] for a in target['answers'])}

    if operators_only:
        return select, describe, ground, existence, consistency
    return run_reflection(select, describe, ground, existence, consistency,
                          max_existence_checks, max_consistency_checks, trace['rounds'])
