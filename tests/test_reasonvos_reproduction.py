"""CPU regression tests for real reasoning paths, coverage and cache provenance."""
import ast
from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
from queue import Queue
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.reasonvos_reflection import (
    run_reflection, require_answers, parse_answer_response, retry_valid,
)
from utils.reasonvos_pipeline import infer_expression
from utils.reasonvos_runtime import ensure_identity, atomic_json
from utils.reorder_reasonvos_meta import complete_frame_metadata


def actual_functions(path, names, globals_):
    tree = ast.parse((ROOT / path).read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith(('_origin', '_cot')):
                    globals_[target.id] = ast.literal_eval(node.value)
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), globals_)
    return globals_


class ReflectionTests(unittest.TestCase):
    def callbacks(self, existence, consistency):
        self.select = Mock(return_value={'key_frame_idx': 0})
        self.describe = Mock(return_value=['target'])
        self.ground = Mock(return_value={'boxes': [[0, 0, 10, 10]], 'points': [[5, 5]],
                                        'descriptions': ['target']})
        return self.select, self.describe, self.ground, Mock(side_effect=existence), Mock(side_effect=consistency)

    def test_success_preserves_single_grounding(self):
        callbacks = self.callbacks([{'status': 'passed'}], [{'status': 'passed'}])
        state = run_reflection(*callbacks)
        self.assertTrue(state['reflection']['verified'])
        self.assertEqual(self.ground.call_count, 1)

    def test_consistency_revision_rechecks_existence_with_its_own_budget(self):
        callbacks = self.callbacks([{'status': 'passed'}] * 2,
            [{'status': 'failed', 'feedback': 'wrong target'}, {'status': 'passed'}])
        trace = []
        state = run_reflection(*callbacks, trace=trace)
        self.assertTrue(state['reflection']['verified'])
        self.assertEqual(state['reflection']['checks_used'], {'existence': 2, 'consistency': 2})
        self.assertEqual([e['action'] for e in trace], ['select', 'describe'])
        self.assertEqual(self.select.call_count, 1)
        self.assertEqual(self.describe.call_args.args[1][-1]['feedback'], 'wrong target')

    def test_existence_exhaustion_preserves_both_consistency_opportunities(self):
        callbacks = self.callbacks([{'status': 'failed', 'feedback': 'occluded'}] * 2,
                                  [{'status': 'failed', 'feedback': 'wrong'}] * 2)
        trace = []
        state = run_reflection(*callbacks, max_existence_checks=2, max_consistency_checks=2, trace=trace)
        self.assertEqual(callbacks[3].call_count, 2)
        self.assertEqual(callbacks[4].call_count, 2)
        self.assertEqual(self.select.call_count, 2)
        self.assertEqual(self.ground.call_count, 3)
        self.assertEqual([e['action'] for e in trace], ['select', 'select', 'describe'])
        self.assertEqual(state['reflection']['checks_used'], {'existence': 2, 'consistency': 2})
        self.assertEqual(state['reflection']['status'], 'budget_exhausted')
        self.assertEqual(state['reflection']['existence'], 'not_rechecked_budget_exhausted')
        self.assertFalse(state['reflection']['existence_checked_on_final_state'])
        self.assertFalse(state['reflection']['verified'])
        self.assertEqual(state['boxes'], trace[-1]['boxes'])

    def test_final_consistency_pass_does_not_reuse_stale_existence_pass(self):
        callbacks = self.callbacks([{'status': 'passed'}],
                                  [{'status': 'failed', 'feedback': 'wrong'}, {'status': 'passed'}])
        state = run_reflection(*callbacks, max_existence_checks=1)
        self.assertEqual(callbacks[3].call_count, 1)
        self.assertEqual(callbacks[4].call_count, 2)
        self.assertEqual(state['reflection']['consistency'], 'passed')
        self.assertFalse(state['reflection']['verified'])
        self.assertEqual(state['reflection']['existence'], 'not_rechecked_budget_exhausted')

    def test_defaults_allow_two_reselections_and_two_description_revisions(self):
        callbacks = self.callbacks([{'status': 'failed', 'feedback': 'occluded'}] * 3,
                                  [{'status': 'failed', 'feedback': 'wrong target'}] * 3)
        trace = []
        state = run_reflection(*callbacks, trace=trace)
        self.assertEqual([r['action'] for r in trace],
                         ['select', 'select', 'select', 'describe', 'describe'])
        self.assertEqual(self.select.call_count - 1, 2)
        self.assertEqual(callbacks[3].call_count, 3)
        self.assertEqual(callbacks[4].call_count, 3)
        self.assertEqual(self.ground.call_count, 5)
        self.assertEqual(state['reflection']['revisions'], 4)
        self.assertEqual(state['reflection']['checks_used'], {'existence': 3, 'consistency': 3})
        self.assertEqual(state['reflection']['status'], 'budget_exhausted')
        self.assertFalse(state['reflection']['verified'])

    def test_second_description_revision_is_checked_and_can_pass(self):
        callbacks = self.callbacks([{'status': 'passed'}] * 3,
                                  [{'status': 'failed'}] * 2 + [{'status': 'passed'}])
        trace = []
        state = run_reflection(*callbacks, trace=trace)
        self.assertEqual([r['action'] for r in trace], ['select', 'describe', 'describe'])
        self.assertTrue(state['reflection']['verified'])
        self.assertEqual(self.select.call_count, 1)
        self.assertEqual(state['reflection']['checks_used'], {'existence': 3, 'consistency': 3})

    def test_stage_budgets_cannot_exceed_three(self):
        for value in (4, -1, True):
            with self.assertRaises(ValueError):
                run_reflection(*self.callbacks([], []), max_existence_checks=value)
            with self.assertRaises(ValueError):
                run_reflection(*self.callbacks([], []), max_consistency_checks=value)

    def test_empty_qa_and_disabled_are_never_marked_verified(self):
        callbacks = self.callbacks([{'status': 'passed'}], [{'status': 'not_checked_empty_qa'}])
        state = run_reflection(*callbacks)
        self.assertEqual(state['reflection']['status'], 'not_checked_empty_qa')
        self.assertEqual(state['reflection']['checks_used']['consistency'], 0)
        self.assertFalse(state['reflection']['verified'])
        callbacks = self.callbacks([], [])
        state = run_reflection(*callbacks, max_existence_checks=0, max_consistency_checks=0)
        self.assertEqual(state['reflection']['status'], 'disabled')
        callbacks[3].assert_not_called()
        callbacks[4].assert_not_called()

    def test_answers_cannot_pass_with_missing_or_extra_or_illegal_choices(self):
        questions = [{'options': {'A': 'yes', 'B': 'no'}, 'correct_answer': 'A'}] * 2
        for answers in [[], [{'answer': ['A'], 'reason': 'seen'}],
                        [{'answer': ['A', 'B'], 'reason': 'seen'}] * 2,
                        [{'answer': ['C'], 'reason': 'seen'}] * 2,
                        [{'answer': ['A'], 'reason': ''}] * 2,
                        [{'answer': ['A'], 'reason': 'seen'}] * 3]:
            with self.assertRaises(ValueError):
                require_answers(questions, {'answers': answers})
        self.assertIsNone(parse_answer_response('{"answers":[]}'))

    def test_valid_retry_returns_immediately(self):
        call = Mock(side_effect=[[], ['valid'], ['must not overwrite']])
        def valid(value):
            if not value:
                raise ValueError('empty')
        self.assertEqual(retry_valid(call, valid, 3, [], 'describe'), ['valid'])
        self.assertEqual(call.call_count, 2)

    def test_actual_feedback_description_function_logs_complete_prompt(self):
        class Tensor:
            def cuda(self): return self
        model = Mock()
        model.preprocess_inputs.return_value = (Tensor(), None, None)
        model.generate.return_value = [[0]]
        model.text_tokenizer.decode.return_value = '{"description":"correct target"}'
        record = Mock()
        env = actual_functions('eval/Ovis_infer_reasonvos.py', {'generate_descriptions_with_vlm_cot'},
                               {'torch': types.SimpleNamespace(no_grad=nullcontext),
                                'json': json, 'record_model_response': record})
        result = env['generate_descriptions_with_vlm_cot'](
            Image.new('RGB', (64, 64)), 'query', 3,
            [{'descriptions': ['old target'], 'feedback': 'wrong causal role'}], 5, model, 0)
        self.assertEqual(result, ['correct target'])
        self.assertIn('wrong causal role', record.call_args.args[1])
        self.assertIn('old target', record.call_args.args[1])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.frames = [f'{i:05}' for i in range(10)]
        self.images = [Image.new('RGB', (96, 96)) for _ in self.frames]
        self.scores = {f + '.jpg': {'clip': 5.0} for f in self.frames}
        self.upstream = Mock()
        self.upstream.score_frame_with_vlm.return_value = list(range(1, 11))
        self.upstream.generate_descriptions_with_vlm.return_value = ['target']
        self.upstream.generate_descriptions_with_vlm_cot.return_value = ['corrected target']
        self.upstream.grounding_with_vlm.return_value = [.1, .1, .8, .8]
        self.upstream.convert_bbox.side_effect = lambda b,h,w: [int(x*w) for x in b]
        self.upstream.refine_grounding_with_vlm.return_value = [.4, .4]
        self.upstream.convert_point.side_effect = lambda p,h,w: [int(x*w) for x in p]
        self.upstream.convert_qa_for_answer_prompt.side_effect = lambda q: q
        self.upstream.image_qa_with_vlm.side_effect = lambda frames,q,questions,*a,**k: {
            'answers': [{'answer': ['B'], 'reason': 'visible'} for _ in questions], 'guidance': ''}
        self.upstream.answer_qa_with_vlm.return_value = {'answers': [{'answer': ['A'], 'reason': 'seen'}]}
        self.scorer = Mock()
        self.scorer.calculate_clip_score.return_value = 5.0
        self.scorer.normalize_scores.side_effect = lambda values: values
        self.qa = [{'question': 'Is the target red?', 'options': {'A':'Yes','B':'No'}, 'correct_answer':'A'}]

    def infer(self, **kwargs):
        trace = {'calls': [], 'selection': [], 'rounds': []}
        state = infer_expression(self.upstream, object(), self.scorer, self.images, self.frames,
                                 '/rgb', 'query', self.scores, self.qa, trace=trace, **kwargs)
        return state, trace

    def test_short_video_uses_actual_score_count(self):
        self.frames = self.frames[:8]
        self.images = self.images[:8]
        self.upstream.score_frame_with_vlm.return_value = list(range(1, 9))
        state, _ = self.infer()
        self.assertEqual(self.upstream.score_frame_with_vlm.call_args.kwargs['num_frames'], 8)
        self.assertEqual(len(state['layout']['evidence_indices']), 5)

    def test_all_five_dynamic_focus_anchor_positions(self):
        for anchor in range(5):
            values = [9 if i % 2 == 0 else 1 for i in range(10)]
            values[2 * anchor] = 10
            self.upstream.score_frame_with_vlm.return_value = values
            state, _ = self.infer()
            layout = state['layout']
            self.assertEqual(layout['key_frame_idx'], anchor * 2)
            self.assertEqual(layout['layout_count'], 7 if anchor % 2 else 5)
            self.assertEqual(layout['layout_position'], anchor + (anchor % 2))

    def test_existence_feedback_rescores_clip_and_rechecks_corrected_state(self):
        original = self.upstream.image_qa_with_vlm.side_effect
        calls = []
        def check(*args, **kwargs):
            response = original(*args, **kwargs)
            if not calls:
                response['answers'][0]['answer'] = ['A']
                response['guidance'] = 'choose visible target'
            calls.append(response)
            return response
        self.upstream.image_qa_with_vlm.side_effect = check
        state, trace = self.infer()
        self.assertTrue(state['reflection']['verified'])
        self.assertTrue(trace['selection'][1]['description_guided_clip'])
        self.assertEqual(trace['selection'][1]['guidance'], 'choose visible target')
        self.assertEqual(self.scorer.calculate_clip_score.call_count, 10)
        self.assertEqual(self.upstream.grounding_with_vlm.call_count, 2)

    def test_consistency_threshold_counts_targets(self):
        self.upstream.generate_descriptions_with_vlm.return_value = ['a','b','c','d']
        for bad, expected in [(1, True), (2, False)]:
            self.upstream.answer_qa_with_vlm.side_effect = [
                {'answers':[{'answer':['B' if i < bad else 'A'], 'reason':'seen'}]} for i in range(20)]
            state, trace = self.infer(max_existence_checks=1, max_consistency_checks=1)
            self.assertEqual(trace['rounds'][0]['consistency']['status'] == 'passed', expected)
            self.assertEqual(trace['rounds'][0]['consistency']['target_count'], 4)

    def test_mixed_empty_boxes_in_revision_continue_existing_reflection(self):
        # Replay the observed sequence: a valid initial result, a failed
        # existence check, then zero box / None / zero box on the new frame.
        original = self.upstream.image_qa_with_vlm.side_effect
        checks = []
        def check(*args, **kwargs):
            result = original(*args, **kwargs)
            if not checks:
                result['answers'][1]['answer'] = ['A']
                result['guidance'] = 'choose clearer frame'
            checks.append(result)
            return result
        self.upstream.image_qa_with_vlm.side_effect = check
        valid = [.1, .1, .8, .8]
        self.upstream.grounding_with_vlm.side_effect = [
            valid, [0., 0., 0., 0.], None, [0., 0., 0., 0.], valid]
        state, trace = self.infer(max_existence_checks=2, max_consistency_checks=2)
        self.assertFalse(state['reflection']['verified'])
        self.assertEqual(state['reflection']['status'], 'not_checked_no_targets')
        self.assertEqual(state['reflection']['revisions'], 1)
        self.assertEqual([r['action'] for r in trace['rounds']], ['select'] * 2)
        self.assertEqual(trace['rounds'][1]['boxes'], [])
        self.assertEqual(trace['rounds'][1]['existence']['status'], 'failed')
        self.assertEqual(trace['rounds'][1]['consistency']['status'], 'not_checked_no_targets')
        calls = [c for c in trace['calls'] if c['stage'] == 'grounding']
        self.assertEqual([c['status'] for c in calls],
                         ['accepted', 'no_candidate', 'no_candidate', 'no_candidate'])
        self.assertEqual(calls[1]['raw_box'], [0., 0., 0., 0.])
        self.assertEqual(self.upstream.refine_grounding_with_vlm.call_count, 1)

    def test_empty_boxes_exhaust_existence_without_becoming_verified(self):
        self.upstream.grounding_with_vlm.side_effect = [[0., 0., 0., 0.], None, [0., 0., 0., 0.]] * 5
        state, trace = self.infer()
        self.assertEqual(state['reflection']['status'], 'not_checked_no_targets')
        self.assertEqual(state['reflection']['revisions'], 2)
        self.assertFalse(state['reflection']['verified'])
        self.assertEqual(state['boxes'], [])
        self.assertEqual(len(trace['rounds']), 3)
        self.assertEqual(self.upstream.grounding_with_vlm.call_count, 9)
        self.upstream.image_qa_with_vlm.assert_not_called()
        self.upstream.answer_qa_with_vlm.assert_not_called()

    def test_nonzero_degenerate_box_and_invalid_point_remain_errors(self):
        for raw_box, raw_point in [([.2, .1, .2, .8], [.4, .4]),
                                   ([float('nan'), .1, .8, .8], [.4, .4]),
                                   ([.1, .1, .8, .8], None)]:
            with self.subTest(raw_box=raw_box, raw_point=raw_point):
                self.upstream.grounding_with_vlm.return_value = raw_box
                self.upstream.refine_grounding_with_vlm.return_value = raw_point
                with self.assertRaisesRegex(ValueError, 'Grounding exhausted 3 calls'):
                    self.infer()

    def test_incomplete_answers_exhaust_retries_instead_of_passing(self):
        self.upstream.answer_qa_with_vlm.return_value = {'answers': []}
        with self.assertRaisesRegex(ValueError, 'consistency_QA: exhausted'):
            self.infer()
        self.assertEqual(self.upstream.answer_qa_with_vlm.call_count, 3)


class CoverageAndProvenanceTests(unittest.TestCase):
    def test_complete_frames_keeps_ids_and_uses_only_annotation_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'rgb/v').mkdir(parents=True); (root/'gt/s_v_o').mkdir(parents=True)
            for f in ['001','002','003']:
                (root/f'rgb/v/{f}.jpg').touch()
            for f in ['001','002']:
                (root/f'gt/s_v_o/{f}.png').touch()
            meta = {'videos': {'v': {'source':'s', 'frames':['001'], 'expressions':[{'obj_id':'o','exp_id':7}]}}}
            before = copy.deepcopy(meta)
            result, changes = complete_frame_metadata(meta, root/'rgb', root/'gt')
            self.assertEqual(meta, before)
            self.assertEqual(result['videos']['v']['frames'], ['001','002'])
            self.assertEqual(result['videos']['v']['expressions'][0]['exp_id'], 7)
            self.assertEqual(changes[0]['added_frames'], ['002'])

    def test_cache_rejects_changed_identity_and_unversioned_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); existing=root/'mask.png';existing.touch()
            with self.assertRaisesRegex(ValueError, 'Unversioned'):
                ensure_identity(root/'identity.json', {'prompt':'new'}, [existing])
            ensure_identity(root/'identity.json', {'prompt':'old'})
            ensure_identity(root/'identity.json', {'prompt':'old'})
            with self.assertRaisesRegex(ValueError, 'changed'):
                ensure_identity(root/'identity.json', {'prompt':'new'})

    def test_evaluator_rejects_missing_mask_and_scores_complete_identical_masks(self):
        import cv2
        from eval.eval_reasonvos import eval_queue
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);gt=root/'gt/s_v_o';pred=root/'pred/v/0';gt.mkdir(parents=True);pred.mkdir(parents=True)
            mask=np.zeros((32,32),np.uint8);mask[8:24,8:24]=255
            cv2.imwrite(str(gt/'001.png'), mask)
            queue=Queue();queue.put(('s_v_o','v','0'))
            with self.assertRaises(FileNotFoundError):
                eval_queue(queue,0,{},str(root/'gt'),str(root/'pred'),{})
            cv2.imwrite(str(pred/'001.png'), mask)
            queue=Queue();queue.put(('s_v_o','v','0'));scores={}
            eval_queue(queue,0,scores,str(root/'gt'),str(root/'pred'),{})
            self.assertEqual(scores['s_v_o_0'], [1.,1.])

    def test_sam2_identity_rejects_changed_grounding(self):
        from eval.post_processing import check_segmentation_identity
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'all_results.json';atomic_json([],path)
            # Weight hashing is mocked; prompt, config and source hashes are real.
            from utils.reasonvos_runtime import file_sha256
            with patch('eval.post_processing.file_sha256', side_effect=lambda p:
                       'weight-hash' if str(p).endswith('.pt') else file_sha256(p)):
                check_segmentation_identity(path, [])
                atomic_json([{'changed':'prompt'}], path)
                with self.assertRaisesRegex(ValueError, 'changed'):
                    check_segmentation_identity(path, [])


if __name__ == '__main__':
    unittest.main()
