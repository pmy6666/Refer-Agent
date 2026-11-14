"""Model-independent reasoning/reflection control for the ReasonVOS baseline.

MAX_TURN is interpreted as feedback-driven revisions after the initial prediction.
Every revised state is checked again, including the final budgeted prediction.
The paper does not publish executable pseudocode defining this counter.
"""
import copy
import json
from contextvars import ContextVar


MODEL_TRACE = ContextVar('reasonvos_model_trace', default=None)


def record_model_response(stage, prompt, response):
    events = MODEL_TRACE.get()
    if events is not None:
        events.append({'stage': stage, 'prompt': prompt, 'response': response})


def parse_answer_response(text, require_guidance=False):
    """Parse complete JSON objects, including fenced JSON, without regex nesting bugs."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            result, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict) or not isinstance(result.get('answers'), list):
            continue
        if not result['answers']:
            continue
        if require_guidance and not isinstance(result.get('guidance'), str):
            continue
        normalized = copy.deepcopy(result)
        valid = True
        for answer in normalized['answers']:
            if not isinstance(answer, dict) or not isinstance(answer.get('reason'), str):
                valid = False
                break
            option = answer.get('answer')
            if isinstance(option, str):
                option = [option.strip()]
            if (not isinstance(option, list) or len(option) != 1
                    or option[0] not in ('A', 'B', 'C', 'D')
                    or not answer['reason'].strip()):
                valid = False
                break
            answer['answer'] = option
        if valid:
            return normalized
    return None


def require_answers(questions, result, require_guidance=False):
    """Missing, extra, ambiguous or illegal answers are failures to answer, not passes."""
    if not questions:
        raise ValueError('Empty QA must be recorded as not checked')
    if not isinstance(result, dict) or not isinstance(result.get('answers'), list):
        raise ValueError('Expected an answers list')
    answers = result['answers']
    if len(answers) != len(questions):
        raise ValueError(f'QA coverage {len(answers)}/{len(questions)}')
    if require_guidance and not isinstance(result.get('guidance'), str):
        raise ValueError('Missing frame-selection guidance')
    for question, answer in zip(questions, answers):
        if not isinstance(answer, dict):
            raise ValueError('Answer must be an object')
        option = answer.get('answer')
        option = [option] if isinstance(option, str) else option
        if (not isinstance(option, list) or len(option) != 1
                or not isinstance(option[0], str) or option[0] not in question['options']
                or not isinstance(answer.get('reason'), str) or not answer['reason'].strip()):
            raise ValueError('Invalid option or missing visual explanation')
    return result


def answers_match(questions, result):
    require_answers(questions, result)
    return all(q['correct_answer'] == (a['answer'] if isinstance(a['answer'], str)
                                       else a['answer'][0])
               for q, a in zip(questions, result['answers']))


def retry_valid(call, validate, retries, trace, stage):
    if retries < 1:
        raise ValueError('Call retry budget must be positive')
    errors = []
    for attempt in range(1, retries + 1):
        try:
            value = call()
            validate(value)
        except (ValueError, TypeError) as error:
            errors.append(str(error))
            trace.append({'stage': stage, 'attempt': attempt, 'status': 'invalid',
                          'error': str(error)})
            continue
        trace.append({'stage': stage, 'attempt': attempt, 'status': 'accepted'})
        return value
    raise ValueError(f'{stage}: exhausted {retries} calls: {errors}')


def validate_descriptions(value):
    if (not isinstance(value, list) or not value
            or any(not isinstance(text, str) or not text.strip() for text in value)):
        raise ValueError('Expected nonempty target descriptions')


def validate_scores(value, count):
    import math
    if (not isinstance(value, list) or len(value) != count
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or not 1 <= x <= 10 for x in value)):
        raise ValueError(f'Expected {count} finite frame scores in [1, 10]')


def run_reflection(select, describe, ground, existence, consistency,
                   max_reflections=4, trace=None):
    """One shared budget; a state is a layout + descriptions + matching boxes/points.

    Existence failure revisits selection; consistency failure revisits intent.
    Both verifiers run on the new state. Valid format retries are separate from
    reflection revisions. Exhaustion exports an explicitly failed final state.
    """
    if isinstance(max_reflections, bool) or max_reflections < 0:
        raise ValueError('Reflection budget must be nonnegative')
    trace = trace if trace is not None else []
    layout, previous, history, guidance = None, [], [], None
    action = 'select'
    for revision in range(max_reflections + 1):
        if action == 'select':
            layout = select(guidance, previous)
        descriptions = describe(layout, copy.deepcopy(history))
        validate_descriptions(descriptions)
        localization = ground(layout, descriptions)
        boxes, points = localization['boxes'], localization['points']
        if len(boxes) != len(points):
            raise ValueError('Grounding box/point count mismatch')
        state = {'layout': layout, **localization}
        if max_reflections == 0:
            state['reflection'] = {'revisions': 0, 'max_reflections': 0,
                                   'existence': 'not_checked_disabled',
                                   'consistency': 'not_checked_disabled',
                                   'verified': False, 'status': 'disabled'}
            trace.append({'revision': 0, 'action': action, 'reflection': state['reflection']})
            return state
        if (not boxes or len(boxes) != len(descriptions)
                or len(localization['descriptions']) != len(descriptions)):
            check1 = {'status': 'failed', 'feedback': 'Not all described targets could be localized.'}
        else:
            check1 = existence(state)
        check2 = consistency(state) if check1['status'] == 'passed' else {'status': 'skipped'}
        if check1['status'] not in ('passed', 'failed') or check2['status'] not in (
                'passed', 'failed', 'skipped', 'not_checked_empty_qa'):
            raise ValueError('Unexpected verification status')
        event = {'revision': revision, 'action': action,
                 'key_frame_idx': layout['key_frame_idx'],
                 'descriptions': copy.deepcopy(localization['descriptions']),
                 'boxes': copy.deepcopy(boxes), 'points': copy.deepcopy(points),
                 'existence': copy.deepcopy(check1), 'consistency': copy.deepcopy(check2)}
        trace.append(event)
        state['reflection'] = {'revisions': revision, 'max_reflections': max_reflections,
                               'existence': check1['status'], 'consistency': check2['status'],
                               'verified': check1['status'] == check2['status'] == 'passed'}
        if state['reflection']['verified']:
            state['reflection']['status'] = 'verified'
            return state
        if check1['status'] == 'passed' and check2['status'] == 'not_checked_empty_qa':
            state['reflection']['status'] = 'not_checked_empty_qa'
            return state
        if revision == max_reflections:
            state['reflection']['status'] = 'budget_exhausted'
            return state
        failed = check1 if check1['status'] == 'failed' else check2
        guidance = failed.get('feedback', '')
        previous = list(localization['descriptions'])
        history.append({'descriptions': previous, 'feedback': guidance})
        action = 'select' if check1['status'] == 'failed' else 'describe'
