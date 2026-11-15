"""Model-independent reasoning/reflection control for the ReasonVOS baseline.

Existence and consistency have independent check budgets, including initial checks.
The default 2+2 protocol is explicitly requested by the user.
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
                   max_existence_checks=3, max_consistency_checks=3, trace=None):
    """Independent stage caps, including the initial check; no unchecked update.

    Defaults allow at most two frame reselections and two consistency-driven
    description revisions (three checks per stage, including the initial one).
    Existence exhaustion cannot consume consistency opportunities. A later
    description revision invalidates an earlier existence pass unless it can
    be checked again with remaining existence budget.
    """
    budgets = {'existence': max_existence_checks, 'consistency': max_consistency_checks}
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 3 for v in budgets.values()):
        raise ValueError('Each stage check budget must be an integer in [0, 3]')
    used = {'existence': 0, 'consistency': 0}
    trace = trace if trace is not None else []
    layout, history, previous, guidance = None, [], [], None
    action, revision = 'select', 0
    while True:
        if action == 'select':
            layout = select(guidance, previous)
        descriptions = describe(layout, copy.deepcopy(history))
        validate_descriptions(descriptions)
        localization = ground(layout, descriptions)
        boxes, points = localization['boxes'], localization['points']
        if len(boxes) != len(points):
            raise ValueError('Grounding box/point count mismatch')
        state = {'layout': layout, **localization}
        complete = bool(boxes) and len(boxes) == len(descriptions) == len(localization['descriptions'])
        check1 = {'status': 'not_checked_disabled' if not budgets['existence']
                  else 'not_rechecked_budget_exhausted'}
        check2 = {'status': 'skipped'}
        if used['existence'] < budgets['existence']:
            used['existence'] += 1
            check1 = (existence(state) if complete else
                      {'status': 'failed', 'feedback': 'Not all described targets could be localized.'})
            if check1['status'] not in ('passed', 'failed'):
                raise ValueError('Unexpected existence verification status')
        # Allow frame repair only if the repaired state can still be checked.
        retry_selection = (check1['status'] == 'failed' and
                           used['existence'] < budgets['existence'])
        if not retry_selection:
            if not budgets['consistency']:
                check2 = {'status': 'not_checked_disabled'}
            elif not boxes or len(localization['descriptions']) != len(boxes):
                check2 = {'status': 'not_checked_no_targets'}
            elif used['consistency'] < budgets['consistency']:
                check2 = consistency(state)
                if check2['status'] not in ('passed', 'failed', 'not_checked_empty_qa'):
                    raise ValueError('Unexpected consistency verification status')
                if check2['status'] in ('passed', 'failed'):
                    used['consistency'] += 1
        verified = check1['status'] == check2['status'] == 'passed'
        retry_description = (not retry_selection and check2['status'] == 'failed'
                             and used['consistency'] < budgets['consistency'])
        terminal = not retry_selection and not retry_description
        status = ('verified' if verified else 'in_progress')
        if terminal and not verified:
            if not any(budgets.values()):
                status = 'disabled'
            elif check2['status'] == 'not_checked_empty_qa':
                status = 'not_checked_empty_qa'
            elif check2['status'] == 'not_checked_no_targets':
                status = 'not_checked_no_targets'
            else:
                status = 'budget_exhausted'
        state['reflection'] = {
            'revisions': revision, 'budgets': dict(budgets), 'checks_used': dict(used),
            'existence': check1['status'], 'consistency': check2['status'],
            'existence_checked_on_final_state': check1['status'] in ('passed', 'failed'),
            'verified': verified, 'status': status,
        }
        trace.append({'revision': revision, 'action': action,
                      'key_frame_idx': layout['key_frame_idx'],
                      'descriptions': copy.deepcopy(localization['descriptions']),
                      'boxes': copy.deepcopy(boxes), 'points': copy.deepcopy(points),
                      'existence': copy.deepcopy(check1), 'consistency': copy.deepcopy(check2),
                      'checks_used': dict(used), 'budgets': dict(budgets)})
        if terminal:
            return state
        failed = check1 if retry_selection else check2
        guidance = failed.get('feedback', '')
        previous = list(localization['descriptions'])
        history.append({'descriptions': previous, 'feedback': guidance})
        action = 'select' if retry_selection else 'describe'
        revision += 1
