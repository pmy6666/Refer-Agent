#!/usr/bin/env python3
"""Read-only ReasonVOS progress report, with no model or third-party dependencies."""
import argparse
from collections import Counter
from datetime import datetime
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.reasonvos_runtime import RUN_ROOT, DATA_ROOT

TZ = ZoneInfo('Asia/Shanghai')
STAGES = {'01_clip': 'CLIP评分', '02_query_qa': 'QA预处理',
          '03_grounding': '选帧、定位与反思', '04_sam2': 'SAM2分割', '05_evaluation': '评测'}
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def read_json(path, warnings):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        warnings.append(f'暂时无法读取 {path.name}: {exc}')
        return None


def tail_lines(path, count=3):
    try:
        with path.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 65536))
            text = stream.read().decode('utf-8', errors='replace')
    except OSError:
        return []
    lines = [ANSI.sub('', line).strip() for line in re.split(r'[\r\n]+', text)]
    return [line for line in lines if line][-count:]


def slurm_status(run, submission, logs):
    ids = {str(submission['job_id'])} if submission.get('job_id') else set()
    for path in logs:
        match = re.fullmatch(r'slurm-(\d+)\.log', path.name)
        if match:
            ids.add(match[1])
        for line in tail_lines(path, 20):
            match = re.search(r'Time-limit continuation submitted:\s*(\d+)', line)
            if match:
                ids.add(match[1])
    try:
        result = subprocess.run(['squeue', '--noheader', '--user', getpass.getuser(),
                                 '--format=%i|%T|%M|%j|%R'],
                                capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return {'available': False, 'jobs': [], 'error': str(exc)}
    jobs = []
    for line in result.stdout.splitlines():
        parts = line.strip().split('|', 4)
        if len(parts) != 5:
            continue
        job_id, state, elapsed, name, reason = (part.strip() for part in parts)
        matched = job_id.split('_')[0] in ids
        if not matched:
            # A custom log path may not have created its first log yet.
            try:
                detail = subprocess.run(['scontrol', 'show', 'job', job_id, '-o'],
                                        capture_output=True, text=True, timeout=5, check=True).stdout
                matched = any(value.startswith(str(run) + '/') for value in
                              re.findall(r'(?:StdOut|StdErr|WorkDir)=(\S+)', detail))
                if not matched:
                    # Continuations use the shared repository log directory and
                    # get new job IDs. Match their explicit run argument too.
                    command = re.search(r'\bCommand=(.*?)\s+WorkDir=', detail)
                    working = re.search(r'\bWorkDir=(\S+)', detail)
                    if command and working:
                        argv = shlex.split(command.group(1))
                        for index, argument in enumerate(argv):
                            value = None
                            if argument == '--run_dir' and index + 1 < len(argv):
                                value = argv[index + 1]
                            elif argument.startswith('--run_dir='):
                                value = argument.split('=', 1)[1]
                            if value is not None:
                                directory = Path(value)
                                if not directory.is_absolute():
                                    directory = Path(working.group(1)) / directory
                                matched = directory.resolve() == run.resolve()
                                break
            except (OSError, subprocess.SubprocessError):
                pass
        if matched:
            jobs.append({'job_id': job_id, 'state': state, 'elapsed': elapsed,
                         'name': name, 'node_or_reason': reason})
    return {'available': True, 'jobs': jobs}


def collect_progress(run, log_count=3):
    run = run.resolve()
    warnings = []
    tracked = []

    def load(relative):
        path = run / relative
        tracked.append(path)
        value = read_json(path, warnings)
        return value if isinstance(value, dict) else {}

    manifest = load('manifest.json')
    submission = load('submission.json')
    status = load('status.json')
    meta = load('meta_expressions.json').get('videos', {})
    smoke_status = load('smoke/status.json')
    summary = load('predictions/summary.json')
    reflection = load('reflection_summary.json')
    coverage = load('coverage.json')
    clips = load('video_scores_reasonvos.json')
    merged_qa = load('query_qa/reasonvos_query_qa.json')
    data_root = Path(manifest.get('data_root', DATA_ROOT))
    expected, clip_done, qa_done, grounding_done = 0, 0, 0, 0
    mask_done, mask_expected, mask_complete_expressions = 0, 0, 0
    gt_cache = {}
    gt_missing = []
    reflection_counts = Counter()
    for video, info in meta.items():
        frames = info.get('frames', [])
        for expression in info.get('expressions', []):
            expected += 1
            exp_id = str(expression['exp_id'])
            scores = clips.get(video, {}).get('CLIP', {}).get(exp_id, {})
            if scores and all(frame + '.jpg' in scores for frame in frames):
                clip_done += 1
            # Address the primary checkpoint by metadata ID: *.json would also
            # count 0.json.trace.json, incorrectly doubling reported QA progress.
            qa_path = run / 'query_qa/checkpoints' / video / (exp_id + '.json')
            qa = read_json(qa_path, warnings)
            tracked.append(qa_path)
            if not isinstance(qa, dict):
                qa = merged_qa.get(f'{video}/{exp_id}')
            if (isinstance(qa, dict) and qa.get('exp', '').strip() == expression['exp_text'].strip()
                    and isinstance(qa.get('qa_complete'), list)):
                qa_done += 1
            prediction_dir = run / 'predictions/Annotations' / video / exp_id
            grounding_path = prediction_dir / 'grounding.json'
            grounding = read_json(grounding_path, warnings)
            tracked.append(grounding_path)
            if (isinstance(grounding, dict) and str(grounding.get('exp_id')) == exp_id
                    and grounding.get('video_name') == video and grounding.get('frames') == frames):
                grounding_done += 1
                reflection_counts[grounding.get('reflection', {}).get('status', 'not_recorded')] += 1
            gt_dir = data_root / 'Annotations' / f"{info['source']}_{video}_{expression['obj_id']}"
            if gt_dir not in gt_cache:
                gt_cache[gt_dir] = [path.name for path in gt_dir.glob('*.png')]
                if not gt_cache[gt_dir]:
                    gt_missing.append(str(gt_dir))
            names = gt_cache[gt_dir]
            mask_expected += len(names)
            present = 0
            for name in names:
                path = prediction_dir / name
                try:
                    if path.is_file() and path.stat().st_size > 0:
                        present += 1
                        tracked.append(path)
                except OSError:
                    pass
            mask_done += present
            mask_complete_expressions += bool(names) and present == len(names)
    if gt_missing:
        warnings.append(f'{len(gt_missing)}个GT目录缺失或为空，掩码总数不完整')
    if not meta:
        warnings.append('运行元数据尚未生成，暂时无法计算各阶段总进度')
    stage = status.get('stage', '')
    logs = sorted(run.glob('slurm-*.log'), key=lambda p: p.stat().st_mtime)
    scheduler = slurm_status(run, submission, logs)
    if status.get('state') == 'running' and scheduler['available'] and not scheduler['jobs']:
        warnings.append('状态文件记录运行中，但未找到对应活动作业；可能已退出或正在切换续跑作业')
    stage_log = run / (stage + '.log') if stage else None
    if stage_log is None or not stage_log.exists():
        stage_logs = [path for path in run.glob('0[1-5]_*.log') if path.is_file()]
        stage_log = max(stage_logs, key=lambda p: p.stat().st_mtime) if stage_logs else None
    tracked += logs
    if stage_log:
        tracked.append(stage_log)
    if (run / 'smoke').exists() and not status:
        warnings.append('全量状态尚未生成，请查看单样本检查状态')
    latest = None
    for path in tracked:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if latest is None or mtime > latest['timestamp']:
            latest = {'file': str(path.relative_to(run)), 'timestamp': mtime,
                      'time': datetime.fromtimestamp(mtime, TZ).isoformat(timespec='seconds')}
    return {'checked_at': datetime.now(TZ).isoformat(timespec='seconds'), 'run_dir': str(run),
            'scope': manifest.get('scope', 'full_run'),
            'source_total_expressions': manifest.get('source_total_expressions'),
            'max_reflections': manifest.get('parameters', {}).get('max_reflections', submission.get('max_reflections')),
            'stage_check_budgets': {k: manifest.get('parameters', {}).get('max_' + k + '_checks')
                                    for k in ('existence', 'consistency')},
            'status': status, 'scheduler': scheduler, 'videos': len(meta), 'expressions': expected,
            'stages': {'clip': {'done': clip_done, 'total': expected},
                       'qa': {'done': qa_done, 'total': expected},
                       'grounding': {'done': grounding_done, 'total': expected},
                       'sam2_expressions': {'done': mask_complete_expressions, 'total': expected},
                       'required_masks': {'done': mask_done, 'total': mask_expected,
                                          'denominator_complete': not gt_missing}},
            'summary': summary or None, 'smoke_status': smoke_status or None,
            'reflection': reflection or dict(reflection_counts), 'coverage': coverage or None,
            'latest_activity': latest, 'stage_log': str(stage_log) if stage_log else None,
            'log_tail': tail_lines(stage_log, log_count) if stage_log and log_count else [],
            'warnings': warnings}


def fraction(value):
    done, total = value['done'], value['total']
    if not total:
        return f'{done}/未知'
    # Rounding must not label an incomplete run as 100.0%.
    percent = min(99.9, 100 * done / total) if done < total else 100 * done / total
    return f'{done}/{total} ({percent:.1f}%)'


def metric_line(summary):
    try:
        return 'J={:.2f}  F={:.2f}  J&F={:.2f}'.format(
            summary['J'], summary['F'], summary['J&F'])
    except (KeyError, TypeError, ValueError):
        return '指标字段不完整'


def render(report):
    status = report['status']
    subset = report.get('scope') == 'completed_subset'
    score_label = '子集' if subset else '全量'
    lines = [f"查询时间：{report['checked_at']}（北京时间）", f"运行目录：{report['run_dir']}",
             f"检查轮数上限：{report.get('stage_check_budgets')}；旧版共享修正预算：{report['max_reflections']}；数据：{report['videos']}视频 / {report['expressions']}条表达式",
             f"文件状态：{status.get('state', '尚未生成')}；阶段：{STAGES.get(status.get('stage'), status.get('stage', '尚未生成'))}"]
    if subset:
        lines.append(f"评测范围：已完成样本子集 {report['expressions']}/{report.get('source_total_expressions')}，不能作为全量成绩")
    scheduler = report['scheduler']
    if not scheduler['available']:
        lines.append('Slurm：无法查询（' + scheduler['error'] + '）')
    elif not scheduler['jobs']:
        lines.append('Slurm：未找到此实验的活动作业')
    for job in scheduler['jobs']:
        lines.append(f"Slurm：{job['job_id']}  {job['state']}  本分片耗时 {job['elapsed']}  {job['node_or_reason']}")
    labels = {'clip': 'CLIP评分', 'qa': 'QA主检查点', 'grounding': '选帧/定位/反思',
              'sam2_expressions': 'SAM2完整表达式', 'required_masks': 'GT所需掩码文件'}
    for key, label in labels.items():
        lines.append(f'{label}：{fraction(report["stages"][key])}')
    lines.append('掩码统计按文件存在且非空计算，完整评测仍须检查图像内容。')
    if report['summary']:
        lines.append(score_label + '已有指标：' + metric_line(report['summary']))
        lines.append('评测口径：' + str(report['summary'].get('evaluation_policy', '未记录')))
        if status.get('state') != 'complete':
            lines.append('当前任务尚未标记完成，请核对已有指标是否属于最后一次运行。')
    else:
        lines.append(score_label + ' J/F/J&F：尚未生成')
    smoke = report['smoke_status']
    if smoke:
        text = f"单样本检查：{smoke.get('state', '未知')}"
        if smoke.get('summary'):
            text += '；' + metric_line(smoke['summary']) + '（仅单样本）'
        lines.append(text)
    if status.get('error'):
        lines.append('记录的错误：' + str(status['error']))
    if report['latest_activity']:
        activity = report['latest_activity']
        lines.append(f"最近写入：{activity['time']}  {activity['file']}")
    if report['reflection']:
        lines.append('反思统计：' + json.dumps(report['reflection'], ensure_ascii=False))
    if report['log_tail']:
        lines.append('最新阶段日志：')
        lines.extend('  ' + line for line in report['log_tail'])
    lines.extend('提示：' + warning for warning in report['warnings'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', '--run_dir', type=Path, default=RUN_ROOT, help='运行目录，默认查询当前6次预算实验')
    parser.add_argument('--json', action='store_true', help='输出JSON；持续模式下每行一个JSON对象')
    parser.add_argument('--watch', nargs='?', const=30., type=float, metavar='SECONDS', help='持续刷新，默认间隔30秒')
    parser.add_argument('--logs', type=int, default=3, help='显示最新阶段日志的末尾行数，0表示不显示')
    args = parser.parse_args()
    if args.watch is not None and args.watch < 1:
        parser.error('--watch must be at least 1 second')
    if args.logs < 0:
        parser.error('--logs must be nonnegative')
    run = args.run_dir.expanduser()
    if not run.is_dir() and not run.is_absolute() and (ROOT / run).is_dir():
        run = ROOT / run
    if not run.is_dir():
        parser.error(f'运行目录不存在: {run}')
    try:
        while True:
            report = collect_progress(run, args.logs)
            if args.json:
                print(json.dumps(report, ensure_ascii=False), flush=True)
            else:
                if args.watch and sys.stdout.isatty():
                    print('\033[2J\033[H', end='')
                print(render(report), flush=True)
            if args.watch is None:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
