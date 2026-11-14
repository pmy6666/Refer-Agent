###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import time
import argparse
import cv2
import json
import glob
import numpy as np
import pandas as pd
from metrics import db_eval_iou, db_eval_boundary
import multiprocessing as mp
from queue import Empty
from utils.reasonvos_runtime import atomic_json

NUM_WOEKERS = 32

def eval_queue(q, rank, out_dict, mask_path, pred_path, meta_exp, allow_missing_masks=False):
    while True:
        try:
            vid_key, vid_name, exp_id = q.get(timeout=2)
        except Empty:
            break
        exp_name = "{}_{}".format(vid_key, exp_id)

        mask_path_vid = os.path.join(mask_path, vid_key)
        pred_path_vid_exp = os.path.join(pred_path, vid_name, exp_id)

        if not os.path.exists(mask_path_vid):
            print(f'{mask_path_vid} not found, not take into metric computation')
            continue
        if not os.path.exists(pred_path_vid_exp):
            print(f'{pred_path_vid_exp} not found, not take into metric computation')
            continue

        gt_mask_list = [x for x in sorted(os.listdir(mask_path_vid)) if str(x).endswith('png')]
        gt_0_path = os.path.join(mask_path_vid, gt_mask_list[0])
        gt_0 = cv2.imread(gt_0_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt_0.shape

        vid_len = len(gt_mask_list)
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        for frame_idx, frame_name in enumerate(gt_mask_list):
            gt_masks[frame_idx] = cv2.imread(os.path.join(mask_path_vid, frame_name), cv2.IMREAD_GRAYSCALE)
            if os.path.exists(os.path.join(pred_path_vid_exp, frame_name)):
                prediction = cv2.imread(os.path.join(pred_path_vid_exp, frame_name), cv2.IMREAD_GRAYSCALE)
                if prediction is None or prediction.shape != (h, w):
                    raise ValueError(f'Unreadable or wrong-sized prediction: {pred_path_vid_exp}/{frame_name}')
                pred_masks[frame_idx] = prediction
            else:
                if not allow_missing_masks:
                    raise FileNotFoundError(f'Missing prediction: {pred_path_vid_exp}/{frame_name}')
                pred_masks[frame_idx] = np.zeros((h, w), dtype=np.uint8)

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[exp_name] = [j, f]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_exp_path", type=str)
    parser.add_argument("--mask_path", type=str)
    parser.add_argument("--pred_path", type=str)
    parser.add_argument("--save_name", type=str, default="reason_vos.json")
    parser.add_argument("--save_csv_name", type=str, default="reason_vos.csv")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--allow_missing_masks', action='store_true',
                        help='Diagnostic only: score missing frames as zero, not a complete reproduction')
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error('num_workers must be positive')

    args.save_name = os.path.join(args.pred_path, '..', args.save_name)
    args.save_csv_name = os.path.join(args.pred_path, '..', args.save_csv_name)

    queue = mp.Queue()
    meta_exp = json.load(open(args.meta_exp_path, 'r'))["videos"]
    output_dict = mp.Manager().dict()

    vid_info_map = {}
    for vid_name in meta_exp.keys():
        vid = meta_exp[vid_name]
        src_dataset = vid['source']
        for sample in vid['expressions']:
            obj_id = sample['obj_id']
            exp_id = sample['exp_id']
            exp_text = sample['exp_text']
            is_sent = sample['is_sent']

            vid_key = f"{src_dataset}_{vid_name}_{obj_id}"
            vid_info_map[f"{vid_key}_{exp_id}"] = {
                'vid_name': vid_name,
                'src_dataset': src_dataset,
                'obj_id': obj_id,
                'exp_id': exp_id,
                'exp_text': exp_text,
                'is_sent': is_sent
            }

            if os.path.exists(os.path.join(args.pred_path, vid_name, str(exp_id))):
                queue.put([vid_key, vid_name, str(exp_id)])

    print("Q-Size:", queue.qsize())

    start_time = time.time()
    processes = []
    for rank in range(args.num_workers):
        p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.mask_path, args.pred_path, meta_exp,
                                              args.allow_missing_masks))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f'Evaluation worker exited with code {p.exitcode}')

    expected = sum(len(v['expressions']) for v in meta_exp.values())
    if len(output_dict) != expected:
        raise RuntimeError(f'Evaluation coverage {len(output_dict)}/{expected}; refusing partial mean')

    raw_scores = np.asarray(list(output_dict.values()), dtype=np.float64)
    atomic_json({'expressions': expected,
                 'evaluation_policy': 'diagnostic_missing_as_zero' if args.allow_missing_masks else 'complete_coverage',
                 'J': float(100 * raw_scores[:, 0].mean()),
                 'F': float(100 * raw_scores[:, 1].mean()),
                 'J&F': float(100 * raw_scores.mean())},
                os.path.join(args.pred_path, '..', 'summary.json'))

    json.dump(dict(output_dict), open(args.save_name, 'w'), indent=4)

    data_list = []
    for exp_name, (j, f) in output_dict.items():
        parts = exp_name.split('_')
        src_dataset = parts[0]
        obj_id = parts[-2]
        exp_id = parts[-1]
        vid_name = '_'.join(parts[1:-2])

        info_key = f"{src_dataset}_{vid_name}_{obj_id}_{exp_id}"
        vid_info = vid_info_map.get(info_key, {})

        data = {
            'video_name': vid_name,
            'src_dataset': src_dataset,
            'obj_id': obj_id,
            'exp_id': exp_id,
            'exp_text': vid_info.get('exp_text', ''),
            'is_sent': vid_info.get('is_sent', 0),
            'videxp': exp_name,
            'J': round(100 * j, 2),
            'F': round(100 * f, 2),
            'JF': round(100 * (j + f) / 2, 2)
        }
        data_list.append(data)

    if data_list:
        df = pd.DataFrame(data_list)
        df.to_csv(args.save_csv_name, index=False)
        print(f"CSV results saved to {args.save_csv_name}")

        j_values = np.array([d['J'] for d in data_list])
        f_values = np.array([d['F'] for d in data_list])
        jf_values = np.array([d['JF'] for d in data_list])

        print(f'Overall J: {j_values.mean():.2f}')
        print(f'Overall F: {f_values.mean():.2f}')
        print(f'Overall J&F: {jf_values.mean():.2f}')

        if 'is_sent' in data_list[0]:
            sent_data = [d for d in data_list if d['is_sent'] == 1]
            non_sent_data = [d for d in data_list if d['is_sent'] == 0]

            if sent_data:
                print(sent_data[0])
                j_sent = np.array([d['J'] for d in sent_data]).mean()
                f_sent = np.array([d['F'] for d in sent_data]).mean()
                jf_sent = np.array([d['JF'] for d in sent_data]).mean()
                print(f'Sentence expressions - J: {j_sent:.2f}, F: {f_sent:.2f}, J&F: {jf_sent:.2f}')

            if non_sent_data:
                print(non_sent_data[0])
                j_non_sent = np.array([d['J'] for d in non_sent_data]).mean()
                f_non_sent = np.array([d['F'] for d in non_sent_data]).mean()
                jf_non_sent = np.array([d['JF'] for d in non_sent_data]).mean()
                print(f'Non-sentence expressions - J: {j_non_sent:.2f}, F: {f_non_sent:.2f}, J&F: {jf_non_sent:.2f}')
    else:
        print("No data to save in CSV")

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" % (total_time))
