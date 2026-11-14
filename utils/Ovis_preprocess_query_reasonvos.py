import argparse
import re
import sys
from typing import Optional, List, Dict, Any

import json
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torch
from tqdm import tqdm
from torch.multiprocessing import set_start_method
import misc as utils
import numpy as np
import os
import random
import math
import time
import multiprocessing as mp
import warnings
from utils.reasonvos_runtime import (OVIS_PATH, DATA_ROOT, META_PATH, RUN_ROOT, atomic_json,
    normalize_qa, ensure_identity, file_sha256)
from utils.reasonvos_reflection import record_model_response, MODEL_TRACE

warnings.filterwarnings("ignore")

OUTPUT = str(RUN_ROOT / 'query_qa')
OVIS_CHECKPOINT = OVIS_PATH


def main(args):
    print("Inference only supports for batch size = 1")
    print(args)
    if args.max_query_num < 1:
        raise ValueError('Retry budget must be positive')
    identity = {'schema': 'reasonvos-attribute-qa-v2', 'meta': file_sha256(META_PATH),
                'source': file_sha256(__file__),
                'runtime': file_sha256(os.path.join(os.path.dirname(__file__), 'reasonvos_runtime.py')),
                'reflection': file_sha256(os.path.join(os.path.dirname(__file__), 'reasonvos_reflection.py')),
                'model_config': file_sha256(os.path.join(OVIS_PATH, 'config.json')),
                'generation_config': file_sha256(os.path.join(OVIS_PATH, 'generation_config.json')),
                'retries': args.max_query_num, 'gpus': args.num_gpus, 'seed': args.seed}
    ensure_identity(RUN_ROOT / 'qa_identity.json', identity,
                    [RUN_ROOT / 'query_qa/checkpoints', RUN_ROOT / 'query_qa/reasonvos_query_qa.json'])

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = args.split
    # save path
    save_dir = OUTPUT
    output_dir = save_dir
    os.makedirs(output_dir, exist_ok=True)

    # load data
    root = str(DATA_ROOT)
    img_folder = os.path.join(root, 'JPEGImages')
    meta_file = str(META_PATH)
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]

    video_list = list(data.keys())

    # create subprocess
    thread_num = args.num_gpus
    result_dict = mp.Manager().dict()

    processes = []
    lock = mp.Lock()

    video_num = len(video_list)
    per_thread_video_num = math.ceil(float(video_num) / float(thread_num))

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, args, data,
                                                   output_dir,
                                                   img_folder, sub_video_list, result_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f'QA worker exited with code {p.exitcode}')

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    all_results = {}
    for pid, results in result_dict.items():
        all_results.update(results)

    json.dump(all_results, open(os.path.join(output_dir, "reasonvos_query_qa.json"), "w"), indent=4)


convert_query_prompt = """
# Subtask1
Given a sentence '{query}'.
If it is an interrogative sentence, convert it into the format of a declarative sentence while minimizing structural changes to the original sentence. Preserve the positions of subject, predicate, object, and adverbial as much as possible while achieving the conversion goal. If the subject cannot be clearly identified, use "target" or "object" as a substitute. Otherwise, use the subject from the query directly. If the query refers to multiple types of definite targets, treat all categories as a single whole.
If the input is a declarative sentence, output it unchanged.
The converted sentence should be a descriptive statement without phrases like "is the one in question" or "is the one in query".

# Subtask2
Classify the converted sentence into one of the following types:
[1] Sentences with clear temporal constraints, especially those describing positions that may change, or actions that cannot be determined from a single image, and those with explicit beginning or end markers (e.g., 'initially', 'at the beginning', 'first', 'finally', 'at the end', 'last'). For such sentences, return type 1.
[2] Sentences where the subject consists of multiple distinct targets, such as a cat and a dog, or a person in white clothes and a person in red clothes, but excluding cases like people in red clothes as they can be categorized as one group. For such sentences, return type 2.
[3] For all other sentences, return type 3.

# Output Json Format
'result': the converted result
'type': the classified type (1, 2, or 3)
"""
def convert_query_with_vlm(query, vl_model):
    question = convert_query_prompt.format(query=query)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user", "content": [{"type": "text", "text": question}]
        }
    ]

    input_ids, pixel_values, grid_thws = vl_model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=True,
        max_pixels=896 * 896
    )
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda().to(vl_model.dtype) if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    with torch.no_grad():
        outputs = vl_model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=True,
            enable_thinking_budget=True,
            max_new_tokens=2048,
            thinking_budget=1536,
            do_sample=True,
            eos_token_id=vl_model.text_tokenizer.eos_token_id,
            pad_token_id=vl_model.text_tokenizer.pad_token_id
        )

    response = vl_model.text_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    record_model_response('convert_query_with_vlm', question, response)
    # Extract JSON from the response (assuming the model outputs valid JSON)
    try:
        # Find the JSON part (after any thinking steps)
        response = response.strip()
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()

        # Parse JSON to extraction
        output_json = json.loads(response)
        result = output_json.get("result", '')
        type = int(output_json.get("type", -1))


    except Exception:
        # Fallback if JSON parsing fails
        result = ''
        type = -1

    return result, type

generate_qa_prompt = """
# Task
Decompose the ORIGINAL query below into 0-3 discriminative attribute checks for the
referred target, following high-level concepts (category, state, role) and low-level
attributes (appearance, shape, spatial location). Query: {description}

# Rules
1. Each question starts with "Is" and has options A="Yes", B="No"; correct_answer="A".
2. Ask only about constraints explicitly given by the query. Preserve negation,
   comparisons and which participant plays the referred role. Never invent a color,
   identity or appearance to make the implicit query easier.
3. Reject vacuous checks such as "Is the target an item/object/thing?" that discard
   the discriminating constraints. Combine related attributes into one question.
4. Checks must be answerable about a highlighted target in a keyframe. Do not turn
   temporal/causal claims into assumed static facts. If no discriminative attribute
   is verifiable from a keyframe, return [] rather than a generic always-true check.
   Temporal intent is analyzed over the video by the intent/existence agents.
5. Apply these rules to all query types, including temporal and multi-target queries.
   Use singular "the target" when checking each predicted target.
6. Output ONLY a JSON list, including [] when no applicable check exists:
[{{"question": "Is the target ...?", "options": {{"A": "Yes", "B": "No"}}, "correct_answer": "A"}}]
"""

def generate_qa_with_vlm(description, type, vl_model):
    question = generate_qa_prompt.format(description=description)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": question}]
        }
    ]

    input_ids, pixel_values, grid_thws = vl_model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=True,
        max_pixels=896 * 896
    )
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda().to(vl_model.dtype) if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    with torch.no_grad():
        outputs = vl_model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=True,
            enable_thinking_budget=True,
            max_new_tokens=2048,
            thinking_budget=1536,
            do_sample=True,
            eos_token_id=vl_model.text_tokenizer.eos_token_id,
            pad_token_id=vl_model.text_tokenizer.pad_token_id
        )

    response = vl_model.text_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    record_model_response('generate_qa_with_vlm', question, response)

    # Extract JSON from the response
    try:
        # Find the JSON part (after any thinking steps)
        response = response.strip()
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()

        qa_data = extract_qa_generation_output(response)
        if qa_data is None:
            raise ValueError('No JSON question list')
        qa_data = normalize_qa(qa_data)
        if len(qa_data) > 3:
            raise ValueError('Expected at most 3 attribute questions')

    except Exception as e:
        print(f"Error in generate_qa_with_vlm: {e}")
        qa_data = None

    return qa_data

def extract_qa_generation_output(data_string: str) -> Optional[List[Dict[str, Any]]]:
    text = data_string.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if not match:
            return None
        text = match.group(1)
    try:
        value = json.loads(text)
        return normalize_qa(value)
    except (ValueError, TypeError):
        return None

def sub_processor(lock, pid, args, data, save_path_prefix, img_folder, video_list, result_dict):
    torch.manual_seed(args.seed + pid)
    np.random.seed(args.seed + pid)
    random.seed(args.seed + pid)
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)
    device = torch.device(f"cuda:{pid}")

    ovis_model = AutoModelForCausalLM.from_pretrained(
        OVIS_CHECKPOINT,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    atomic_json({'ovis': ovis_model.generation_config.to_dict(),
                 'llm': ovis_model.llm.generation_config.to_dict()},
                RUN_ROOT / f'qa_generation_config_{pid}.json')

    ovis_model.eval()
    results = {}
    encountered_new = False

    # 1. For each video
    for video in video_list:
        torch.cuda.empty_cache()
        metas = []  # list[dict], length is number of expressions

        expressions = data[video]["expressions"]
        num_expressions = len(expressions)

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["exp"] = expressions[i]["exp_text"]
            meta["exp_id"] = str(expressions[i]['exp_id'])
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # 2. For each expression
        for i in range(num_expressions):
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            video_exp = f'{video}/{exp_id}'
            checkpoint = os.path.join(save_path_prefix, 'checkpoints', video, exp_id + '.json')
            if os.path.exists(checkpoint):
                if encountered_new:
                    raise ValueError('QA resume must be a complete prefix')
                results[video_exp] = json.load(open(checkpoint))
                entry = results[video_exp]
                if entry['exp'] != exp.strip():
                    raise ValueError('QA checkpoint query mismatch')
                entry['qa_complete'] = normalize_qa(entry['qa_complete'])
                for name, suffix in [('rng', '.rng.pt'), ('trace', '.trace.json')]:
                    if file_sha256(checkpoint + suffix) != entry['artifact_sha256'][name]:
                        raise ValueError('QA checkpoint artifact mismatch')
                rng = torch.load(checkpoint + '.rng.pt', map_location='cpu', weights_only=False)
                random.setstate(rng['python'])
                np.random.set_state(rng['numpy'])
                torch.set_rng_state(rng['torch'])
                torch.cuda.set_rng_state(rng['cuda'], device)
                continue
            encountered_new = True
            trace = []
            token = MODEL_TRACE.set(trace)

            # exp = 'Which shoe(s) is/are on the right foot of the man?'

            try:
                exp = exp.strip()
                for query_round in range(args.max_query_num):
                    description, type = convert_query_with_vlm(exp, ovis_model)
                    if description and type != -1:
                        break

                if not exp.strip().endswith('?'):
                    description = exp.strip()
                else:
                    description = description.strip()

                if not description or type not in (1, 2, 3):
                    raise ValueError('Query conversion failed after retries')

                for query_round in range(args.max_query_num):
                    # Preserve the original query instead of a potentially lossy conversion.
                    qa_complete = generate_qa_with_vlm(exp, type, ovis_model)
                    if qa_complete is not None:
                        break
                if qa_complete is None:
                    raise ValueError('Attribute QA generation failed after retries')

                results[video_exp] = {
                    'exp': exp,
                    'description': description,
                    'type': type,
                    'qa_complete': qa_complete,
                    'qa_status': 'applicable' if qa_complete else 'not_applicable',
                    'qa_schema': 'reasonvos-attribute-qa-v2',
                }
                atomic_json(trace, checkpoint + '.trace.json')
                rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
                       'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}
                torch.save(rng, checkpoint + '.rng.pt.tmp')
                os.replace(checkpoint + '.rng.pt.tmp', checkpoint + '.rng.pt')
                results[video_exp]['artifact_sha256'] = {
                    'rng': file_sha256(checkpoint + '.rng.pt'),
                    'trace': file_sha256(checkpoint + '.trace.json')}
                atomic_json(results[video_exp], checkpoint)
            except Exception:
                atomic_json({'video': video, 'exp_id': exp_id, 'model_calls': trace},
                            os.path.join(save_path_prefix, 'failed_expression.json'))
                raise
            finally:
                MODEL_TRACE.reset(token)

        with lock:
            progress.update(1)

    # json.dump(results, open(os.path.join(save_path_prefix, f'result_{pid}.json'), 'w'), indent=2)

    result_dict[str(pid)] = results
    with lock:
        progress.close()


if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='valid',
                        help='valid or valid_u')
    parser.add_argument("--refine_round", default=0, type=int)
    parser.add_argument("--max_query_num", default=3, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_gpus', '-ng', type=int, required=True,
                        help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
    args = parser.parse_args()

    main(args)
