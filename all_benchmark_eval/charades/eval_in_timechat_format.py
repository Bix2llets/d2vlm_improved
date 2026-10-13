import json
import os
import time
import sys
import argparse
import pdb
import nncore

def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas


def iou(A, B):
    # print(A, B)
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    return max(min1 - max0, 0) / (max1 - min0)


def toSec(timeStr):
    t = time.strptime(timeStr, "%H:%M:%S")
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec

def captiondata_modify(steps):
    modify_data = {}
    for i, step in enumerate(steps[0]):
        for key in step["step"].keys():
            name = step["step"][key]["query_idx"]
            modify_data[name] = [[step['step'][key]["startime"], step['step'][key]["endtime"]]]
        
    return modify_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pred_file", type=str, default="/users/wenzheng/code/projects/TRACE/eval_results/tvg_trace_embed_128/refmt_charades_test_f128_result.json")

    parser.add_argument("--pred_file", type=str, default="/storage/wenzheng/model_zoo/0306_final_important/dpo_focused_with_only_dpo_rvc_1loss_1/charades/result_converted.json") 
    parser.add_argument('--gt_file', type=str, default="/storage/wenzheng/dataset/TimeChat/TimeIT/data/temporal_video_grounding/charades/test.caption_coco_format.json")
    parser.add_argument('--sample', action='store_true', default=False)
    args = parser.parse_args()
    '''
    {
        "query_idx": [start_time, end_time],
        ...
    }
    '''
    answer = read_json(args.gt_file)
    answer = answer["annotations"]
    gt_meta = {}
    for jterm in answer:
        gt_meta[jterm["id"]] = {
            'video_id': jterm.get("image_id"),
            'interval': jterm["timestamp"],
            'caption': jterm.get("caption"),
        }

    submission = read_json(args.pred_file)

    thresholds = [0.0, 0.3, 0.5, 0.7]
    Result = {0: 0, 0.3: 0, 0.5: 0, 0.7: 0, 'miou': 0}
    Result_k = {0: 0, 0.3: 0, 0.5: 0, 0.7: 0, 'miou': 0}
    records = []
    num = 0
    k_name = 1

    for qid, jterm in submission.items():
        idx = int(qid)
        meta = gt_meta.get(idx)
        if meta is None or not isinstance(jterm, dict):
            continue

        gt_interval = meta['interval']
        video_id = meta.get('video_id', jterm.get('vid'))

        candidates = jterm.get('candidates') or [jterm.get('response', '')]
        candidate_timestamps = jterm.get('candidate_timestamps')
        if candidate_timestamps is None:
            candidate_timestamps = jterm.get('timestamps') or [jterm.get('timestamp')]

        k_name = max(k_name, len(candidates))

        per_example_ious = []
        for trial, (prompt_result, pred_interval) in enumerate(zip(candidates, candidate_timestamps)):
            parse_success = pred_interval is not None
            trial_iou = iou(gt_interval, pred_interval) if parse_success else 0.0
            per_example_ious.append(trial_iou)
            records.append({
                'example_idx': idx,
                'trial': trial,
                'video_id': video_id,
                'prompt_result': prompt_result,
                'ground_truth': {
                    'video_id': video_id,
                    'interval': gt_interval,
                    'caption': meta.get('caption'),
                },
                'iou': trial_iou,
                'parse_success': parse_success,
            })

        top1_iou = per_example_ious[0] if per_example_ious else 0.0
        best_iou = max(per_example_ious) if per_example_ious else 0.0
        for c_iou in thresholds:
            if top1_iou >= c_iou:
                Result[c_iou] += 1
            if best_iou >= c_iou:
                Result_k[c_iou] += 1
        Result['miou'] += top1_iou
        Result_k['miou'] += best_iou
        num += 1

    denom = num if num > 0 else 1
    print(f"# pred examples {len(submission)}; # evaluated examples {num}")
    print(num, sum(Result.values()), Result)
    print("IOU 0.3: {0}\nIOU 0.5: {1}\nIOU 0.7: {2}\n mIOU: {3}".format(Result[0.3]*100/denom, Result[0.5]*100/denom, Result[0.7]*100/denom, Result['miou']/denom))
    print("Recall@{K} 0.3: {0}\nRecall@{K} 0.5: {1}\nRecall@{K} 0.7: {2}\n mIOU@{K}: {3}".format(Result_k[0.3]*100/denom, Result_k[0.5]*100/denom, Result_k[0.7]*100/denom, Result_k['miou']/denom, K=k_name))

    good_cases = []
    good_cases.append(f'IOU 0.3: {Result[0.3]*100/denom}')
    good_cases.append(f'IOU 0.5: {Result[0.5]*100/denom}')
    good_cases.append(f'IOU 0.7: {Result[0.7]*100/denom}')
    good_cases.append(f'mIOU: {Result["miou"]/denom}')
    good_cases.append(f'Recall@{k_name} 0.3: {Result_k[0.3]*100/denom}')
    good_cases.append(f'Recall@{k_name} 0.5: {Result_k[0.5]*100/denom}')
    good_cases.append(f'Recall@{k_name} 0.7: {Result_k[0.7]*100/denom}')
    good_cases.append(f'mIOU@{k_name}: {Result_k["miou"]/denom}')

    output_path = '/'.join(args.pred_file.split("/")[:-1])
    output_file = os.path.join(output_path, "score_" + args.pred_file.split("/")[-1])
    with open(output_file, "w") as f:
        f.write("\n".join(good_cases))

    summary = {
        'num_examples': num,
        'num_trials': k_name,
        'top1': {
            'iou@0.3': Result[0.3] * 100 / denom,
            'iou@0.5': Result[0.5] * 100 / denom,
            'iou@0.7': Result[0.7] * 100 / denom,
            'mIoU': Result['miou'] / denom,
        },
        'best_of_k': {
            'iou@0.3': Result_k[0.3] * 100 / denom,
            'iou@0.5': Result_k[0.5] * 100 / denom,
            'iou@0.7': Result_k[0.7] * 100 / denom,
            'mIoU': Result_k['miou'] / denom,
        },
    }

    per_sample_file = os.path.join(output_path, "per_sample_" + args.pred_file.split("/")[-1])
    with open(per_sample_file, "w") as f:
        json.dump({'summary': summary, 'records': records}, f, indent=2, ensure_ascii=False)
    print(f"Wrote per-sample results to {per_sample_file}")