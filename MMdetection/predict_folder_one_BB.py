import ast
import os
import json
import glob
import cv2
import numpy as np
from argparse import ArgumentParser
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('inputs', type=str, help='Input image file or folder path.')
    parser.add_argument('model', type=str, help='Config file or model alias or checkpoint .pth file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument('--out-dir', type=str, default='fasterRCNN_pred', help='Output directory')
    parser.add_argument('--texts', help='text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--pred-score-thr', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--show', action='store_true', help='Show the result')
    parser.add_argument('--no-save-vis', action='store_true', help='Do not save vis results')
    parser.add_argument('--no-save-pred', action='store_true', help='Do not save json results')
    parser.add_argument('--print-result', action='store_true', help='Print the results')
    parser.add_argument('--palette', default='none', choices=['coco', 'voc', 'citys', 'random', 'none'], help='Color palette')
    parser.add_argument('--custom-entities', '-c', action='store_true', help='Customize class names')
    parser.add_argument('--chunked-size', '-s', type=int, default=-1, help='Truncate predictions for large category set')
    parser.add_argument('--tokens-positive', '-p', type=str, help='Token positions of interest')

    args = vars(parser.parse_args())

    if args['no_save_vis'] and args['no_save_pred']:
        args['out_dir'] = ''

    if args['model'].endswith('.pth'):
        print_log('Model is a .pth file. Using it as weights.')
        args['weights'] = args['model']
        args['model'] = None

    if args['texts'] is not None and args['texts'].startswith('$:'):
        dataset_name = args['texts'][3:].strip()
        class_names = get_classes(dataset_name)
        args['texts'] = [tuple(class_names)]

    if args['tokens_positive'] is not None:
        args['tokens_positive'] = ast.literal_eval(args['tokens_positive'])

    init_keys = ['model', 'weights', 'device', 'palette']
    init_args = {k: args.pop(k) for k in init_keys}

    return init_args, args


def draw_bbox(image, bbox, label=None, score=None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f'{label}: {score:.2f}' if label is not None and score is not None else f'{score:.2f}'
    cv2.putText(image, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image


def main():
    init_args, call_args = parse_args()

    inferencer = DetInferencer(**init_args)
    chunked_size = call_args.pop('chunked_size')
    inferencer.model.test_cfg.chunked_size = chunked_size

    input_pattern = call_args['inputs']
    if os.path.isdir(input_pattern):
        image_paths = sorted(glob.glob(os.path.join(input_pattern, '*')))
    elif '*' in input_pattern:
        image_paths = sorted(glob.glob(input_pattern))
    else:
        image_paths = [input_pattern]

    results = inferencer(**call_args)

    out_dir = call_args['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    for i, pred in enumerate(results['predictions']):
        image_path = image_paths[i]
        image_name = os.path.basename(image_path)
        image_id = os.path.splitext(image_name)[0]

        if 'scores' in pred and len(pred['scores']) > 0:
            scores = pred['scores']
            max_idx = int(np.argmax(scores))
            top_score = float(scores[max_idx])
            top_label = int(pred['labels'][max_idx])
            top_bbox = [float(x) for x in pred['bboxes'][max_idx]]

            # Save JSON
            out_json = {
                'image': image_name,
                'label_id': top_label,
                'score': top_score,
                'bbox': top_bbox
            }
            if not call_args['no_save_pred']:
                with open(os.path.join(out_dir, f'{image_id}_maxbbox.json'), 'w') as f:
                    json.dump(out_json, f, indent=2)

            # Load and draw
            if not call_args['no_save_vis']:
                img = cv2.imread(image_path)
                vis_img = draw_bbox(img, top_bbox, label=top_label, score=top_score)
                save_path = os.path.join(out_dir, f'{image_id}_vis.jpg')
                cv2.imwrite(save_path, vis_img)

                if call_args['show']:
                    cv2.imshow('Result', vis_img)
                    cv2.waitKey(0)

    if out_dir and not (call_args['no_save_vis'] and call_args['no_save_pred']):
        print_log(f'Results have been saved at {out_dir}')


if __name__ == '__main__':
    main()
