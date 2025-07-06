import ast
import os
import json
import glob
import numpy as np
import cv2
from argparse import ArgumentParser
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('inputs', type=str, help='Input image or folder path')
    parser.add_argument('model', type=str, help='Config file or model alias or checkpoint .pth file')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument('--out-dir', type=str, default='fasterRCNN_pred', help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--pred-score-thr', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-classes', type=int, default=16, help='Number of classes for color palette')
    parser.add_argument('--no-save-vis', action='store_true', help='Do not save visualization images')
    parser.add_argument('--no-save-pred', action='store_true', help='Do not save JSON results')
    parser.add_argument('--print-result', action='store_true', help='Print results')
    parser.add_argument('--chunked-size', '-s', type=int, default=-1, help='Chunked size for large models')

    args = vars(parser.parse_args())

    if args['no_save_vis'] and args['no_save_pred']:
        args['out_dir'] = ''

    if args['model'].endswith('.pth'):
        print_log('Model is a .pth file. Using it as weights.')
        args['weights'] = args['model']
        args['model'] = None

    init_keys = ['model', 'weights', 'device']
    init_args = {k: args.pop(k) for k in init_keys}

    return init_args, args

def get_palette(num_classes):
    """Generate unique colors for each class ID"""
    np.random.seed(42)
    return [tuple(np.random.randint(0, 256, size=3).tolist()) for _ in range(num_classes)]

def draw_bbox(image, bbox, label, score, class_colors):
    x1, y1, x2, y2 = map(int, bbox)
    color = class_colors[label % len(class_colors)]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f'{label:02d} {score:.2f}'  # ví dụ: "03 0.91"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
    cv2.putText(image, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image

def main():
    init_args, call_args = parse_args()

    inferencer = DetInferencer(**init_args)
    chunked_size = call_args.pop('chunked_size')
    inferencer.model.test_cfg.chunked_size = chunked_size

    num_classes = call_args.pop('num_classes')
    class_colors = get_palette(num_classes)

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
        if 'scores' in pred and len(pred['scores']) > 0:
            scores = pred['scores']
            max_idx = int(np.argmax(scores))
            top_score = float(scores[max_idx])
            top_label = int(pred['labels'][max_idx])
            top_bbox = [float(x) for x in pred['bboxes'][max_idx]]

            image_path = image_paths[i]
            image_name = os.path.basename(image_path)
            image_id = os.path.splitext(image_name)[0]

            # Save JSON
            if not call_args['no_save_pred']:
                out_json = {
                    'image': image_name,
                    'label_id': top_label,
                    'score': top_score,
                    'bbox': top_bbox
                }
                with open(os.path.join(out_dir, f'{image_id}_maxbbox.json'), 'w') as f:
                    json.dump(out_json, f, indent=2)

            # Save visualization
            if not call_args['no_save_vis']:
                image = cv2.imread(image_path)
                vis_image = draw_bbox(image, top_bbox, top_label, top_score, class_colors)
                cv2.imwrite(os.path.join(out_dir, f'{image_id}_vis.jpg'), vis_image)

    if out_dir and not (call_args['no_save_vis'] and call_args['no_save_pred']):
        print_log(f'Results have been saved at {out_dir}')

if __name__ == '__main__':
    main()


#### RUN 1 image
# python demo\image_demo_thinh.py data\coco\test2017\00_00003_.jpg checkpoints\frcnn_r2000\vis_data\config.py --weights checkpoints\frcnn_r2000\epoch_98.pth --out-dir pred_fasterRCNN --num-classes 16

#### RUN folder
# python demo\image_demo_thinh.py data\coco\test2017 checkpoints\frcnn_r2000\vis_data\config.py --weights checkpoints\frcnn_r2000\epoch_98.pth --out-dir pred_fasterRCNN --num-classes 16
