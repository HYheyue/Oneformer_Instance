# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import argparse
import multiprocessing as mp
import os
import torch
import random
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
import cv2
import numpy as np
import tqdm
import json
from pycocotools import mask as mask_tools

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from predictor import VisualizationDemo
from panopticapi.utils import IdGenerator
COCO_CATEGORIES = json.load(open('datasets/panoptic_basket_categories.json', 'r'))
categories = {0:COCO_CATEGORIES[0]}
id_generator = IdGenerator(categories)
import random
import cv2

# constants
WINDOW_NAME = "OneFormer Demo"
CUT_SIZE = 0
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../configs/ade20k/swin/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--task", help="Task type")
    parser.add_argument(
        "--input",
        nargs="+",
        help="Json format input or image list",
    )
    parser.add_argument(
        "--output",
        default="",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def get_json_pan_out(panoptic_seg, img_id, img_shape, left=0, right=0, top=0, bottom=0):
    pan_seg = np.array(predictions['panoptic_seg'][0].to('cpu'))
    new_label = np.zeros(img_shape)
    bbox = []
    rle_mask = []
    for i in np.unique(pan_seg):
        if i == 0:
            continue
        mask = (pan_seg == i).astype(np.uint8)
        if CUT_SIZE > 0:
            add_mask = np.zeros((CUT_SIZE, mask.shape[1]))
            mask = np.vstack([add_mask, mask])
        # import pdb; pdb.set_trace()
        # cv2.imwrite('mask.png', 255*mask)
        mask = np.pad(mask,((top, bottom),(left, right)), 'constant', constant_values=(0,0))
        # cv2.imwrite('mask2.png', 255*mask)

        # color = id_generator.get_color(0)   
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        new_label[mask == 1] = color

        rle = mask_tools.encode(np.asfortranarray(np.array(mask).astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_mask.append(rle)
        box_list = mask_tools.toBbox(rle).tolist()
        box_list[2], box_list[3] = box_list[0] + box_list[2], box_list[1] + box_list[3]
        bbox.append(box_list)
    cv2.imwrite('color_out/new_label_' + str(img_id) + '.png', new_label)
    # import pdb; pdb.set_trace()
    cls_id = [int(0) for i in range(len(rle_mask))]
    score = [int(1) for i in range(len(rle_mask))]

    image_result = {
        "labels": cls_id, 
        "scores": score,
        "bboxes": bbox,
        "masks":  rle_mask,
        "img_id": img_id
    }
    return image_result

def get_json_out(instances, img_id, img_shape, left=0, right=0, top=0, bottom=0):
    score = list(np.array(instances.get_fields()['scores'].to('cpu')))
    score = [float(i) for i in score]
    bbox = []
    rle_mask = []
    for box, mask in zip(instances.get_fields()['pred_boxes'].to('cpu'), instances.get_fields()['pred_masks'].to('cpu')):
        # import pdb; pdb.set_trace()
        if CUT_SIZE > 0:
            add_mask = np.zeros((CUT_SIZE, mask.shape[1]))
            mask = np.vstack([add_mask, mask])
        rle = mask_tools.encode(np.asfortranarray(np.array(mask).astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_mask.append(rle)
        box_list = mask_tools.toBbox(rle).tolist()
        box_list[2], box_list[3] = box_list[0] + box_list[2], box_list[1] + box_list[3]
        bbox.append(box_list)
    cls_id = list(np.array(instances.get_fields()['pred_classes'].to('cpu')))
    cls_id = [int(i) for i in cls_id]

    image_result = {
        "labels": cls_id, 
        "scores": score,
        "bboxes": bbox,
        "masks":  rle_mask,
        "img_id": img_id
    }
    return image_result

def get_edge(label):
    mask = np.array(label[0].to('cpu')) > 0
    if mask.max == 0:
        return 0, 0, 0, 0
    rle = mask_tools.encode(np.asfortranarray(np.array(mask).astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    box_list = mask_tools.toBbox(rle).tolist()
    if box_list[2] < 800:
        left = max(0, box_list[0] + box_list[2] / 2 - 400)
        right = mask.shape[1] - min((box_list[0] + box_list[2] / 2 + 400), mask.shape[1])
    else:
        left = max(box_list[0] - 100, 0)
        right = mask.shape[1] - min((box_list[0] +  box_list[2] + 100, mask.shape[1]))
    if box_list[3] < 800:
        top = max(0, box_list[1] + box_list[3] / 2 - 400)
        bottom = mask.shape[0] - min((box_list[1] + box_list[3] / 2 + 400), mask.shape[0])
    else:
        top = max(box_list[1] - 100, 0)
        bottom = mask.shape[0] - min((box_list[1] +  box_list[3] + 100, mask.shape[0]))
    # import pdb; pdb.set_trace()
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    # cv2.imwrite('mask.png', 255 * mask[top:-bottom, left:-right])
    return int(left), int(right), int(top), int(bottom)



if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    output_data,  output_pan = [], []
    if args.input:
        if args.input[0].endswith('.json'):
            data_list = json.load(open(args.input[0], 'r'))["images"]
            img_id_list = [i['id'] for i in data_list]
            # import pdb; pdb.set_trace()
            data_list = [os.path.join('/'.join(args.input[0].split('/')[:1]), \
                'basketball-instants-dataset', i['file_name']) for i in data_list]
        else:
            data_list = args.input
        index = 0
        for path in tqdm.tqdm(data_list):
            # use PIL, to be consistent with evaluation
            img_id = img_id_list[index]
            index += 1
            img = read_image(path, format="BGR")
            img_shape = img.shape
            img = img[CUT_SIZE:, :, :]
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.task)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            image_result = get_json_out(predictions['instances'], img_id, img_shape)
            output_data.append(image_result)
            left, right, top, bottom = 0, 0, 0, 0
            # left, right, top, bottom = get_edge(predictions['panoptic_seg'])

            if left * right * top * bottom > 0:
                print( left, right, top, bottom)
                predictions, visualized_output = demo.run_on_image(img[top:-bottom, left:-right, :], args.task)
                pan_result = get_json_pan_out(predictions['panoptic_seg'], img_id, img_shape, left, right, top, bottom)
                output_pan.append(pan_result)
            else:
                pan_result = get_json_pan_out(predictions['panoptic_seg'], img_id, img_shape)
                output_pan.append(pan_result)
            
            # import pdb; pdb.set_trace()
            if args.output:
                for k in visualized_output.keys():
                    os.makedirs(os.path.join('infer_result', k, args.output), exist_ok=True)
                    out_filename = os.path.join('infer_result', k, args.output, '-'.join(path.split('/')[-3:]))
                    visualized_output[k].save(out_filename)   
        # import pdb; pdb.set_trace()
        json.dump(output_data, open(f'test2.json', 'w'))
        json.dump(output_pan, open(args.input[1], 'w'))
    else:
        raise ValueError("No Input Given")
