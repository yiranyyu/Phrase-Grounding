import random
import subprocess

import numpy as np
import torch
from util import iou
from treelib import Tree


def cuda_is_available():
    import ctypes
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            del cuda
            try:
                import torch.cuda
            except ImportError:
                continue
            return True
    else:
        return False


def cuda_device_count(python_interpreter='python'):
    import os
    key = 'CUDA_VISIBLE_DEVICES'
    if key in os.environ:
        return len(os.environ[key].split(','))
    else:
        return int(subprocess.getoutput(f"{python_interpreter} -c 'import torch as th; print(th.cuda.device_count())'"))


def set_random_seed(s, deterministic=True):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    if cuda_is_available():
        torch.cuda.manual_seed_all(s)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = not deterministic
        cudnn.deterministic = deterministic


def normalize_bboxes(bboxes, image_w, image_h):
    # bbox: x1, y1, x2, y2
    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 2], bboxes[:, 3]
    box_w = x2 - x1
    box_h = y2 - y1
    scaled_x = x1 / image_w
    scaled_y = y1 / image_h
    scaled_w = box_w / image_w
    scaled_h = box_h / image_h

    scaled_w = scaled_w[..., np.newaxis]
    scaled_h = scaled_h[..., np.newaxis]
    scaled_x = scaled_x[..., np.newaxis]
    scaled_y = scaled_y[..., np.newaxis]
    spatial_features = np.concatenate(
        (
            scaled_x,
            scaled_y,
            scaled_x + scaled_w,
            scaled_y + scaled_h,
            scaled_w,
            scaled_h,
        ),
        axis=1,
    )
    return spatial_features


def detectGT(GT_bboxes, RoIs):
    """Find matched ROI indices By IoU >= 0.5.
    Args:
        GT_bboxes: entity GT bboxes
        RoIs: all detected ROI bboxes
    """

    indices = set()
    for GT_bbox in GT_bboxes:
        for i, detected_RoI in enumerate(RoIs):
            if iou(GT_bbox, detected_RoI) >= 0.5:
                indices.add(i)

    return sorted(indices)


def create_tree(width, height, entity_GT_boxes):
    tree = Tree()
    tree.create_node(tuple([0, 0, width, height]), -1)
    entity_GT_boxes = sorted(entity_GT_boxes, key=lambda x: [(x[2] - x[0]) * (x[3] - x[1])], reverse=True)
    id = -1
    for i, bbox in enumerate(entity_GT_boxes):
        max_iou = 0
        children = tree.all_nodes()
        for child in children[1:]:
            iou0 = iou(bbox, child.tag)
            if iou0 > max_iou:
                max_iou = iou0
                id = child.identifier
        if max_iou >= 0.1:
            tree.create_node(tuple(bbox), i, id)
        else:
            tree.create_node(tuple(bbox), i, -1)
    return tree
