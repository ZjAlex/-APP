from eval import get_yolo_model
import numpy as np
from tqdm import tqdm
from PIL import Image



def ap_ar():
    yolo = get_yolo_model()
    val_path = '../input/widervals/wider_val/WIDER_val/val.txt'
    img_path = '../input/widervals/wider_val/WIDER_val/images/'
    with open(val_path) as f:
        val_lines = f.readlines()
    ap = 0.0
    ar = 0.0
    for i in tqdm(range(len(val_lines))):
        line = val_lines[i].split()
        image = Image.open(img_path + line[0])
        boxes, scores, classes = yolo.evaluate(image)
        if len(boxes) == 0:
            ap += 0.0
            ar += 0.0
            continue
        true_boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        true_boxes = np.reshape(true_boxes, (-1, 5))
        true_boxes = true_boxes[..., :4]
        precision, recall = precision_recall(true_boxes, boxes)
        ap += precision
        ar += recall
    ap = ap / len(val_lines)
    ar = ar / len(val_lines)

    print('ap: ' + str(ap))
    print('ar: ' + str(ar))


def precision_recall(true_boxes, pred_boxes):
    true_boxes_num = len(true_boxes)
    pred_boxes_num = len(pred_boxes)

    tp = box_iou_metric(true_boxes, pred_boxes)
    fp = pred_boxes_num - tp

    precision = tp / (tp + fp)
    recall = tp / true_boxes_num
    return precision, recall


def box_iou_metric(true_boxes, pred_boxes, iou_thresh=0.5):
    pred_boxes = np.expand_dims(pred_boxes, axis=0)
    pred_x1y1 = pred_boxes[..., 0:2] * 1.0
    pred_x2y2 = pred_boxes[..., 2:4] * 1.0
    pred_x1y1 = pred_x1y1[..., ::-1]
    pred_x2y2 = pred_x2y2[..., ::-1]
    true_boxes = np.expand_dims(true_boxes, axis=-2)
    true_x1y1 = true_boxes[..., 0:2] * 1.0
    true_x2y2 = true_boxes[..., 2:4] * 1.0

    pred_wh = np.maximum(0, pred_x2y2 - pred_x1y1)
    true_wh = np.maximum(0, true_x2y2 - true_x1y1)

    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    true_area = true_wh[..., 0] * true_wh[..., 1]

    inter_min = np.maximum(pred_x1y1, true_x1y1)
    inter_max = np.minimum(pred_x2y2, true_x2y2)

    inter_wh = np.maximum(0, inter_max - inter_min)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    iou = inter_area / (pred_area + true_area - inter_area + 1e-6)

    iou = np.max(iou, axis=-1)

    return np.sum(iou > iou_thresh)