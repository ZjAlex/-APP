import numpy as np
import keras.backend as k
import tensorflow as tf



def box_iou(a, b):
    a = k.expand_dims(a, axis=-2)
    a_xy = a[..., :2]
    a_wh = a[..., 2:4]
    a_wh_half = a_wh / 2.0
    a_mins = a_xy - a_wh_half
    a_maxs = a_xy + a_wh_half

    b = k.expand_dims(b, axis=0)
    b_xy = b[..., :2]
    b_wh = b[..., 2:4]
    b_wh_half = b_wh / 2.0
    b_mins = b_xy - b_wh_half
    b_maxs = b_xy + b_wh_half

    inter_min = k.maximum(a_mins, b_mins)
    inter_max = k.maximum(a_maxs, b_maxs)

    inter_wh = k.maximum(inter_max - inter_min, 0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    a_area = a_wh[..., 0] * a_wh[..., 1]
    b_area = b_wh[..., 0] * b_wh[..., 1]

    iou = inter_area / (a_area + b_area - inter_area + 0.000001)
    return iou


def convert_outputs_to_box(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    anchor_tensor = k.reshape(k.constant(anchors), [1, 1, 1, num_anchors, 2])
    grid_shape = k.shape(feats)[1:3]
    grid_y = k.tile(k.reshape(k.arange(0, grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])

    grid_x = k.tile(k.reshape(k.arange(0, grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])

    grid = k.concatenate([grid_x, grid_y])
    grid = k.cast(grid, k.dtype(feats))

    feats = k.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    box_xy = (k.sigmoid(feats[..., :2]) + grid) / k.cast(grid_shape[::-1], k.dtype(feats))

    box_wh = k.exp(feats[..., 2:4]) * anchor_tensor / k.cast(input_shape[::-1], k.dtype(feats))

    box_confidence = k.sigmoid(feats[..., 4:5])

    box_class_prob = k.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_prob


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = k.cast(input_shape, k.dtype(box_yx))
    image_shape = k.cast(image_shape, k.dtype(box_yx))
    new_shape = k.round(image_shape * k.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = k.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= k.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = convert_outputs_to_box(feats,
                                                                             anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = k.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = k.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]] # default setting
    input_shape = k.shape(yolo_outputs[0])[1:3] * 16
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = k.concatenate(boxes, axis=0)
    box_scores = k.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = k.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = k.gather(class_boxes, nms_index)
        class_box_scores = k.gather(class_box_scores, nms_index)
        classes = k.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = k.concatenate(boxes_, axis=0)
    scores_ = k.concatenate(scores_, axis=0)
    classes_ = k.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    input_shape = input_shape[0:2]
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:16, 1:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand d im to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')#为什么不去掉w《0的
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=True):
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]
    input_shape = k.cast(k.shape(yolo_outputs[0])[1:3] * 16, k.dtype(y_true[0]))
    grid_shapes = [k.cast(k.shape(yolo_outputs[l])[1:3], k.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = k.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = k.cast(m, k.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = convert_outputs_to_box(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = k.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = k.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = k.switch(object_mask, raw_true_wh, k.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(k.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = k.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = k.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, k.cast(best_iou<ignore_thresh, k.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = k.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = k.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * k.binary_crossentropy(raw_true_xy, raw_pred[...,0:2]+1e-6, from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * k.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * k.binary_crossentropy(object_mask, raw_pred[...,4:5]+1e-6, from_logits=True)*(1+k.square(1-raw_pred[...,4:5])) + \
            (1-object_mask) * k.binary_crossentropy(object_mask, raw_pred[...,4:5]+1e-6, from_logits=True) * ignore_mask*(1+k.square(raw_pred[...,4:5]))
        class_loss = object_mask * k.binary_crossentropy(true_class_probs, raw_pred[...,5:]+1e-6 , from_logits=True)*(1+k.square(1-raw_pred[...,5:]))

        xy_loss = k.sum(xy_loss) / mf
        wh_loss = k.sum(wh_loss) / mf
        confidence_loss = k.sum(confidence_loss) / mf
        class_loss = k.sum(class_loss) / mf
        loss += xy_loss +  wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, k.sum(ignore_mask)], message='loss: ')
    return loss
