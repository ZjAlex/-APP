
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from genData import *
from baseModel import *
from loss import *
import os
from eval import YOLO

def train():
    # annotation_path = '../input/widerface/widerface/WiderFace/wider_face_yolo_style_annotation.txt'

    annotation_path = '../input/anchors/wider_face_less_nums_annotation.txt'
    # annotation_path = '../input/trainanno/2012_train.txt'
    # annotation_val_path = '../input/trainval/2012_trainval.txt'
    log_dir = 'voc_data/logs/'
    classes_path = '../input/widerface/widerface/WiderFace/wider_face_classes.txt'
    anchors_path = '../input/anchors/wf_yolo_anchors_less_num_tiny.txt'
    # classes_path = '../input/voc-classes/voc_classes.txt'
    # anchors_path = '../input/yolov3-anchors/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (224, 224, 3)
    weights_path ='../input/tinyweightsv5/trained_weights_final_v5_tiny.h5'

    model = create_model(input_shape, anchors, num_classes, weights_path)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1)

    with open(annotation_path) as f:
        lines = f.readlines()
    # with open(annotation_val_path) as f:
    #        val_lines = f.readlines()
    #   lines = np.concatenate((lines, val_lines), axis=0)
    np.random.seed(44)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_train = len(lines) - 1000
    num_val = 1000

    print('stage 1')
    if True:
        for i in range(122):
            model.layers[i].trainable = False

        opt = Adam(lr=1e-3)
        model.compile(optimizer=opt, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        batch_size =32
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=5,
                            initial_epoch=0,
                            callbacks=[reduce_lr, early_stopping])
        model.save_weights('trained_weights_stage_1.h5')
    print('stage 2')
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable =True

        opt = Adam(lr=1e-4)
        model.compile(optimizer=opt, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        batch_size =32
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=15,
                            initial_epoch=5,
                            callbacks=[reduce_lr, early_stopping])
        model.save_weights('trained_weights_final_v3.h5')


def create_model(input_shape, anchors, num_classes, load_pretrained=True,
                 weights_path='../input/weights/trained_weights_final_v3_5.h5'):
    k.clear_session()  # get a new session
    h, w = input_shape[0:2]
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 16, 1: 8}[l], w // {0: 16, 1: 8}[l], \
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = mini_yolo(num_classes, num_anchors // 2, input_shape)
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model


def get_scores():
    data_path = 'rank/rank_score.txt'
    annotation = open(data_path, 'r')
    scores = {}
    max_ = 0
    min_ = 10
    while True:
        line = next(annotation, -1)
        if line == -1:
            break
        line = line.replace('\n', '')
        imgs_name = line.split(' ')[0]
        imgs_score = line.split(' ')[1]
        imgs_score = float(imgs_score)
        if imgs_score > max_:
            max_ = imgs_score
        if imgs_score < min_:
            min_ = imgs_score
        scores[imgs_name] = float(imgs_score)
    return scores, max_, min_


def train_rank_model():
    FLAGS = {}
    model_path = 'weightsv1/trained_weights_final_v5.h5'
    anchors_path = 'wider_face/wf_yolo_anchors_less_num.txt'
    classes_path = 'wider_face/wider_face_classes.txt'
    FLAGS['model_path'] = model_path
    FLAGS['anchors_path'] = anchors_path
    FLAGS['classes_path'] = classes_path
    FLAGS['score'] = 0.5
    yolo_model = YOLO(**(FLAGS))

    scores, max_, min_ = get_scores()
    scale = (max_ - min_) / 10
    img_path = '/home/alex/PycharmProjects/FFD/rank/Images/'
    image_list = os.listdir(img_path)
    md = rank_model()
    for j in range(5):
        for i in range(len(image_list)):
            img = Image.open(img_path + image_list[i])
            out_boxes, out_scores, out_classes, feats = yolo_model.train_rank(img)
            if len(out_boxes) > 1:
                continue
            out_boxes = np.rint(out_boxes / 8.0)
            out_boxes = np.maximum(0, out_boxes)
            out_boxes = out_boxes.astype(np.int32)
            y = round((scores[image_list[i]] - min_) / scale)
            y_bat = np.zeros((1, 10))
            y_bat[0][y - 1] = 1
            loss = md.train_on_batch(feats[:, out_boxes[0, 0]:out_boxes[0, 2], out_boxes[0, 1]:out_boxes[0, 3], :], y_bat)
            print('epochs: %0.2f, iters: %0.2f, loss: %0.2f' % (j, i, loss))
    md.save_weights('rank_model.h5')