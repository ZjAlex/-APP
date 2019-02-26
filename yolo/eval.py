
import numpy as np
import keras.backend as K
import colorsys
import os
from baseModel import *
from loss import yolo_eval
from genData import letterbox_image
from timeit import default_timer as timer
from PIL import Image


class YOLO(object):
    _defaults = {
        "model_path": 'weightsv1/V3.h5',
        "anchors_path": 'wider_face/wf_yolo_anchors.txt',
        "classes_path": 'wider_face/wider_face_classes.txt',
        "score": 0.3,
        "iou": 0.01,
        "model_image_size" : (224, 224),
        "gpu_num" : 0,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        print(class_names)
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        print(num_classes)
        print(num_anchors)
        self.yolo_model = mini_yolo(num_classes, num_anchors // 3)
        self.yolo_model.load_weights(self.model_path, by_name=True, skip_mismatch=True)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def evaluate(self, image):

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        return out_boxes, out_scores, out_classes


    def train_rank(self, image):

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        return out_boxes, out_scores, out_classes

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    rank_md = rank_model()
    rank_md.load_weights('rank/rank_model.h5')
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image_with_score(rank_md, image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


def get_yolo_model():
    FLAGS = {}
    model_path = '../input/weights/trained_weights_final_v3_4.h5'
    classes_path = '../input/widerface/widerface/WiderFace/wider_face_classes.txt'
    anchors_path = '../input/widerfaceanchors/wf_yolo_anchors.txt'
    FLAGS['model_path'] = model_path
    FLAGS['anchors_path'] = anchors_path
    FLAGS['classes_path'] = classes_path
    FLAGS['score'] = 0.3
    return YOLO(**FLAGS)


def detect_img(yolo):
    test_path = '/home/alex/PycharmProjects/keras-yolo3-master/TestData'
    rank_md = rank_model()
    rank_md.load_weights('rank/rank_model_v1.h5')
    while True:
        img_name = input('please input image name: ')
        img_path = os.path.join(test_path, img_name)

        try:
            image = Image.open(img_path)
        except:
            print('Open Error! Try again!')
            continue
        image = yolo.detect_image_with_score(rank_md, image)
        image.show()
    yolo.close_session()

FLAGS = {}


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    model_path = 'weightsv1/trained_weights_final_v5.h5'
    anchors_path = 'wider_face/wf_yolo_anchors_less_num.txt'
    classes_path = 'wider_face/wider_face_classes.txt'
    gpu_num = 0
    image_mode = True
    video_input_path = ''
    video_output_path = ''
    FLAGS['model_path'] = model_path
    FLAGS['anchors_path'] = anchors_path
    FLAGS['classes_path'] = classes_path
    FLAGS['gpu_num'] = gpu_num
    FLAGS['score'] = 0.5
    yolo = YOLO(**FLAGS)
    if image_mode:
        print("Image detection mode")
        detect_img(yolo)
    else:
        print("Video detection mode")
        detect_video(yolo, 0)