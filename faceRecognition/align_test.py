import detect_face
import tensorflow as tf
import cv2
from PIL import Image, ImageDraw
import numpy as np
import time


img_path = '/home/alex/PycharmProjects/FaceRecognition/YOLO/Test/'
model_path = '/home/alex/PycharmProjects/facenet-master/src/align'

with tf.Session() as sess:
    pnet, rnet, onet = detect_face.create_mtcnn(sess, model_path)
    while True:
        name = input('please input img name: ')
        img = cv2.imread(img_path + name)
        start = time.time()
        boxes, points = detect_face.detect_face(img, 20, pnet, rnet, onet, [0.5, 0.5, 0.5], 0.709)
        end = time.time()
        print('inference time: %.2f' % (end - start))
        img = Image.open(img_path + name)
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle((box[0], box[1], box[2], box[3]))
        points = np.array(points).reshape((-1, 10))
        for point in points:
            draw.point((point[0], point[5]))
            draw.point((point[1], point[6]))
            draw.point((point[2], point[7]))
            draw.point((point[3], point[8]))
            draw.point((point[4], point[9]))
        img.show()