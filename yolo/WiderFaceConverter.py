import os


wf_box_path = '/home/alex/PycharmProjects/keras-yolo3-master/WiderFace/wider_face_split/wider_face_val_bbx_gt.txt'
wf_img_path = '/home/alex/PycharmProjects/keras-yolo3-master/WiderFace/WIDER_val/images'


annotation = open(wf_box_path, 'r')
new_annotation = '/home/alex/PycharmProjects/keras-yolo3-master/WiderFace/val.txt'

anno_file = open(new_annotation, 'w')

while True:
    line = next(annotation, -1)
    if line == -1:
        break

    line = line.replace('\n', '')
    folder_name = line.split('/')
    folder = folder_name[0]
    name = folder_name[1]
    img_path = wf_img_path + '/' + str(folder) + '/' + str(name)
    #if not os.path.exists(img_path):
    #    continue
    anno_file.write(str(folder) + '/' + str(name))
    boxes_num = next(annotation, -1)
    if int(boxes_num) > 10:
        for i in range(int(boxes_num)):
            box_ = next(annotation, -1)
        anno_file.write('\n')
        continue
    for i in range(int(boxes_num)):
        box_ = next(annotation, -1).split(' ')
        box = [int(box_[0]), int(box_[1]), int(box_[0]) + int(box_[2]), int(box_[1]) + int(box_[3])]
        anno_file.write(" " + ",".join([str(a) for a in box]) + ',' + str(0))
    anno_file.write('\n')

anno_file.close()

with open(new_annotation, 'r') as f:
    lines = f.readlines()

with open(new_annotation, 'w') as f:
    for ln in lines:
        if len(ln.split(' ')) == 1:
            continue
        f.write(ln)
