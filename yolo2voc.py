import os
import xml.etree.cElementTree as ET
import cv2
import argparse


parser = argparse.ArgumentParser(description='yolo2voc')
parser.add_argument('-o', '--output_dir', default='/home/vladimir/github/FaceBoxes.PyTorch/data/data', type=str, help='Dir to save results')
parser.add_argument('-c', '--class_names', default='/home/vladimir/dataset/data.names', type=str, help='Path to YOLOv3 classes names file')
args = parser.parse_args()


def get_classes():
    with open(args.class_names, 'r') as f:
        names = [{'index': i, 'class': l} for i, l in enumerate(f)]
        return names


def parse_yolo_data(label_path, names):
    labels = []
    with open(label_path, 'r') as f:
        for l in f:
            buf = l[:-1].split(' ')
            if os.path.exists(buf[1]):
                img = cv2.imread(buf[1], cv2.IMREAD_COLOR)
                h, w, c = img.shape
                bboxs = []
                for i in range(4, len(buf), 5):
                    bboxs.append({'class': names['index' == buf[i]]['class'],
                                  'x_min': buf[i + 1],
                                  'y_min': buf[i + 2],
                                  'x_max': buf[i + 3],
                                  'y_max': buf[i + 4]})
                labels.append({'path': buf[1], 'width': w, 'height': h, 'channels': c, 'bboxs': bboxs})
    return labels


def save_voc_data(data_dict):
    file_name = os.path.split(data_dict['path'])[1]
    _, file_ext = os.path.splitext(data_dict['path'])
    dir_name = os.path.dirname(data_dict['path'])

    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = file_name
    ET.SubElement(root, "folder").text = dir_name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(data_dict['width'])
    ET.SubElement(size, "height").text = str(data_dict['height'])
    ET.SubElement(size, "depth").text = str(data_dict['channels'])

    for b in data_dict['bboxs']:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = b['class']
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(b['x_min'])
        ET.SubElement(bbox, "ymin").text = str(b['y_min'])
        ET.SubElement(bbox, "xmax").text = str(b['x_max'])
        ET.SubElement(bbox, "ymax").text = str(b['y_max'])

    tree = ET.ElementTree(root)
    xml_name = f'{file_name[:-len(file_ext)]}.xml'
    tree.write(os.path.join(os.path.join(args.output_dir, 'annotations'), xml_name))
    os.system(f'cp {data_dict["path"]} {os.path.join(os.path.join(args.output_dir, "images"), file_name)}')
    return file_name, xml_name

img_list = []
label_files = ['/home/vladimir/dataset/train.txt', '/home/vladimir/dataset/val.txt', '/home/vladimir/dataset/label.txt']

names = get_classes()
for l in label_files:
    labels = parse_yolo_data(l, names)
    for label in labels:
        img_name, xml_path = save_voc_data(label)
        img_list.append(f'{img_name} {xml_path}')

with open(os.path.join(args.output_dir, 'img_list.txt'), 'w') as f:
    for data in img_list:
        f.write(f'{data}\n')
