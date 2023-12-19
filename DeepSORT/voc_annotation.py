import xml.etree.ElementTree as ET
from os import getcwd

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert_annotation(year, image_id, list_file):
    with open(f'VOCdevkit/VOC{year}/Annotations/{image_id}.xml') as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                 int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for year, image_set in sets:
    with open(f'VOCdevkit/VOC{year}/ImageSets/Main/{image_set}.txt') as image_ids_file:
        image_ids = image_ids_file.read().strip().split()
        with open(f'{wd}/{year}_{image_set}.txt', 'w') as list_file:
            for image_id in image_ids:
                list_file.write(f'{wd}/VOCdevkit/VOC{year}/JPEGImages/{image_id}.jpg')
                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
