import json
from collections import defaultdict

def adjust_category(cat):
    adjustment = next((adj for start, end, adj in [
        (1, 11, 1),
        (13, 25, 2),
        (27, 28, 3),
        (31, 44, 5),
        (46, 65, 6),
        (67, 67, 7),
        (70, 70, 9),
        (72, 82, 10),
        (84, 90, 11),
    ] if start <= cat <= end), 0)
    return cat - adjustment

name_box_id = defaultdict(list)

with open("mscoco2017/annotations/instances_train2017.json", encoding='utf-8') as f:
    data = json.load(f)

annotations = data['annotations']
for ant in annotations:
    id = ant['image_id']
    name = f'mscoco2017/train2017/{id:012d}.jpg'
    cat = adjust_category(ant['category_id'])
    name_box_id[name].append([ant['bbox'], cat])

with open('train.txt', 'w') as f:
    for key, box_infos in name_box_id.items():
        box_info_strings = [
            f" {int(x)}, {int(y)}, {int(x+w)}, {int(y+h)}, {int(cat)}"
            for x, y, w, h, cat in (
                (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], cat)
                for bbox, cat in box_infos
            )
        ]
        f.write(f"{key}{''.join(box_info_strings)}\n")
