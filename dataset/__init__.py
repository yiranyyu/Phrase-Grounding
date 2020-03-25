import csv
from pathlib import Path

FIELD_NAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features', 'global']


# noinspection PyShadowingNames
def _count_bbox_num(img_ids: set, file_path: Path):
    num = 0
    with file_path.open() as img_feature_file:
        reader = csv.DictReader(img_feature_file, delimiter='\t', fieldnames=FIELD_NAMES)
        for img_data in reader:
            img_id = int(img_data['image_id'])
            if img_id in img_ids:
                num += int(img_data['num_boxes'])
    return num
