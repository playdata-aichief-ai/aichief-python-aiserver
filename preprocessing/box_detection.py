import json
import os
from ai.settings.settings import BASE_DIR
import cv2


class Box_Detection():
    def __init__(self):
        self.json_path = os.path.join(
            BASE_DIR, 'preprocessing', 'label', 'label.json')
        self.image_dict = {}

    def box_detect(self, name, cropped_image):
        with open(self.json_path, encoding='utf-8') as f:
            info = json.load(f)

        for i in info['images']:
            if not i['file_name'].startswith(f'{name}'):
                continue

            for j in info['annotations']:
                if i['id'] != j['image_id']:
                    continue

                for k in info['categories']:
                    if j['category_id'] == k['id']:
                        column_name = k['name']

                x = j['bbox'][0]
                y = j['bbox'][1]
                w = j['bbox'][2]
                h = j['bbox'][3]

                box_image = cropped_image[int(y): int(y+h), int(x): int(x+w)]
                
                if box_image.shape[0] < 32:
                    y_per = 32/box_image.shape[0]
                    # box_image = cv2.resize(box_image, dsize=(0, 0), fx=y_per, fy=y_per, interpolation=cv2.INTER_AREA)
                    box_image = cv2.resize(box_image, dsize=(0, 0), fx=y_per, fy=y_per, interpolation=cv2.INTER_LANCZOS4)

                self.image_dict[f'{column_name}'] = box_image

        return self.image_dict
