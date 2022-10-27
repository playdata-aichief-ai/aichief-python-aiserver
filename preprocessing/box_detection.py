import json

class Box_Detection():
    def __init__(self):
        self.json_path = './preprocessing/label/label.json'
        self.image_dict = {}
    
    def box_detect(self, name, cropped_image):
        with open(self.json_path) as f:
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
                    
                    self.image_dict[f'{column_name}'] = box_image
                    
        return self.image_dict