import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import models, applications

class Classification():
    
    
    def __init__(self):
        self.label = {0: 'abl-1', 1: 'abl-2', 2: 'aia', 3: 'bnp', 4: 'chubb', 
                    5: 'db', 6: 'dgb', 7: 'fubon', 8: 'hana-1', 9: 'hana-2', 10: 'hanhwa',
                    11: 'heungkuk', 12: 'kb', 13: 'kdb', 14: 'kyobo', 15: 'lina',
                    16: 'metlife', 17: 'miraeasset', 18: 'nh', 19: 'prudential', 20: 'samsung',
                    21: 'shinhan', 22: 'tongyang'}
        self.model = models.load_model('saved_models/vgg16_0.02_0.97.h5')
        
    def classify(self, scanned_image):
        img = cv2.resize(scanned_image, (448, 448), interpolation=cv2.INTER_AREA)
        
        img_arr = img_to_array(img)[np.newaxis, ...]

        input_tensor = applications.vgg16.preprocess_input(img_arr)

        pred = self.model.predict(input_tensor)
        category = self.label[np.argmax(pred)]
        # print(f'{file}')
        # print('pred: ', category, np.argmax(pred))
        
        return category