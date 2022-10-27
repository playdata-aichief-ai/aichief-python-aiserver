import cv2

class Crop():
    def __init__(self):
        pass
    
    def crop(self, scanned_image):
        img = scanned_image[10:scanned_image.shape[0]-10, 10:scanned_image.shape[1]-10]
        
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img,None)
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
        for i in kp:
            if x_min == 0:
                x_min = i.pt[0]
                x_max = i.pt[0]
                y_min = i.pt[1]
                y_max = i.pt[1]
                
            if x_min > i.pt[0]:
                x_min = i.pt[0]
                
            if x_max < i.pt[0]:
                x_max = i.pt[0]
                
            if y_min > i.pt[1]:
                y_min = i.pt[1]
                
            if y_max < i.pt[1]:
                y_max = i.pt[1]
                
        cropped = img[int(y_min): int(y_max), int(x_min): int(x_max)]
        cropped_image = cv2.resize(cropped, (774,1000), interpolation=cv2.INTER_AREA)
        
        # cv2.imwrite(os.path.join(base_path, crop_path) + '/cropped-' + file, cropped_image)
        
        # plt.figure(figsize=(6,6))
        # plt.imshow(cropped_image, cmap='gray')
        # plt.title(f'cropped {file}')
        # plt.show() 
        
        return cropped_image