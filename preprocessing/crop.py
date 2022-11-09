import cv2
import numpy as np
from imutils.perspective import four_point_transform
from collections import defaultdict

class Crop():
    def __init__(self):
        self.size = {'abl-1' : (600, 725, 50, 175, 150),
                    'abl-2' : (610, 670, 50, 175, 150),
                    'aia' : (607, 800, 50, 150, 250),
                    'bnp' : (615, 760, 50, 125 ,100),
                    'chubb' : (585, 640, 50, 150, 150),
                    'db' : (655, 877, 10, 175, 150),
                    'dgb' : (600, 830, 50, 150, 150),
                    'fubon' : (620, 760, 50, 200, 100),
                    'hana-1' : (570, 810, 50, 150, 150),
                    'hana-2' : (580, 815, 50, 150, 150),
                    'hanhwa' : (630, 840, 50, 150, 150),
                    'heungkuk' : (635, 825, 50, 150, 150),
                    'kb' : (615, 790, 50, 150, 150),
                    'kdb' : (653, 912, 50, 150, 250),
                    'kyobo' : (628, 815, 50, 170, 125),
                    'lina' : (572, 838, 50, 150, 150),
                    'metlife' : (600, 730, 50, 150, 150),
                    'miraeasset' : (635, 720, 50, 150, 125),
                    'nh' : (575, 807, 10, 200, 65),
                    'prudential' : (650, 745, 75, 200, 125),
                    'samsung' : (630, 835, 40, 140, 100),
                    'shinhan' : (610, 800, 75, 200, 125),
                    'tongyang' : (600, 840, 50, 150, 150),
                    }
    
    
    def crop(self, name, img):
        def area_detect(img, lineWitdth=30, k=15, houghThresh1=200, houghThresh2=65):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(blur)
            thr = cv2.adaptiveThreshold(cl1, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 21, 7)
            thr=~thr
            
            line_min_width = lineWitdth
            kernel_h = np.ones((1, line_min_width), np.uint8)
            kernel_v = np.ones((line_min_width, 1), np.uint8)
            k2 = np.ones((k,1), np.uint8)
            
            thr_h = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_h)
            thr_v = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_v)

            morph = cv2.morphologyEx(thr_h, cv2.MORPH_CLOSE, k2)
            
            combined = morph | thr_v
            
            ke = np.ones((10,10), np.uint8)
            ad = cv2.morphologyEx(thr_h, cv2.MORPH_DILATE, ke)
            contours, hierarchy = cv2.findContours(ad, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key = cv2.contourArea, reverse = True)

            newImage = img.copy()

            largestContour = contours[0]
            minAreaRect = cv2.minAreaRect(largestContour)

            angle = minAreaRect[-1]
            if angle < -45:
                angle = 90 + angle
            if angle > 1:
                angle = 90 - angle
                angle = -1.0 * angle
            
            (h, w) = newImage.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            combined = cv2.warpAffine(combined, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(blur)
            thr = cv2.adaptiveThreshold(cl1, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 21, 7)
            thr=~thr
            
            line_min_width = lineWitdth
            kernel_h = np.ones((1, line_min_width), np.uint8)
            kernel_v = np.ones((line_min_width, 1), np.uint8)
            k2 = np.ones((k,1), np.uint8)
            
            thr_h = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_h)
            thr_v = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_v)

            
            lines = cv2.HoughLinesP(thr_h, 1, np.pi / 180., houghThresh1, minLineLength=20, maxLineGap=20)
            
            xmin, ymin, xmax, ymax = 0, 0, 0, 0
            if lines is not None: # 라인 정보를 받았으면
                for i in range(lines.shape[0]):
                    pt1 = (lines[i][0][0], lines[i][0][1]) # 시작점 좌표 x,y
                    pt2 = (lines[i][0][2], lines[i][0][3]) # 끝점 좌표, 가운데는 무조건 0
                    if (pt1[0] < 10) or (pt1[0] > newImage.shape[1]-10) or (pt2[0] < 10) or (pt2[0] > newImage.shape[1]-10):
                        continue
                    if (pt1[1] < 10) or (pt1[1] > newImage.shape[0]-10) or (pt2[1] < 10) or (pt2[1] > newImage.shape[0]-10):
                        continue

                    if xmin == 0:
                        xmin = pt1[0]
                        ymin = pt1[1]
                        xmax = pt1[0]
                        ymax = pt1[1]
                    
                    if xmin > pt1[0]:
                        xmin = pt1[0]
                    if xmin > pt2[0]:
                        xmin = pt2[0]
                        
                    if ymin > pt1[1]:
                        ymin = pt1[1]
                    if ymin > pt2[1]:
                        ymin = pt2[1]
                        
                    if xmax < pt1[0]:
                        xmax = pt1[0]
                    if xmax < pt2[0]:
                        xmax = pt2[0]
                        
                    if ymax < pt1[1]:
                        ymax = pt1[1]
                    if ymax < pt2[1]:
                        ymax = pt2[1]
                    
            pts = np.array([(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)])

            cropped = four_point_transform(newImage, pts)
            
            return cropped
        
        result = area_detect(img, 50, self.size[f'{name}'][2], self.size[f'{name}'][3], self.size[f'{name}'][4])
        if (result.shape[0] < 450) or (result.shape[1] < 350):
            raise Exception('Area Detection Failed')
        
        result = cv2.resize(result, (self.size[f'{name}'][0], self.size[f'{name}'][1]), interpolation=cv2.INTER_AREA)
        
        return result