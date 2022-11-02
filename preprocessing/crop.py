import cv2
import numpy as np
from imutils.perspective import four_point_transform
from collections import defaultdict

class Crop():
    def __init__(self):
        self.size = {'abl-1' : (600, 725),
                    'abl-2' : (610, 670),
                    'aia' : (607, 800),
                    'bnp' : (615, 760),
                    'chubb' : (585, 640),
                    'db' : (655, 877),
                    'dgb' : (600, 830),
                    'fubon' : (620, 760),
                    'hana-1' : (570, 810),
                    'hana-2' : (580, 815),
                    'hanhwa' : (630, 840),
                    'heungkuk' : (635, 825),
                    'kb' : (615, 790),
                    'kdb' : (653, 912),
                    'kyobo' : (628, 815),
                    'lina' : (572, 838),
                    'metlife' : (600, 730),
                    'miraeasset' : (635, 720),
                    'nh' : (575, 807),
                    'prudential' : (650, 745),
                    'samsung' : (630, 835),
                    'shinhan' : (610, 800),
                    'tongyang' : (600, 840),
                    }
    
    
    def crop(self, name, img):
        def segment_by_angle_kmeans(lines, k=2, **kwargs):
            """
            Group lines by their angle using k-means clustering.
            """

            # Define criteria = (type, max_iter, epsilon)
            default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
            criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))

            flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
            attempts = kwargs.get('attempts', 10)

            # Get angles in [0, pi] radians
            angles = np.array([line[0][1] for line in lines])

            # Multiply the angles by two and find coordinates of that angle on the Unit Circle
            pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)

            # Run k-means
            labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]

            labels = labels.reshape(-1) # Transpose to row vector

            # Segment lines based on their label of 0 or 1
            segmented = defaultdict(list)
            for i, line in zip(range(len(lines)), lines):
                segmented[labels[i]].append(line)

            segmented = list(segmented.values())
            # print("Segmented lines into two groups: %d, %d" % (len(segmented[0]), len(segmented[1])))

            return segmented


        def intersection(line1, line2):
            """
            Find the intersection of two lines 
            specified in Hesse normal form.

            Returns closest integer pixel locations.
            """

            rho1, theta1 = line1[0]
            rho2, theta2 = line2[0]
            A = np.array([[np.cos(theta1), np.sin(theta1)],
                        [np.cos(theta2), np.sin(theta2)]])
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))

            return [[x0, y0]]


        def segmented_intersections(lines):
            """
            Find the intersection between groups of lines.
            """

            intersections = []
            for i, group in enumerate(lines[:-1]):
                for next_group in lines[i+1:]:
                    for line1 in group:
                        for line2 in next_group:
                            intersections.append(intersection(line1, line2)) 

            return intersections
        
        def area_detect(img, lineWitdth=30, k=15, houghThresh1=200, houghThresh2=65):
            img = img[15:img.shape[0]-15, 15:img.shape[1]-15]
            background = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
                
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
            
            edged = cv2.Canny(combined, 50, 300)
            
            kernel_h = np.ones((1, 5), np.uint8)
            kernel_v = np.ones((5, 1), np.uint8)
            thr_h = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel_h)
            thr_v = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel_v)
            
            lines1 = cv2.HoughLines(thr_h, 1, np.pi / 180, houghThresh1)
            lines2 = cv2.HoughLines(thr_v, 1, np.pi / 180, houghThresh2)
            
            lines = np.vstack((lines1, lines2))
            
            for line in lines:
                for rho,theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 3000*(-b))
                    y1 = int(y0 + 3000*(a))
                    x2 = int(x0 - 3000*(-b))
                    y2 = int(y0 - 3000*(a))
                    cv2.line(background,(x1,y1),(x2,y2),(0,255,0),1, lineType=cv2.LINE_AA)
                    
            
            segmented = segment_by_angle_kmeans(lines, 2)
            intersections = segmented_intersections(segmented)
            
            p1 = (0, 0)
            p2 = (0, 0)
            p3 = (0, 0)
            p4 = (0, 0)
            for pnt in intersections:
                pt = (pnt[0][0], pnt[0][1])
                
                if pt[0] < background.shape[1]//2-100:
                    if (p1 == (0, 0)) & (p3 == (0, 0)):
                        p1 = pt
                        p3 = pt
                        
                    if pt[1] < p1[1]:
                        p1 = pt
                    if pt[1] == p1[1]:
                        if pt[0] < p1[0]:
                            p1 = pt
                            
                    if pt[1] > p3[1]:
                        p3 = pt
                    if pt[1] == p3[1]:
                        if pt[0] < p3[0]:
                            p3 = pt
                if pt[0] > background.shape[1]//2+100:
                    if (p2 == (0, 0)) & (p4 == (0, 0)):
                        p2 = pt
                        p4 = pt
                        
                    if pt[1] < p2[1]:
                        p2 = pt
                    if pt[1] == p2[1]:
                        if pt[0] > p2[0]:
                            p2 = pt
                            
                    if pt[1] > p4[1]:
                        p4 = pt
                    if pt[1] == p4[1]:
                        if pt[0] > p4[0]:
                            p4 = pt
            
            pts = np.array([p1, p2, p3, p4])
            
            cv2.circle(background, p1, 3, (255, 0, 0), -1)
            cv2.circle(background, p2, 3, (255, 0, 0), -1)
            cv2.circle(background, p3, 3, (255, 0, 0), -1)
            cv2.circle(background, p4, 3, (255, 0, 0), -1)
            
            cropped = four_point_transform(img, pts)
            # cv2.imwrite(os.path.join(base_path, crop_path) + '/cropped-' + name, cropped)
            
            # plt.figure(figsize=(10,10))
            # plt.imshow(background, cmap='gray')
            # plt.imshow(cropped, cmap='gray')
            # plt.title(f'cropped {name}')
            # plt.show()
            # print(name, cropped.shape)
            
            return cropped
        
        if name.startswith('db'):
            result = area_detect(img, 50, 10, 175, 150)
            result = cv2.resize(result, self.size[f'{name}'])
            
        elif name.startswith('nh'):
            result = area_detect(img, 50, 10, 200, 65)
            result = cv2.resize(result, self.size[f'{name}'])
            
        elif (name.startswith('prudential')) or (name.startswith('shinhan')):
            result = area_detect(img, 50, 75, 200, 125)
            result = cv2.resize(result, self.size[f'{name}'])
            
        elif (name.startswith('kyobo')):
            result = area_detect(img, 50, 50, 170, 125)
            result = cv2.resize(result, self.size[f'{name}'])
            
        elif (name.startswith('miraeasset')):
            result = area_detect(img, 50, 50, 150, 125)
            result = cv2.resize(result, self.size[f'{name}'])
            
        elif (name.startswith('bnp')):
            result = area_detect(img, 50, 50, 125, 100)
            result = cv2.resize(result, self.size[f'{name}'])
            
        elif (name.startswith('samsung')):
            result = area_detect(img, 50, 40, 140, 100)
            result = cv2.resize(result, self.size[f'{name}'])
            
        elif (name.startswith('fubon')):
            result = area_detect(img, 50, 50, 200, 100)
            result = cv2.resize(result, self.size[f'{name}'])
            
        elif (name.startswith('abl')):
            result = area_detect(img, 50, 50, 175, 150)
            result = cv2.resize(result, self.size[f'{name}'])
            
        elif (name.startswith('aia')) or (name.startswith('kdb')):
            result = area_detect(img, 50, 50, 150, 250)
            result = cv2.resize(result, self.size[f'{name}'])
        
        else:
            result = area_detect(img, 50, 50, 150, 150)
            result = cv2.resize(result, self.size[f'{name}'])
        
        return result