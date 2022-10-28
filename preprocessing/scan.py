import numpy as np
import requests
import cv2
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform

class Scan():
    def __init__(self):
        pass
    
    def scan(self, url):
        image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
        img = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)

        if (img.shape[0] > 2000) or (img.shape[1] > 1200):
            img = cv2.resize(img, (810, 1440), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 350)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
            if len(approx) == 4:
                doc_cnts = approx
                break

        scanned_img = four_point_transform(img, doc_cnts.reshape(4, 2))

        # 2차 시도
        if (scanned_img.shape[1] < 600) or (scanned_img.shape[0] < 900):
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            blur = cv2.bilateralFilter(blur, -1, 10, 10)
            
            img_YUV = cv2.cvtColor(blur, cv2.COLOR_BGR2YUV)  
            y = img_YUV[:,:,0]    

            rows = y.shape[0]    
            cols = y.shape[1]

            imgLog = np.log1p(np.array(y, dtype='float') / 255) # y값을 0~1사이로 조정한 뒤 log(x+1)

            M = 2*rows + 1
            N = 2*cols + 1

            sigma = 30
            (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M)) # 0~N-1(and M-1) 까지 1단위로 space를 만듬
            Xc = np.ceil(N/2) # 올림 연산
            Yc = np.ceil(M/2)
            gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2 # 가우시안 분자 생성

            LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))
            HPF = 1 - LPF

            LPF_shift = np.fft.ifftshift(LPF.copy())
            HPF_shift = np.fft.ifftshift(HPF.copy())

            img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
            img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N))) # low frequency 성분
            img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N))) # high frequency 성분

            gamma1 = 10.0
            gamma2 = 16.0
            img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]

            img_exp = np.expm1(img_adjusting) # exp(x) + 1
            img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) # 0~1사이로 정규화
            img_out = np.array(255*img_exp, dtype = 'uint8') # 255를 곱해서 intensity값을 만들어줌

            img_YUV[:,:,0] = img_out
            result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
            
            saturationScale = 4.0  
            hsvImage = cv2.cvtColor(result , cv2.COLOR_BGR2HSV)
            hsvImage = np.float32(hsvImage)
            H, S, V = cv2.split(hsvImage)    # 분리됨
            S = np.clip( S * saturationScale , 0,255 ) # 계산값, 최소값, 최대값
            S = np.uint8(S)
            
            kernal = np.ones((3,3), np.uint8)
            dilate = cv2.dilate(S, kernal, iterations=3)
            edged = cv2.Canny(dilate, 50, 300)

            contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for cnt in contours:

                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if  len(approx) == 4:
                    doc_cnts = approx
                    break
                
            scanned_img = four_point_transform(img, doc_cnts.reshape(4, 2))
                
            # 3차 시도
            if (scanned_img.shape[1] < 600) or (scanned_img.shape[0] < 900):
                img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  
                y = img_YUV[:,:,0]    

                rows = y.shape[0]    
                cols = y.shape[1]

                imgLog = np.log1p(np.array(y, dtype='float') / 255) # y값을 0~1사이로 조정한 뒤 log(x+1)

                M = 2*rows + 1
                N = 2*cols + 1

                sigma = 10
                (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M)) # 0~N-1(and M-1) 까지 1단위로 space를 만듬
                Xc = np.ceil(N/2) # 올림 연산
                Yc = np.ceil(M/2)
                gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2 # 가우시안 분자 생성

                LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))
                HPF = 1 - LPF

                LPF_shift = np.fft.ifftshift(LPF.copy())
                HPF_shift = np.fft.ifftshift(HPF.copy())

                img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
                img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N))) # low frequency 성분
                img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N))) # high frequency 성분

                gamma1 = 0.3
                gamma2 = 1.5
                img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]

                img_exp = np.expm1(img_adjusting) # exp(x) + 1
                img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) # 0~1사이로 정규화
                img_out = np.array(255*img_exp, dtype = 'uint8') # 255를 곱해서 intensity값을 만들어줌

                img_YUV[:,:,0] = img_out
                result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)

                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) 
                gblur = cv2.GaussianBlur(gray, (5, 5), 0)
                blur = cv2.bilateralFilter(gblur, -1, 10, 10)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl1 = clahe.apply(blur)

                edged = cv2.Canny(cl1, 50, 300)

                background = np.zeros((edged.shape[0], edged.shape[1], 3), dtype = "uint8")
                background2 = background.copy()

                contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                for cnt in contours:
                    if cv2.arcLength(cnt, closed=True) < 2800:
                        continue

                    epsilon = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    if  len(approx) >= 3:
                        approx2 = approx
                        cv2.drawContours(background, [approx2], 0, (255, 255, 255), 1, lineType=cv2.LINE_AA)

                edged2 = cv2.Canny(background, 50, 300)

                contours, _ = cv2.findContours(edged2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                for cnt in contours:
                    if cv2.arcLength(cnt, closed=True) < 2800:
                        continue
                    hull = cv2.convexHull(cnt)
                    peri = cv2.arcLength(hull, True)
                    approx = cv2.approxPolyDP(hull, 0.05 * peri, True)
                    if len(approx) == 4:
                        cv2.drawContours(background2, [approx], 0, (255, 0, 255), 1, lineType=cv2.LINE_AA)

                edged3 = cv2.Canny(background2, 50, 300)

                contours, _ = cv2.findContours(edged3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                for cnt in contours:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
                    if len(approx) == 4:
                        doc_cnts = approx
                        break
                    
                scanned_img = four_point_transform(img, doc_cnts.reshape(4, 2))

                # 4차 시도
                if (scanned_img.shape[1] < 600) or (scanned_img.shape[0] < 900):
                    blur = cv2.GaussianBlur(img, (5, 5), 0)
                    blur = cv2.bilateralFilter(blur, -1, 20, 20)
                    
                    img_YUV = cv2.cvtColor(blur, cv2.COLOR_BGR2YUV)  
                    y = img_YUV[:,:,0]    

                    rows = y.shape[0]    
                    cols = y.shape[1]

                    imgLog = np.log1p(np.array(y, dtype='float') / 255) # y값을 0~1사이로 조정한 뒤 log(x+1)

                    M = 2*rows + 1
                    N = 2*cols + 1

                    sigma = 50
                    (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M)) # 0~N-1(and M-1) 까지 1단위로 space를 만듬
                    Xc = np.ceil(N/2) # 올림 연산
                    Yc = np.ceil(M/2)
                    gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2 # 가우시안 분자 생성

                    LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))
                    HPF = 1 - LPF

                    LPF_shift = np.fft.ifftshift(LPF.copy())
                    HPF_shift = np.fft.ifftshift(HPF.copy())

                    img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
                    img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N))) # low frequency 성분
                    img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N))) # high frequency 성분

                    gamma1 = 0.1
                    gamma2 = 2.0

                    img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]

                    img_exp = np.expm1(img_adjusting) # exp(x) + 1
                    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) # 0~1사이로 정규화
                    img_out = np.array(255*img_exp, dtype = 'uint8') # 255를 곱해서 intensity값을 만들어줌

                    img_YUV[:,:,0] = img_out
                    result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
                    
                    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

                    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
                    morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k)
                    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, k)

                    _, thr = cv2.threshold(morph, 10, 255, cv2.THRESH_BINARY)

                    edged = cv2.Canny(thr, 50, 300)

                    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
                    dilate = cv2.dilate(edged, k, iterations=1)

                    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)

                    for cnt in contours:
                        if cv2.arcLength(cnt, closed=True) < 2800:
                            continue
                        
                        hull = cv2.convexHull(cnt)
                        peri = cv2.arcLength(hull, True)
                        approx = cv2.approxPolyDP(hull, 0.012 * peri, True)
                        if len(approx) == 4:
                            doc_cnts = approx
                            break
                    
                    scanned_img = four_point_transform(img, doc_cnts.reshape(4, 2))
            
        # cv2.imwrite(os.path.join(base_path, scan_path) + '/scanned_img-' + file, scanned_img)
        # plt.figure(figsize=(3,3))
        # plt.imshow(scanned_img, cmap='gray')
        # plt.title(f'{file}')
        # plt.show()
        
        return scanned_img