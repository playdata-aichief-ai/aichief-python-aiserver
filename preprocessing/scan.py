import numpy as np
import requests
import cv2
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform


class Scan():
    def __init__(self):
        pass

    def scan(self, img):
        if (img.shape[0] > 2000) or (img.shape[1] > 1200):
            img = cv2.resize(img, (810, 1440), interpolation=cv2.INTER_AREA)

        background = np.zeros((img.shape[0], img.shape[1], 3), dtype = "uint8")
        
        img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  
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

        gamma1 = 5.0
        gamma2 = 5.0
        img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]

        img_exp = np.expm1(img_adjusting) # exp(x) + 1
        img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) # 0~1사이로 정규화
        img_out = np.array(255*img_exp, dtype = 'uint8') # 255를 곱해서 intensity값을 만들어줌

        img_YUV[:,:,0] = img_out
        hf = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
        
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (20,20,img.shape[1]-20,img.shape[0]-20)
        cv2.setRNGSeed(0)
        cv2.grabCut(hf,mask,rect,bgdModel,fgdModel,15,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        gc = img*mask2[:,:,np.newaxis]
        
        k = np.ones((3, 15), np.uint8)
        mor = cv2.morphologyEx(gc, cv2.MORPH_CLOSE, k, iterations=3)
        
        blur = cv2.GaussianBlur(mor, (5,5), 0)
        
        edged = cv2.Canny(blur, 30, 150)
        
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            if cv2.arcLength(cnt, True) < 2000:
                continue
            hull = cv2.convexHull(cnt)
            epsilon = 0.05 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if cv2.arcLength(approx, True) < 2000:
                continue

            if len(approx) == 4:
                outlier = False
                doc_cnts = None
                for i in range(0,len(approx)):
                    if (approx[i][0][0] > background.shape[1]-20).any() or (approx[i][0][0] < 20).any() or (approx[i][0][1] < 10).any() or (approx[i][0][1] > background.shape[0]-10).any():
                        outlier = True
                if outlier == False:
                    doc_cnts = approx
                    cv2.drawContours(background, [doc_cnts], 0, (255, 0, 0), 1, lineType=cv2.LINE_AA)
                    break
        
        scanned = four_point_transform(img, doc_cnts.reshape(4, 2))

        return scanned