import json
import os
# from django.shortcuts import render

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny

import cv2
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from ai.settings.settings import BASE_DIR
from utils.logging_time import logging_time
from .apps import ControllerConfig
from .models import Requested, Responsed
from .serializers import RequestedSerializer, ResponsedSerializer
from preprocessing.box_detection import Box_Detection
from preprocessing.crop import Crop
from preprocessing.scan import Scan

box_detector = Box_Detection()
cropper = Crop()
scanner = Scan()


class GetInformation(APIView):
    permission_classes = [AllowAny]
    serializer_class = RequestedSerializer

    def re_crop_detection(self, img):
        targets, _, _ = ControllerConfig.td.predict(img)
        min_h = 999999
        max_h = -999999
        min_w = 999999
        max_w = -999999
        for i, l in enumerate(targets):
            l = sorted(l, key=lambda x: (x[0], x[1]))
            d1 = list(map(int, l[0]))
            d2 = list(map(int, l[3]))
            min_h = min(min_h, d1[1])
            max_h = max(max_h, d2[1])
            min_w = min(min_w, d1[0])
            max_w = max(max_w, d2[0])
        try:
            sliced_img = img[min_h:max_h, min_w:max_w]  # 높이, 너비
        except:
            pass

        try:
            for ci in sliced_img:
                cv2.imshow('crop_img', ci)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except:
            pass
        return sliced_img

    @logging_time
    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)  # data 유효성 검사
        if serializer.is_valid():
            contract_id = request.data.get('contract_id')
            company = request.data.get('company')
            image_path = request.data.get('image_path')
            image = request.data.get('image')
            user = user = serializer.data.get('user')
            req = Requested(user=user,
                            image=image
                            )

            file_name, file_ext = os.path.splitext(image.name)
            coordinates = []
            pred_img = []

            # 미구현
            # img url에서 s3 img file read

            # Inmemoryuploadedfile 을 이미지로 읽기
            img = cv2.imdecode(np.fromstring(
                image.read(), np.uint8), cv2.IMREAD_COLOR)

            # Image Scan
            scanned_img = scanner.scan(img)
            print('finished scan')

            # Image Crop
            cropped_img = cropper.crop(scanned_img)
            print('finished crop')

            # Image Classification
            category = ControllerConfig.cf.classify(scanned_img)
            print('finished classification', category)

            # Area detection
            img_dic = box_detector.box_detect(
                name=category, cropped_image=cropped_img)

            print('finished box detect')

            # Super Resolution
            sr_img_dic = ControllerConfig.sr.inference(img_dic)
            print('finished super resolution')

            img_key = list(sr_img_dic.keys())
            img_values = []

            # Text Detection
            for k in img_key:
                re_cropped_img = self.re_crop_detection(sr_img_dic[k])
                try:
                    re_cropped_img = Image.fromarray(re_cropped_img)
                except:
                    re_cropped_img = Image.fromarray(sr_img_dic[k])
                img_values.append(re_cropped_img)

            # crop_img = self.re_crop_detection(img)
            # try:
            #     crop_img = list(map(Image.fromarray, crop_img))
            # except:
            #     crop_img = [Image.fromarray(img)]

            # Text Recognition : 최대 predict 이미지 개수 500개

            result = [{'contract_id': contract_id,
                       'image_path': image_path,
                       'result': {}}]
            predict_result = ControllerConfig.tr.predict(
                file_name, img_values)

            for i in range(len(img_key)):
                result[0]['result'][img_key[i]] = predict_result[0][i]
            res = Responsed(user=user, result=result)

            # req.save()

            return Response(ResponsedSerializer(res).data, status=status.HTTP_200_OK)
        return Response({'Bad Request': 'Invalid Data..'}, status=status.HTTP_400_BAD_REQUEST)


################ no use########################
        # 1번 방식: AI Hub Label 데이터 추출해서 Detection 하는 형식으로
        # 추출 -> static/images에 영역별로 분리해서 저장
        # 2번 방식: Text Detection demo버젼(손글씨, 인쇄글씨 모두 검출)

        # # 1번 방식
        # try:
        #     with open(os.path.join(BASE_DIR, 'static', 'json', file_name + '.json'), 'r', encoding='utf8') as f:
        #         json_files = json.load(f)
        #     img_info = json_files['images'][0]

        #     # img = cv2.imread(image, cv2.IMREAD_COLOR)
        #     targets = json_files['annotations'][0]['polygons']

        #     for i, l in enumerate(targets):
        #         if l['type'] != int(2):
        #             continue
        #         l['points'] = sorted(
        #             l['points'], key=lambda x: (x[0], x[1]))
        #         d1 = list(map(int, l['points'][0]))
        #         d2 = list(map(int, l['points'][3]))
        #         try:
        #             sliced_img = img[d1[1]:d2[1], d1[0]:d2[0]]  # 높이, 너비
        #             # sliced_img_name = img_info['identifier'] + \
        #             #     f'_{i}.' + img_info['type']

        #             # ndarray -> 이미지로 다시 전환해서 바로 보내기. (서버에 나눈 이미지 저장 후 불러오는게 X)
        #             pred_img.append(Image.fromarray(sliced_img))
        #             coordinates.append(l['points'])
        #             # cv2.imwrite(os.path.join(BASE_DIR, 'static',
        #             #             'images', image, sliced_img_name), sliced_img)
        #         except:
        #             pass

        # # 2번 방식
        # except:
        #     targets, _, _ = ControllerConfig.td.predict(img)

        #     for i, l in enumerate(targets):
        #         # if l['type'] != int(2):
        #         #     continue
        #         # l['points'] = sorted(l['points'], key=lambda x: (x[0], x[1]))
        #         # d1 = list(map(int, l['points'][0]))
        #         # d2 = list(map(int, l['points'][3]))

        #         l = sorted(l, key=lambda x: (x[0], x[1]))
        #         d1 = list(map(int, l[0]))
        #         d2 = list(map(int, l[3]))
        #         try:
        #             sliced_img = img[d1[1]:d2[1], d1[0]:d2[0]]  # 높이, 너비
        #             # sliced_img_name = img_info['identifier'] + \
        #             #     f'_{i}.' + img_info['type']

        #             # ndarray -> 이미지로 다시 전환해서 바로 보내기. (서버에 나눈 이미지 저장 후 불러오는게 X)
        #             pred_img.append(Image.fromarray(sliced_img))
        #             coordinates.append(l)
        #             # cv2.imwrite(os.path.join(BASE_DIR, 'static',
        #             #             'images', image, sliced_img_name), sliced_img)
        #         except:
        #             pass


# 이미지에 결과 그려서 확인. local 작업시 편의 위한 기능. 배포시 주석
        # try:
        #     img2 = np.ones((img.shape[0], img.shape[1], 3), np.uint8) * 255
        #     for i in range(len(coordinates)):
        #         points = np.array([list(map(int, p))
        #                            for p in coordinates[i]]).astype(np.int32)
        #         txt = result[0]['result'][0][i]
        #         center = [(points[3][0] + points[0][0]) // 2,
        #                   (points[3][1] + points[0][1]) // 2]
        #         m = points[3][0] - center[0]
        #         area = ((points[3][0] - points[0][0]) *
        #                 (points[3][1] - points[0][1])) // 150
        #         cv2.rectangle(img, points[0], points[3],
        #                       (255, 0, 255), 2, cv2.LINE_AA)
        #         b, g, r, a = 100, 100, 250, 0
        #         fontpath = "fonts/gulim.ttc"
        #         font = ImageFont.truetype(fontpath, int(area))
        #         img_pil = Image.fromarray(img2)
        #         draw = ImageDraw.Draw(img_pil)
        #         draw.text((center[0], center[1]),
        #                   txt, font=font, fill=(b, g, r, a))
        #         img2 = np.array(img_pil)

        #     img = cv2.resize(img, (800, 1000))
        #     img2 = cv2.resize(img2, (800, 1000))

        #     cv2.imshow('img', img)  # [:, :, ::-1])
        #     cv2.imshow('img2', img2)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # except:
        #     pass
