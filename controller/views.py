import json
import os
# from django.shortcuts import render

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from ai.settings import BASE_DIR
from utils.logging_time import logging_time
from .apps import ControllerConfig
from .models import Requested, Responsed
from .serializers import RequestedSerializer, ResponsedSerializer


class GetInformation(APIView):
    permission_classes = [AllowAny]
    serializer_class = RequestedSerializer

    @logging_time
    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)  # data 유효성 검사
        if serializer.is_valid():
            image = request.data.get('image')
            user = user = serializer.data.get('user')
            req = Requested(user=user,
                            image=image
                            )
            file_name, file_ext = os.path.splitext(image.name)
            coordinates = []
            pred_img = []
            # req.save()

            # Super Resolution 초해상화

            # Image Classification

            # Text Detection
            # 1번 방식: AI Hub Label 데이터 추출해서 Detection 하는 형식으로
            # 추출 -> static/images에 영역별로 분리해서 저장
            # 2번 방식: Text Detection demo버젼(손글씨, 인쇄글씨 모두 검출)

            # Inmemoryuploadedfile 을 이미지로 읽기
            img = cv2.imdecode(np.fromstring(
                image.read(), np.uint8), cv2.IMREAD_COLOR)

            # 1번 방식
            try:
                with open(os.path.join(BASE_DIR, 'static', 'json', file_name + '.json'), 'r', encoding='utf8') as f:
                    json_files = json.load(f)
                img_info = json_files['images'][0]

                # img = cv2.imread(image, cv2.IMREAD_COLOR)
                targets = json_files['annotations'][0]['polygons']

                for i, l in enumerate(targets):
                    if l['type'] != int(2):
                        continue
                    l['points'] = sorted(
                        l['points'], key=lambda x: (x[0], x[1]))
                    d1 = list(map(int, l['points'][0]))
                    d2 = list(map(int, l['points'][3]))
                    try:
                        sliced_img = img[d1[1]:d2[1], d1[0]:d2[0]]  # 높이, 너비
                        # sliced_img_name = img_info['identifier'] + \
                        #     f'_{i}.' + img_info['type']

                        # ndarray -> 이미지로 다시 전환해서 바로 보내기. (서버에 나눈 이미지 저장 후 불러오는게 X)
                        pred_img.append(Image.fromarray(sliced_img))
                        coordinates.append(l['points'])
                        # cv2.imwrite(os.path.join(BASE_DIR, 'static',
                        #             'images', image, sliced_img_name), sliced_img)
                    except:
                        pass

            # 2번 방식
            except:
                targets, _, _ = ControllerConfig.td.predict(img)

                for i, l in enumerate(targets):
                    # if l['type'] != int(2):
                    #     continue
                    # l['points'] = sorted(l['points'], key=lambda x: (x[0], x[1]))
                    # d1 = list(map(int, l['points'][0]))
                    # d2 = list(map(int, l['points'][3]))

                    l = sorted(l, key=lambda x: (x[0], x[1]))
                    d1 = list(map(int, l[0]))
                    d2 = list(map(int, l[3]))
                    try:
                        sliced_img = img[d1[1]:d2[1], d1[0]:d2[0]]  # 높이, 너비
                        # sliced_img_name = img_info['identifier'] + \
                        #     f'_{i}.' + img_info['type']

                        # ndarray -> 이미지로 다시 전환해서 바로 보내기. (서버에 나눈 이미지 저장 후 불러오는게 X)
                        pred_img.append(Image.fromarray(sliced_img))
                        coordinates.append(l)
                        # cv2.imwrite(os.path.join(BASE_DIR, 'static',
                        #             'images', image, sliced_img_name), sliced_img)
                    except:
                        pass

            # Text Recognition : 최대 predict 이미지 개수 500개
            result = [{'name': file_name + file_ext,
                       'coordinates': coordinates, 'result': []}]
            result[0]['result'] = ControllerConfig.tr.predict(
                file_name, pred_img)

            res = Responsed(user=user, result=result)

            # 이미지에 결과 그려서 확인
            try:
                img2 = np.ones((img.shape[0], img.shape[1], 3), np.uint8) * 255
                for i in range(len(coordinates)):
                    points = np.array([list(map(int, p))
                                       for p in coordinates[i]]).astype(np.int32)
                    txt = result[0]['result'][0][i]
                    center = [(points[3][0] + points[0][0]) // 2,
                              (points[3][1] + points[0][1]) // 2]
                    m = points[3][0] - center[0]
                    area = ((points[3][0] - points[0][0]) *
                            (points[3][1] - points[0][1])) // 150
                    cv2.rectangle(img, points[0], points[3],
                                  (255, 0, 255), 2, cv2.LINE_AA)
                    b, g, r, a = 100, 100, 250, 0
                    fontpath = "fonts/gulim.ttc"
                    font = ImageFont.truetype(fontpath, int(area))
                    img_pil = Image.fromarray(img2)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((center[0], center[1]),
                              txt, font=font, fill=(b, g, r, a))
                    img2 = np.array(img_pil)

                img = cv2.resize(img, (800, 1000))
                img2 = cv2.resize(img2, (800, 1000))

                cv2.imshow('img', img)  # [:, :, ::-1])
                cv2.imshow('img2', img2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                pass

            return Response(ResponsedSerializer(res).data, status=status.HTTP_200_OK)
        return Response({'Bad Request': 'Invalid Data..'}, status=status.HTTP_400_BAD_REQUEST)
