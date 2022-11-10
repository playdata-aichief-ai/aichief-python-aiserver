import json
import os
# from django.shortcuts import render

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from ai.settings.permissions import IPBasedPermission
from aws.download.utils import AWSDownload

import cv2
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from text_detection_yolo.text_detection_yolo import Text_Detection_Yolo
from ai.settings.settings import BASE_DIR, AWS_STORAGE_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3DIRECT_REGION, PROTECTED_DIR_NAME, PROTECTED_MEDIA_URL, AWS_STORAGE_BUCKET_NAME, PROTECTED_DIR_NAME, AWS_DOWNLOAD_EXPIRE
from .apps import ControllerConfig
from .models import Requested, Responsed, ProcessLog
from .serializers import RequestedSerializer, ResponsedSerializer, ProcessLogSerializer
from preprocessing.box_detection import Box_Detection
from preprocessing.crop import Crop
from preprocessing.scan import Scan

box_detector = Box_Detection()
cropper = Crop()
scanner = Scan()


class GetInformation(APIView):
    permission_classes = [IPBasedPermission]
    serializer_class = RequestedSerializer

    def convert_value_format_from_key(self, result, k, v):
        if "SocialSecurityNumber" in k:
            try:
                v2 = result[k] + v
                result[k] = v2
            except:
                v2 = v + "-"
                result[k] = v2

        elif "PhoneNumber" in k:
            try:
                v2 = result[k] + "-" + v
                result[k] = v2
            except:
                v2 = v + "-"
                result[k] = v2
        elif "Email" in k:
            try:
                v2 = result[k] + "." + v
                result[k] = v2
            except:
                v2 = v + "@"
                result[k] = v2
        elif "DateTime" in k:
            try:
                v2 = result[k] + "-" + v
                result[k] = v2
            except:
                v2 = v + "-"
                result[k] = v2
        else:
            try:
                v2 = result[k] + " " + v
                result[k] = v2
            except:
                v2 = v
                result[k] = v2
        return

    def get_image_from_s3(self, img_path, download=False):
        bucket = AWS_STORAGE_BUCKET_NAME
        region = S3DIRECT_REGION
        access_key = AWS_ACCESS_KEY_ID
        secret_key = AWS_SECRET_ACCESS_KEY
        target_img_path = img_path
        file_name = img_path.split('/')[-1]
        aws_download = AWSDownload(access_key, secret_key, bucket, region)
        s3 = aws_download.s3connect()
        if download:
            save2 = os.path.join(BASE_DIR, file_name)
            return aws_download.download_file(s3, target_img_path, save2)
        else:
            return aws_download.read_image_from_s3(s3, target_img_path)

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

        return sliced_img

    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)  # data 유효성 검사

        if serializer.is_valid():
            contract_id = request.data.get('contract_id')
            company = request.data.get('company')
            image_path = request.data.get('image_path')

            user = user = serializer.data.get('user')
            file_name = file_name = image_path.split('/')[-1]

            # img url에서 s3 img file read or 전달받은 Inmemoryuploadedfile 을 이미지로 읽기
            try:
                img = np.asarray(Image.open(
                    self.get_image_from_s3(image_path)), dtype=np.uint8)
            except:
                img = cv2.imdecode(np.fromstring(
                    request.data.get('image').read(), np.uint8), cv2.IMREAD_COLOR)

            # Image Scan
            scanned_img = scanner.scan(img)
            print('finished scan')
            ProcessLog(user=user, img_path=image_path, finished='scan').save()

            # Image Classification
            category = ControllerConfig.cf.classify(scanned_img)
            print('finished classification', category)
            ProcessLog(user=user, img_path=image_path,
                       finished='classification').save()

            # Image Crop
            cropped_img = cropper.crop(category, scanned_img)
            print('finished crop')
            ProcessLog(user=user, img_path=image_path, finished='crop').save()

            # Area detection
            img_dic = box_detector.box_detect(
                name=category, cropped_image=cropped_img)
            print('finished box detect')
            ProcessLog(user=user, img_path=image_path,
                       finished='box_detect').save()

            # Super Resolution
            sr_img_dic = ControllerConfig.sr.inference(img_dic, light=True)
            print('finished super resolution')
            ProcessLog(user=user, img_path=image_path,
                       finished='super_resolution').save()

            # Text Detection Yolov5x
            yolo_cropped_img_dic = ControllerConfig.td_yolo.predict(sr_img_dic)
            print('finished text detection(yolo)')
            ProcessLog(user=user, img_path=image_path,
                       finished='yolo_crop').save()

            # Text Recognition : 최대 predict 이미지 개수 500개
            img_key = list(yolo_cropped_img_dic.keys())

            result_key = []
            result_value = []
            for i in range(len(img_key)):
                for v in yolo_cropped_img_dic[img_key[i]]:
                    result_key.append(img_key[i])
                    result_value.append(Image.fromarray(v))
            recogntion_result = ControllerConfig.tr.predict(
                file_name, result_value)[0]
            ProcessLog(user=user, img_path=image_path,
                       finished='recognition').save()

            result = {}
            for i in range(len(result_key)):
                self.convert_value_format_from_key(
                    result, result_key[i], recogntion_result[i])
                # v = result.get(result_key[i]) or ""
                # v += " " + recogntion_result[i]
                # result[result_key[i]] = v

            res = Responsed(user=user, contractId=contract_id,
                            imagePath=image_path, result=result)

            # # test image save
            cv2.imwrite('./result/scan.jpg', scanned_img)
            # Crop
            cv2.imwrite('./result/crop.jpg', cropped_img)
            # Area_detection
            for k, i in img_dic.items():
                cv2.imwrite(f'./bd/bd{k}.jpg', i)
            for k, i in sr_img_dic.items():
                cv2.imwrite(f'./result/SR{k}.jpg', i)
            # Text Detection
            for k, l in yolo_cropped_img_dic.items():
                for idx, i in enumerate(l):
                    cv2.imwrite(f'./result/TD{k}{idx}.jpg', i)

            return Response(ResponsedSerializer(res).data, status=status.HTTP_200_OK)
        return Response({'Bad Request': 'Invalid Data..'}, status=status.HTTP_400_BAD_REQUEST)


class GetProcessLog(APIView):
    permission_classes = [IPBasedPermission]

    def get(self, request, format=None):
        processes = ProcessLogSerializer(
            ProcessLog.objects.filter(user=request.GET.get('id')).order_by('-finished_time'), many=True).data
        if len(processes) > 0:
            return Response(processes, status=status.HTTP_200_OK)
        return Response({'Processes Not Found': 'Invalid Request'}, status=status.HTTP_400_BAD_REQUEST)


class ClickProcessLog(APIView):
    permission_classes = [IPBasedPermission]

    def post(self, request, format=None):
        print(request.data)
        pl = ProcessLog.objects.filter(
            user=request.data['user'], id=request.data['notification'])[0]
        if pl != None:
            pl.view_count = pl.view_count + 1
            pl.save()
            return Response(status=status.HTTP_200_OK)
        return Response({'Processes Not Found': 'Invalid Request'}, status=status.HTTP_400_BAD_REQUEST)
