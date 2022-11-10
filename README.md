# ocr-python-ai-server

Web Server와 연동할 AI Server(OCR)

# BEST MODEL LINK

## text_recognition saved_models(path: text_recognition/saved_models)

- text_recognition 모델 (accuracy 95%, 문자최대길이 20자 이내(띄어쓰기 포함) 학습모델)
  https://drive.google.com/file/d/1W7Y90G3YFED75D1wy-cI6t11QwhXaR-f/view?usp=share_link

## text_detection saved_models(path: text_detection/saved_models)

- text_detection 모델(craft_mlt_25k/pretrained 모델)
  https://drive.google.com/file/d/1LG_UxK_dwMagXZHg4fcPia6LoiAjTJqM/view?usp=sharing

## text_detection yolov5(path: text_detection_yolo/models/best_yolo5x.pt)

https://drive.google.com/drive/folders/1YuA5CrXsoS9WYELPSsJTOSfpxk0l_Qsb?usp=sharing

## Super-resolution saved_models(path: super_resolution/saved_models)

- SwinIR GPP loss, ligtweight, grayscale(ours)  
  https://drive.google.com/file/d/1j4K-7LI3ga-BDtTkSQDKLq24G9NbxEzV/view?usp=share_link

- SwinIR perceptual loss, lightweight, grayscale(ours)  
  https://drive.google.com/file/d/1Vh9D5ZZsxc89SVbHPZJi5xQ4SwNWw0FJ/view?usp=share_link
- SwinIR pretrined model(for RGB Image, pretrained model)  
  https://drive.google.com/file/d/1gV8s9MQ8HKMLOjIj0nQ0Gwa-lgQ3hmoh/view?usp=share_link

## Classification saved_models(path: classification/saved_models)

- CNN Document Image Classification (VGG16 backbone + classifier head) - acc: 0.97 loss: 0.14
  https://drive.google.com/file/d/1oD6L3yco63L8uksK2pL8DXGrrRUb8pyt/view

## 실행

python manage.py runserver

<!-- python manage.py runserver --settings=ai.settings.settings -->
