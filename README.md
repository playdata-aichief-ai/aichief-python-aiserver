# ocr-python-ai-server

Web Server와 연동할 AI Server(OCR)

# BEST MODEL LINK

## 다운로드 후, 각각 text_recognition/saved_models, text_detection/saved_models안에 넣기

- text_recognition 모델 (accuracy 62%, 문자최대길이 20자 이내(띄어쓰기 포함) 학습모델)
  https://drive.google.com/file/d/1D1HugvAF_iTkNLMmFGaFSgeFAPA949Ca/view?usp=sharing

- text_recognition 모델 (accuracy 95%, 어절 단위(띄어쓰기 학습 X) 학습모델)
  https://drive.google.com/file/d/1-MrQx6ZIEegdB1aFoEfiY8WvIxmAoYYE/view?usp=sharing

- text_detection 모델(craft_mlt_25k/pretrained 모델)
  https://drive.google.com/file/d/1LG_UxK_dwMagXZHg4fcPia6LoiAjTJqM/view?usp=sharing

## 실행

<!-- python manage.py runserver --settings=ai.settings.settings -->

python manage.py runserver
