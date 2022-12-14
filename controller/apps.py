from django.apps import AppConfig
from classification.classification import Classification
from text_detection.text_detection import Text_Detection

from text_recognition.text_recognition import Text_Recognition
from super_resolution.super_resolution import Super_Resolution
from text_detection_yolo.text_detection_yolo import Text_Detection_Yolo

class ControllerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "controller"
    sr = Super_Resolution(light=True)
    td = Text_Detection()
    tr = Text_Recognition()
    cf = Classification()
    td_yolo = Text_Detection_Yolo()
