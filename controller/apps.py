from django.apps import AppConfig
from text_detection.text_detection import Text_Detection

from text_recognition.text_recognition import Text_Recognition


class ControllerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "controller"
    td = Text_Detection()
    tr = Text_Recognition()
