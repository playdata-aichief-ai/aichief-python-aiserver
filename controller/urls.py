from django.urls import path
from .views import *

urlpatterns = [
    path('get-information/', GetInformation.as_view()),
    path('get-processes/', GetProcessLog.as_view()),
    path('click-processes/', ClickProcessLog.as_view()),
]
