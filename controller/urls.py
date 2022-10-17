from django.urls import path
from .views import *

urlpatterns = [
    path('get-information/', GetInformation.as_view()),
]
