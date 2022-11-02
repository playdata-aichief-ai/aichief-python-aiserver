import os
from django.db import models
from ai.settings.settings import BASE_DIR


def upload_to(instance, filename):
    print(BASE_DIR)
    return os.path.join(BASE_DIR, 'static', 'images', filename).format(filename=filename)


class Requested(models.Model):
    user = models.CharField(null=False, max_length=50)
    image = models.ImageField(verbose_name='image',
                              null=True, blank=True, upload_to=upload_to)


class Responsed(models.Model):
    user = models.CharField(null=False, max_length=50)
    result = models.CharField(null=False, max_length=500)


class ProcessLog(models.Model):
    user = models.CharField(null=False, max_length=50)
    img_path = models.CharField(null=False, max_length=500)
    finished = models.CharField(null=False, max_length=500)
    finished_time = models.DateTimeField(auto_now_add=True)
