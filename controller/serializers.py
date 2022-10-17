from dataclasses import field
from operator import mod
from attr import fields
from rest_framework import serializers
from rest_framework.parsers import MultiPartParser, FormParser

from .models import Requested, Responsed


class RequestedSerializer(serializers.ModelSerializer):
    image = serializers.ImageField(required=False)

    class Meta:
        model = Requested
        fields = ('__all__')
        parser_classes = (MultiPartParser, FormParser)


class ResponsedSerializer(serializers.ModelSerializer):
    image = serializers.ImageField(required=False)

    class Meta:
        model = Responsed
        fields = ('__all__')
        parser_classes = (MultiPartParser, FormParser)
