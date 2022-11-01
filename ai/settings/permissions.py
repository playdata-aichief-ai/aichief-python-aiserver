from rest_framework import permissions

from ai.settings.settings import ALLOWED_IP_ADDRESSES

ALLOWED_IP_ADDRESSES = ALLOWED_IP_ADDRESSES


class IPBasedPermission(permissions.BasePermission):
    def has_permission(self, request, view):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip_address = x_forwarded_for.split(',')[0]
        else:
            ip_address = request.META.get('REMOTE_ADDR')

        return ip_address in ALLOWED_IP_ADDRESSES

    def has_object_permission(self, request, view):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip_address = x_forwarded_for.split(',')[0]
        else:
            ip_address = request.META.get('REMOTE_ADDR')

        return ip_address in ALLOWED_IP_ADDRESSES
