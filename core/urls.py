# core/urls.py
from django.urls import path
from .views import predict_view, register_view

urlpatterns = [
    path("", predict_view, name="predict"),
    path("register/", register_view, name="register"),
]
