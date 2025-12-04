from django.urls import path
from . import views

urlpatterns = [
    path("home/", views.sinistros_home_view, name="sinistros_home"),
    # No futuro: path("novo/", views.novo_sinistro, name="novo_sinistro"),
]