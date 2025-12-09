from django.urls import path
from . import views

urlpatterns = [
    path("home/", views.sinistros_home_view, name="sinistros_home"),
    path("novo/", views.novo_sinistro_view, name="novo_sinistro"),
    path("processo/<int:pk>/", views.editar_sinistro_view, name="editar_sinistro"),
    path("dashboard/", views.dashboard_sinistros_view, name="dashboard_sinistros"),
    path("exportar/", views.exportar_sinistros_excel, name="exportar_sinistros_excel"),
    
    # API para o JavaScript chamar
    path("api/buscar-placa/", views.api_buscar_dados_sinistro, name="api_buscar_dados_sinistro"),
]