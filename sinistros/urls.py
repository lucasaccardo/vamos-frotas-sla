from django.urls import path
from . import views

urlpatterns = [
    # --- VIEWS (HTML) ---
    path("home/", views.sinistros_home_view, name="sinistros_home"),
    path("novo/", views.novo_sinistro_view, name="novo_sinistro"),
    path("processo/<int:pk>/", views.editar_sinistro_view, name="editar_sinistro"),
    path("dashboard/", views.dashboard_sinistros_view, name="dashboard_sinistros"),
    path('<int:pk>/history/', views.sinistro_history_view, name='sinistro_history'),

    # --- AÇÕES ---
    path("delete-selected/", views.delete_selected_sinistros, name="excluir_sinistros_selecionados"),
    path("delete-selected/", views.delete_selected_sinistros, name="delete_selected_sinistros"),

    # --- APIS EXISTENTES ---
    path("api/buscar-placa/", views.api_buscar_dados_sinistro, name="api_buscar_dados_sinistro"),

    # --- NOVAS ROTAS (DASHBOARD AVANÇADO E TIMELINE) ---
    path('api/dashboard-stats/', views.dashboard_stats_api, name='dashboard_stats_api'),
    path('api/dashboard-export/', views.dashboard_export_csv, name='dashboard_export_csv'),
    path('<int:pk>/timeline/', views.sinistro_timeline_api, name='sinistro_timeline_api'),
]
