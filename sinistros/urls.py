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
    # Mantive as duas rotas apontando para a mesma view conforme seu código original
    # (Isso permite usar ambos os 'names' nos templates sem erro)
    path("delete-selected/", views.delete_selected_sinistros, name="excluir_sinistros_selecionados"),
    path("delete-selected/", views.delete_selected_sinistros, name="delete_selected_sinistros"),

    # --- APIS EXISTENTES ---
    path("api/buscar-placa/", views.api_buscar_dados_sinistro, name="api_buscar_dados_sinistro"),

    # --- NOVAS ROTAS (DASHBOARD AVANÇADO E TIMELINE) ---
    path('api/dashboard-stats/', views.dashboard_stats_api, name='dashboard_stats_api'),
    path('api/dashboard-export/', views.dashboard_export_csv, name='dashboard_export_csv'),
    path('<int:pk>/timeline/', views.sinistro_timeline_api, name='sinistro_timeline_api'),
    
    # --- EXPORT / RELATÓRIOS ---
    path('relatorios/', views.exportar_relatorios_view, name='exportar_relatorios'),
    path('export/xlsx/', views.exportar_xlsx, name='exportar_xlsx'),
    path('export/csv/', views.exportar_csv, name='exportar_csv'),
    
    # --- CUSTOM SECTORS (ADMIN ONLY) ---
    path('admin/criar-setor/', views.criar_setor_customizado_view, name='criar_setor_customizado'),
    path('admin/desativar-setor/<int:sector_id>/', views.desativar_setor_customizado_view, name='desativar_setor_customizado'),
]
