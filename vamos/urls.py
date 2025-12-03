from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # === Autenticação ===
    path("login/", views.login_view, name="login"),
    path("signup/", views.signup_view, name="signup"),
    path("logout/", views.logout_view, name="logout"),

    # === Reset de Senha ===
    path('reset_password/', 
         auth_views.PasswordResetView.as_view(
             template_name="vamos/password_reset.html",
             html_email_template_name="vamos/password_reset_email.html"
         ), 
         name='reset_password'
    ),
    path('reset_password_sent/', auth_views.PasswordResetDoneView.as_view(template_name="vamos/password_reset_sent.html"), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name="vamos/password_reset_form.html"), name='password_reset_confirm'),
    path('reset_password_complete/', auth_views.PasswordResetCompleteView.as_view(template_name="vamos/password_reset_done.html"), name='password_reset_complete'),

    # === PORTAL & SELEÇÃO DE MÓDULO (NOVO) ===
    path("", views.portal_view, name="portal"),  # A Raiz agora é o Portal
    path("selecionar-modulo/<str:modulo>/", views.selecionar_modulo, name="selecionar_modulo"),

    # === MÓDULO MANUTENÇÃO ===
    path("manutencao/home/", views.manutencao_home_view, name="manutencao_home"), # Antiga Home
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("sla-mensal/", views.sla_mensal_view, name="sla_mensal"),
    path("cenarios/", views.cenarios_view, name="cenarios"),
    path("buscar-clientes/", views.buscar_clientes_view, name="buscar_clientes"),
    path("analises/", views.analise_list_view, name="lista_analises"), # Histórico costuma ficar na manutenção
    
    # === MÓDULO SINISTROS (NOVO) ===
    path("sinistros/home/", views.sinistros_home_view, name="sinistros_home"),

    # === Funcionalidades Gerais / Admin ===
    path("assistente-ia/", views.assistente_ia_view, name="assistente_ia"),
    path("gestao/upload-bases/", views.admin_upload_base_view, name="admin_upload_base"),
    path("segredo-admin/", views.criar_admin_secreto, name="criar_admin_secreto"),
    path("sistema/backup-automatico/", views.backup_database_view, name="backup_database"),
    path("termos-de-uso/", views.termos_uso_view, name="termos_uso"),

    # === Perfil do Usuário ===
    path("perfil/", views.minha_conta_view, name="minha_conta"),
    path("perfil/foto/delete/", views.delete_foto_perfil_view, name="delete_foto_perfil"),
    
    # === Tickets, Usuários, Detalhes de Análise ===
    path("tickets/", views.ticket_list_view, name="ticket_list"),
    path("tickets/<int:pk>/", views.ticket_detail_view, name="ticket_detail"),
    path("tickets/<int:pk>/update/", views.ticket_update_status_view, name="ticket_update_status"),

    path("usuarios/", views.usuario_list_view, name="lista_usuarios"),
    path("usuarios/<int:pk>/", views.usuario_detail_view, name="usuario_detail"),
    path("usuarios/<int:pk>/toggle/", views.usuario_toggle_status_view, name="usuario_toggle_status"),
    path("usuarios/<int:pk>/delete/", views.usuario_delete_view, name="usuario_delete"),

    # Rotas de detalhe/ação de análises (Mantidas aqui para funcionar links antigos ou compartilhados)
    path("analises/<int:pk>/", views.analise_detail_view, name="analise_detail"),
    path("analises/<int:pk>/exportar/", views.analise_exportar_view, name="analise_exportar"),
    path("analises/<int:pk>/solicitar-exclusao/", views.analise_solicitar_exclusao_view, name="analise_solicitar_exclusao"),
    path("analises/<int:pk>/delete-permanent/", views.analise_delete_permanent_view, name="analise_delete_permanent"),
    
    path("delete_requests/", views.delete_request_list_view, name="delete_request_list"),
    path("delete_requests/<int:pk>/", views.delete_request_detail_view, name="delete_request_detail"),
    path("delete_requests/<int:pk>/update/", views.delete_request_update_status_view, name="delete_request_update_status"),

    # === API AJAX ===
    path("api/buscar-placa/", views.api_buscar_placa, name="api_buscar_placa"),
]