from django.contrib import admin
from .models import Analise, Ticket, DeleteRequest

# === Configuração da Análise ===
@admin.register(Analise)
class AnaliseAdmin(admin.ModelAdmin):
    # Colunas que aparecem na lista (usando os nomes novos do models.py)
    list_display = ('tipo', 'placa', 'cliente', 'usuario', 'data_criacao')
    
    # Filtros laterais
    list_filter = ('tipo', 'data_criacao', 'usuario')
    
    # Barra de pesquisa (busca na placa, cliente e dentro do JSON de dados)
    search_fields = ('placa', 'cliente', 'dados')
    
    # Ordenação padrão (mais recente primeiro)
    ordering = ('-data_criacao',)

# === Configuração dos Tickets ===
@admin.register(Ticket)
class TicketAdmin(admin.ModelAdmin):
    list_display = ('id', 'titulo', 'status', 'usuario', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('titulo', 'descricao')

# === Configuração de Solicitação de Exclusão ===
@admin.register(DeleteRequest)
class DeleteRequestAdmin(admin.ModelAdmin):
    list_display = ('id', 'solicitante', 'status', 'data_solicitacao')
    list_filter = ('status',)