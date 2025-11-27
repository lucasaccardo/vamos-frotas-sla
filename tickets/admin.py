from django.contrib import admin
from .models import Ticket

@admin.register(Ticket)
class TicketAdmin(admin.ModelAdmin):
    list_display = ("assunto", "username", "status", "criado_em", "atualizado_em")
    search_fields = ("assunto", "username", "descricao", "status")
    list_filter = ("status", "criado_em")
