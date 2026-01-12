from django.contrib import admin
from .models import ProcedureTemplate, ProcedureInstance, NodeInstance


@admin.register(ProcedureTemplate)
class ProcedureTemplateAdmin(admin.ModelAdmin):
    list_display = ['name', 'version', 'is_active', 'created_at', 'created_by']
    list_filter = ['is_active', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('Informações Básicas', {
            'fields': ('name', 'description', 'version', 'is_active')
        }),
        ('Estrutura', {
            'fields': ('structure',)
        }),
        ('Metadados', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ProcedureInstance)
class ProcedureInstanceAdmin(admin.ModelAdmin):
    list_display = ['id', 'template', 'status', 'current_node_id', 'started_at', 'started_by']
    list_filter = ['status', 'started_at', 'template']
    search_fields = ['id', 'template__name']
    readonly_fields = ['started_at', 'completed_at', 'started_by']
    
    fieldsets = (
        ('Informações', {
            'fields': ('template', 'status', 'current_node_id')
        }),
        ('Dados', {
            'fields': ('data',)
        }),
        ('Metadados', {
            'fields': ('started_by', 'started_at', 'completed_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(NodeInstance)
class NodeInstanceAdmin(admin.ModelAdmin):
    list_display = ['id', 'procedure', 'node_id', 'answered_at', 'answered_by']
    list_filter = ['answered_at']
    search_fields = ['procedure__id', 'node_id', 'question']
    readonly_fields = ['answered_at', 'answered_by']
    
    fieldsets = (
        ('Informações', {
            'fields': ('procedure', 'node_id')
        }),
        ('Conteúdo', {
            'fields': ('question', 'answer')
        }),
        ('Metadados', {
            'fields': ('answered_by', 'answered_at'),
            'classes': ('collapse',)
        }),
    )
