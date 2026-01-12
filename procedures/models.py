from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class ProcedureTemplate(models.Model):
    """
    Template/definition of a procedure workflow.
    Contains the structure (nodes/steps) in JSON format.
    """
    name = models.CharField(max_length=200, help_text="Nome do procedimento (ex: 'Sinistro')")
    description = models.TextField(blank=True, help_text="Descrição do procedimento")
    version = models.CharField(max_length=20, default="1.0", help_text="Versão do template")
    
    # JSON structure defining the workflow nodes
    # Example: {"nodes": [{"id": "1", "type": "text", "question": "...", "next": "2"}, ...]}
    structure = models.JSONField(help_text="Estrutura JSON com os nodes do workflow")
    
    is_active = models.BooleanField(default=True, help_text="Se o template está ativo")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='created_templates')
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Template de Procedimento'
        verbose_name_plural = 'Templates de Procedimentos'
    
    def __str__(self):
        return f"{self.name} (v{self.version})"


class ProcedureInstance(models.Model):
    """
    An instance/execution of a procedure template.
    Tracks the state of a specific workflow execution.
    """
    template = models.ForeignKey(ProcedureTemplate, on_delete=models.PROTECT, related_name='instances')
    
    # Current state
    current_node_id = models.CharField(max_length=50, blank=True, null=True, help_text="ID do node atual")
    status = models.CharField(
        max_length=20,
        choices=[
            ('IN_PROGRESS', 'Em Progresso'),
            ('COMPLETED', 'Completado'),
            ('CANCELLED', 'Cancelado'),
        ],
        default='IN_PROGRESS'
    )
    
    # Metadata
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    started_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='started_procedures')
    
    # Accumulated data from all nodes
    data = models.JSONField(default=dict, help_text="Dados acumulados do procedimento")
    
    class Meta:
        ordering = ['-started_at']
        verbose_name = 'Instância de Procedimento'
        verbose_name_plural = 'Instâncias de Procedimentos'
    
    def __str__(self):
        return f"{self.template.name} - {self.id} ({self.status})"


class NodeInstance(models.Model):
    """
    Record of a node execution within a procedure instance.
    Stores the answer/response for each step.
    """
    procedure = models.ForeignKey(ProcedureInstance, on_delete=models.CASCADE, related_name='nodes')
    node_id = models.CharField(max_length=50, help_text="ID do node no template")
    
    # Question/prompt shown to user (from template)
    question = models.TextField(help_text="Pergunta/prompt apresentado")
    
    # User's answer/response
    answer = models.JSONField(help_text="Resposta do usuário (pode ser texto, número, etc.)")
    
    # Timestamps
    answered_at = models.DateTimeField(default=timezone.now)
    answered_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='answered_nodes')
    
    class Meta:
        ordering = ['answered_at']
        verbose_name = 'Node Instance'
        verbose_name_plural = 'Node Instances'
    
    def __str__(self):
        return f"{self.procedure.id} - Node {self.node_id}"
