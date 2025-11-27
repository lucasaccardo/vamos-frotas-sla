import uuid
from django.db import models

class Analise(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    username = models.CharField(max_length=150)
    tipo = models.CharField(max_length=50)  # ex: 'cenarios' ou 'sla_mensal'
    data_hora = models.DateTimeField()
    dados_json = models.JSONField()
    pdf_path = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"{self.tipo} - {self.username}"
