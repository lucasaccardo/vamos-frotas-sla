import uuid
from django.db import models

class Ticket(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    username = models.CharField(max_length=150)
    assunto = models.CharField(max_length=255)
    descricao = models.TextField()
    status = models.CharField(max_length=50)  # ex: aberto, fechado, em andamento
    criado_em = models.DateTimeField(auto_now_add=True)
    atualizado_em = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.assunto} - {self.username}"
