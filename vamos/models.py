from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import random 

# Importações necessárias para o Perfil e Sinais
from django.db.models.signals import post_save
from django.dispatch import receiver

# === Função Auxiliar ===
def gerar_protocolo():
    """Gera um número de protocolo de 8 dígitos aleatórios."""
    return str(random.randint(10000000, 99999999))

# === Tickets ===
class Ticket(models.Model):
    STATUS_CHOICES = [
        ("Pendente", "Pendente"),
        ("Em andamento", "Em andamento"),
        ("Concluído", "Concluído"),
        ("Cancelado", "Cancelado"),
    ]

    usuario = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Protocolo único
    protocolo = models.CharField(max_length=8, unique=True, default=gerar_protocolo)
    
    titulo = models.CharField(max_length=200)
    descricao = models.TextField()
    
    # Resposta do Admin
    resposta_admin = models.TextField(blank=True, null=True, help_text="Resposta oficial do suporte")
    data_resposta = models.DateTimeField(blank=True, null=True)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="Pendente")
    created_at = models.DateTimeField(default=timezone.now)   
    updated_at = models.DateTimeField(auto_now=True)          

    def save(self, *args, **kwargs):
        # Garante que tenha protocolo ao salvar
        if not self.protocolo:
            self.protocolo = gerar_protocolo()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Ticket #{self.protocolo} - {self.titulo}"


# === Análises ===
class Analise(models.Model):
    TIPO_CHOICES = [
        ("sla_mensal", "SLA Mensal"),
        ("cenarios", "Análise de Cenários"),
    ]

    # Quem fez a análise
    usuario = models.ForeignKey(User, on_delete=models.CASCADE) 
    
    # Protocolo
    protocolo = models.CharField(max_length=8, unique=True, default=gerar_protocolo)
    
    # Tipo da análise
    tipo = models.CharField(max_length=50, choices=TIPO_CHOICES)
    
    # Data e Hora
    data_criacao = models.DateTimeField(default=timezone.now)
    
    # Campo para guardar todos os dados do cálculo
    dados = models.JSONField(default=dict, blank=True) 
    
    # Campo para guardar o PDF gerado automaticamente
    arquivo_pdf = models.FileField(upload_to='pdfs/%Y/%m/', blank=True, null=True)
    
    # Campos extras para facilitar a busca no Admin depois
    placa = models.CharField(max_length=20, blank=True, null=True)
    cliente = models.CharField(max_length=200, blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.protocolo:
            self.protocolo = gerar_protocolo()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Análise {self.protocolo} - {self.tipo}"


# === Delete Requests ===
class DeleteRequest(models.Model):
    STATUS_CHOICES = [
        ("Pendente", "Pendente"),
        ("Aprovado", "Aprovado"),
        ("Rejeitado", "Rejeitado"),
    ]

    motivo = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="Pendente")
    data_solicitacao = models.DateTimeField(default=timezone.now)
    solicitante = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f"DeleteRequest {self.id} - {self.status}"


# === Perfil de Usuário (Matrícula) ===
class Perfil(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='perfil')
    matricula = models.CharField(max_length=20, blank=True, null=True)
    
    # NOVO CAMPO:
    termos_aceitos_em = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Perfil de {self.user.username}"

# === Sinais (Signals) ===
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Perfil.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.perfil.save()
# === Perfil - Foto ===
class Perfil(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='perfil')
    matricula = models.CharField(max_length=20, blank=True, null=True)
    termos_aceitos_em = models.DateTimeField(null=True, blank=True)
    
    # NOVO CAMPO:
    foto = models.ImageField(upload_to='perfil_fotos/', blank=True, null=True)

    def __str__(self):
        return f"Perfil de {self.user.username}"
        
    