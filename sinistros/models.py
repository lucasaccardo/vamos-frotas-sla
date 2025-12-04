from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Sinistro(models.Model):
    # --- 1. IDENTIFICAÇÃO (Automático via Placa) ---
    placa = models.CharField(max_length=10, verbose_name="Placa")
    cliente = models.CharField(max_length=200, verbose_name="Cliente")
    chassi = models.CharField(max_length=100, blank=True, null=True)
    modelo_ativo = models.CharField(max_length=200, blank=True, null=True)
    n_contrato = models.CharField(max_length=50, blank=True, null=True, verbose_name="Nº Contrato")
    
    SEGMENTOS = [('AGRO', 'Agro'), ('PESADOS', 'Pesados'), ('INTRA', 'Intra'), ('OUTROS', 'Outros')]
    segmento = models.CharField(max_length=20, choices=SEGMENTOS, blank=True, null=True)
    
    tem_protecao_casco = models.BooleanField(default=False, verbose_name="Proteção do Casco?")

    # --- 2. DADOS DO CHAMADO (Manual) ---
    n_chamado = models.CharField(max_length=50, verbose_name="Nº Chamado (GEO/Vetor)")
    data_ocorrencia = models.DateField(verbose_name="Data da Ocorrência")
    telefone_contato = models.CharField(max_length=50, blank=True, null=True)
    endereco_ativo = models.CharField(max_length=255, blank=True, null=True)
    
    MOTIVOS = [
        ('FURTO_ROUBO', 'Furto / Roubo'),
        ('SINISTRO_PT', 'Sinistro / Perda Total'),
        ('INCENDIO_PT', 'Incêndio / Perda Total'),
    ]
    motivo = models.CharField(max_length=20, choices=MOTIVOS)

    # --- 3. CHECKLIST DOCUMENTAÇÃO ---
    check_bo = models.BooleanField(default=False, verbose_name="B.O. Oficial")
    check_ficha = models.BooleanField(default=False, verbose_name="Ficha de Ocorrência")
    check_cnh = models.BooleanField(default=False, verbose_name="CNH Condutor")
    check_fotos = models.BooleanField(default=False, verbose_name="Fotos Avarias")

    # --- 4. FINANCEIRO (Lógica de Bloqueio na Tela) ---
    valor_fipe = models.DecimalField(max_digits=12, decimal_places=2, default=0.00, verbose_name="Valor FIPE")
    modelo_implemento = models.CharField(max_length=200, blank=True, null=True)
    valor_implemento = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    
    valor_seguradora = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    valor_cliente = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    
    total_a_pagar = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    total_pago = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)

    # --- 5. ESTEIRA / FLUXO ---
    TIPOS_PROCESSO = [
        ('PAG_CLIENTE', 'Pagamento pelo Cliente (100%)'),
        ('PAG_SEGURADORA', 'Pagamento por Seguradora'),
        ('PAG_CLIENTE_20', 'Pagamento pelo Cliente (20% - Proteção)'),
    ]
    tipo_processo = models.CharField(max_length=20, choices=TIPOS_PROCESSO, blank=True, null=True)

    SETORES = [
        ('ABERTURA', 'Abertura / Triagem'),
        ('CLIENTE', 'Cliente'),
        ('MANUTENCAO', 'Manutenção'),
        ('DEPTO_SINISTRO', 'Depto. Sinistro'),
        ('FINANCEIRO', 'Financeiro'),
        ('GERENCIA', 'Gerência'),
        ('PRECIFICACAO', 'Precificação'),
        ('DESMOBILIZACAO', 'Desmobilização'),
        ('FINALIZADO', 'Finalizado'),
    ]
    setor_atual = models.CharField(max_length=20, choices=SETORES, default='ABERTURA')
    responsavel_setor = models.CharField(max_length=100, blank=True, null=True, verbose_name="Responsável Atual")

    # Prazos e SLA
    data_inicio_tratativa = models.DateTimeField(default=timezone.now)
    ultima_interacao = models.DateTimeField(auto_now=True)
    retornar_ate = models.DateField(blank=True, null=True, verbose_name="Prazo para Retorno")
    
    # Campos específicos Simpar
    data_envio_simpar = models.DateField(blank=True, null=True)
    data_retorno_simpar = models.DateField(blank=True, null=True)

    criado_por = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    observacoes = models.TextField(blank=True, null=True)

    def save(self, *args, **kwargs):
        # Soma automática antes de salvar
        self.total_a_pagar = float(self.valor_seguradora) + float(self.valor_cliente)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.placa} - {self.n_chamado}"

class HistoricoSinistro(models.Model):
    """Tabela para auditar por onde o processo passou."""
    sinistro = models.ForeignKey(Sinistro, on_delete=models.CASCADE, related_name='historico')
    setor_anterior = models.CharField(max_length=50)
    setor_novo = models.CharField(max_length=50)
    data_mudanca = models.DateTimeField(auto_now_add=True)
    alterado_por = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    comentario = models.CharField(max_length=255, blank=True, null=True)