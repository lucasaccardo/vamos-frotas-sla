from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Sinistro(models.Model):
    # === 1. CLASSIFICAÇÃO (As 3 Ferramentas) ===
    SEGMENTOS = [
        ('AGRO', 'Agro (Máquinas/Implementos)'),
        ('PESADOS', 'Pesados (Caminhões)'),
        ('INTRA', 'Intra (Empilhadeiras/Outros)'),
    ]
    segmento = models.CharField(max_length=20, choices=SEGMENTOS, default='OUTROS', verbose_name="Segmento")

    # === 2. IDENTIFICAÇÃO (Automático via Placa) ===
    placa = models.CharField(max_length=20, verbose_name="Placa / Série")
    cliente = models.CharField(max_length=200, verbose_name="Cliente")
    chassi = models.CharField(max_length=100, blank=True, null=True)
    modelo_ativo = models.CharField(max_length=200, blank=True, null=True)
    n_contrato = models.CharField(max_length=100, blank=True, null=True, verbose_name="Nº Contrato")
    
    # Proteção: Define se cobra 20% ou Integral
    tem_protecao_casco = models.BooleanField(default=False, verbose_name="Tem Proteção do Casco?")

    # === 3. DADOS DA ABERTURA (Manual) ===
    n_chamado = models.CharField(max_length=50, verbose_name="Nº Chamado (GEO/Vetor)")
    data_ocorrencia = models.DateField(verbose_name="Data da Ocorrência")
    telefone_contato = models.CharField(max_length=100, blank=True, null=True)
    endereco_ativo = models.CharField(max_length=255, blank=True, null=True)
    
    MOTIVOS = [
        ('FURTO_ROUBO', 'Furto / Roubo'),
        ('SINISTRO_PT', 'Sinistro / Perda Total'),
        ('INCENDIO_PT', 'Incêndio / Perda Total'),
        ('COLISAO', 'Colisão (Avarias)'),
    ]
    motivo = models.CharField(max_length=20, choices=MOTIVOS, verbose_name="Motivo")

    # === 4. DOCUMENTAÇÃO (Checklist do Documento Word) ===
    check_bo = models.BooleanField(default=False, verbose_name="B.O. Oficial")
    check_ficha = models.BooleanField(default=False, verbose_name="Ficha de Ocorrência")
    check_cnh = models.BooleanField(default=False, verbose_name="CNH Condutor")
    check_fotos = models.BooleanField(default=False, verbose_name="Fotos Avarias")
    # Adicionei este extra que vi no documento:
    check_laudo = models.BooleanField(default=False, verbose_name="Laudo Pericial (BVQI)")

    # === 5. FINANCEIRO (Só libera edição nas etapas finais) ===
    valor_fipe = models.DecimalField(max_digits=15, decimal_places=2, default=0.00, verbose_name="Valor FIPE")
    
    modelo_implemento = models.CharField(max_length=200, blank=True, null=True)
    valor_implemento = models.DecimalField(max_digits=15, decimal_places=2, default=0.00)
    
    valor_franquia = models.DecimalField(max_digits=15, decimal_places=2, default=0.00, verbose_name="Valor Franquia")
    
    # Quem vai pagar?
    valor_seguradora = models.DecimalField(max_digits=15, decimal_places=2, default=0.00, verbose_name="Valor Seguradora")
    valor_cliente = models.DecimalField(max_digits=15, decimal_places=2, default=0.00, verbose_name="Valor Cliente")
    
    total_a_pagar = models.DecimalField(max_digits=15, decimal_places=2, default=0.00, verbose_name="Total Previsto")
    total_pago = models.DecimalField(max_digits=15, decimal_places=2, default=0.00, verbose_name="Total Recebido")

    # === 6. ESTEIRA DE STATUS (Workflow do Documento) ===
    TIPOS_PROCESSO = [
        ('CLIENTE_100', 'Pagamento Cliente (100%)'),
        ('SEGURADORA', 'Pagamento Seguradora'),
        ('CLIENTE_20', 'Pagamento Cliente (20% - Proteção)'),
    ]
    tipo_processo = models.CharField(max_length=20, choices=TIPOS_PROCESSO, blank=True, null=True)

    SETORES = [
        ('ABERTURA', 'Abertura'),
        ('CLIENTE', 'Com o Cliente'),
        ('MANUTENCAO', 'Manutenção (Orçamento)'),
        ('JURIDICO_SINISTRO', 'Depto. Sinistro / Jurídico'),
        ('VISTORIA', 'Vistoria (BVQI)'),
        ('FINANCEIRO', 'Financeiro (Pagamento)'),
        ('DESMOBILIZACAO', 'Desmobilização'),
        ('FINALIZADO', 'Finalizado'),
    ]
    setor_atual = models.CharField(max_length=20, choices=SETORES, default='ABERTURA')
    responsavel_setor = models.CharField(max_length=100, blank=True, null=True)

    # Prazos e Datas (Para o Dashboard de SLA)
    data_inicio_tratativa = models.DateTimeField(default=timezone.now)
    ultima_interacao = models.DateTimeField(auto_now=True)
    retornar_ate = models.DateField(blank=True, null=True, verbose_name="Prazo (Retornar Até)")
    
    # Controle Simpar (Específico para Seguradora)
    data_envio_simpar = models.DateField(blank=True, null=True)
    data_retorno_simpar = models.DateField(blank=True, null=True)

    criado_por = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    observacoes = models.TextField(blank=True, null=True)

    # --- PROPRIEDADES DE SLA (ADICIONADAS AGORA) ---
    @property
    def esta_atrasado(self):
        """Retorna True se a data de 'Retornar Até' já passou e não foi finalizado."""
        if self.retornar_ate and self.setor_atual != 'FINALIZADO':
            return self.retornar_ate < timezone.now().date()
        return False

    @property
    def dias_no_setor(self):
        """Conta há quantos dias o processo está parado no setor atual."""
        delta = timezone.now() - self.ultima_interacao
        return delta.days

    def save(self, *args, **kwargs):
        # Soma automática: Total = Cliente + Seguradora
        self.total_a_pagar = float(self.valor_seguradora) + float(self.valor_cliente)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"[{self.segmento}] {self.placa} - {self.n_chamado}"

class HistoricoSinistro(models.Model):
    """Rastreia toda a movimentação para calcular o SLA de cada setor."""
    sinistro = models.ForeignKey(Sinistro, on_delete=models.CASCADE, related_name='historico')
    setor_anterior = models.CharField(max_length=50)
    setor_novo = models.CharField(max_length=50)
    data_mudanca = models.DateTimeField(auto_now_add=True)
    alterado_por = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    comentario = models.CharField(max_length=255, blank=True, null=True)