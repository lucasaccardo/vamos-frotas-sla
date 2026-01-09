from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class Sinistro(models.Model):
    # Opções de Status/Setores
    SETORES = [
        ('ABERTURA', 'Abertura'),
        ('CLIENTE', 'Aprovação Cliente'),
        ('PRECIFICACAO', 'Precificação'),
        ('FINANCEIRO', 'Financeiro (Pagamento)'),
        ('DESMOBILIZACAO_MEDICAO', 'Desmobilização / Medição'),
        ('DEPTO_SINISTRO', 'Depto. Sinistro'),
        ('MANUTENCAO', 'Manutenção'),
        ('FINALIZADO', 'Finalizado'),
    ]
    
    MOTIVOS = [
        ('COLISAO', 'Colisão'),
        ('FURTO_ROUBO', 'Furto/Roubo'),
        ('INCENDIO', 'Incêndio'),
        ('NATUREZA', 'Fenômenos da Natureza'),
        ('TERCEIROS', 'Danos a Terceiros'),
    ]
    
    SEGMENTOS = [
        ('AGRO', 'Agro (Máquinas/Implementos)'),
        ('PESADOS', 'Pesados (Caminhões)'),
        ('INTRA', 'Intralogística (Empilhadeiras)'),
    ]

    # --- DADOS DO ATIVO ---
    placa = models.CharField(max_length=20)
    cliente = models.CharField(max_length=100)
    modelo_ativo = models.CharField(max_length=100)
    chassi = models.CharField(max_length=50, blank=True, null=True)
    n_contrato = models.CharField(max_length=50, blank=True, null=True)
    segmento = models.CharField(max_length=20, choices=SEGMENTOS, default='PESADOS')
    
    # --- DADOS DA OCORRÊNCIA ---
    n_chamado = models.CharField(max_length=50, verbose_name="Nº Chamado")
    data_ocorrencia = models.DateField()
    motivo = models.CharField(max_length=20, choices=MOTIVOS)
    endereco_ativo = models.CharField(max_length=200, blank=True, null=True)
    telefone_contato = models.CharField(max_length=20, blank=True, null=True)
    
    # --- FINANCEIRO (OPCIONAIS NA ABERTURA) ---
    valor_fipe = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    valor_implemento = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    valor_franquia = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    valor_seguradora = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    valor_cliente = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    
    # Totais calculados
    total_a_pagar = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    total_pago = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    
    # --- CONTROLE E SLA ---
    setor_atual = models.CharField(max_length=30, choices=SETORES, default='ABERTURA')
    responsavel_setor = models.CharField(max_length=100, blank=True, null=True)
    
    # --- CONTROLE DE APROVAÇÃO (NOVOS CAMPOS) ---
    aguarda_aprovacao_os = models.BooleanField(default=False, help_text="Se o processo está aguardando aprovação de O.S.")
    aprovador_os = models.CharField(max_length=255, blank=True, null=True, help_text="Nome do aprovador da O.S., se houver")
    
    criado_por = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    criado_em = models.DateTimeField(auto_now_add=True)
    ultima_interacao = models.DateTimeField(auto_now=True)
    retornar_ate = models.DateField(blank=True, null=True, verbose_name="Prazo SLA")

    # --- CHECKLIST ---
    check_bo = models.BooleanField(default=False, verbose_name="B.O.")
    check_ficha = models.BooleanField(default=False, verbose_name="Ficha de Ocorrência")
    check_cnh = models.BooleanField(default=False, verbose_name="CNH")
    check_fotos = models.BooleanField(default=False, verbose_name="Fotos")
    
    tem_protecao_casco = models.BooleanField(default=False)
    observacoes = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.placa

    @property
    def dias_no_setor(self):
        if self.setor_atual == 'FINALIZADO': return 0
        return (timezone.now() - self.ultima_interacao).days

    @property
    def esta_atrasado(self):
        if self.retornar_ate and self.retornar_ate < timezone.now().date():
            return True
        return False

    def sla_por_setor(self):
        """
        Retorna a SLA máxima (em dias corridos) e o label descritivo para o setor atual.
        Returns: tuple (dias: Optional[int], label: str)
        
        Regras:
        - CLIENTE -> 5 dias corridos
        - PRECIFICACAO -> 5 dias corridos  
        - FINANCEIRO -> 4 dias corridos
        - DESMOBILIZACAO_MEDICAO -> sem prazo
        - DEPTO_SINISTRO -> 60 dias corridos
        - MANUTENCAO -> 15 dias corridos (ou 2 dias se aguarda_aprovacao_os == True)
        - FINALIZADO -> sem prazo
        """
        # Mapeamento de SLA por setor
        SLA_CONFIG = {
            'CLIENTE': (5, '5 dias corridos'),
            'PRECIFICACAO': (5, '5 dias corridos'),
            'FINANCEIRO': (4, '4 dias corridos'),
            'DESMOBILIZACAO_MEDICAO': (None, 'SEM PRAZO'),
            'DEPTO_SINISTRO': (60, '60 dias corridos'),
            'FINALIZADO': (None, 'SEM PRAZO'),
            'ABERTURA': (None, 'SEM PRAZO'),
        }
        
        setor = self.setor_atual or ''
        
        # Caso especial: MANUTENCAO depende de aguarda_aprovacao_os
        if setor == 'MANUTENCAO':
            if self.aguarda_aprovacao_os:
                return (2, '2 dias corridos')
            else:
                return (15, '15 dias corridos')
        
        # Retorna do mapeamento ou default
        return SLA_CONFIG.get(setor, (None, 'SEM PRAZO'))

class HistoricoSinistro(models.Model):
    sinistro = models.ForeignKey(Sinistro, on_delete=models.CASCADE, related_name='historico')
    data_mudanca = models.DateTimeField(auto_now_add=True)
    setor_anterior = models.CharField(max_length=50)
    setor_novo = models.CharField(max_length=50)
    alterado_por = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    comentario = models.TextField(blank=True, null=True)

# === NOVA TABELA DE FROTA ===
class Frota(models.Model):
    placa = models.CharField(max_length=20, unique=True, db_index=True)
    cliente = models.CharField(max_length=200, blank=True, null=True)
    modelo = models.CharField(max_length=200, blank=True, null=True)
    chassi = models.CharField(max_length=100, blank=True, null=True)
    contrato = models.CharField(max_length=100, blank=True, null=True)
    centro_custo = models.CharField(max_length=50, blank=True, null=True)
    segmento = models.CharField(max_length=50, blank=True, null=True)
    
    atualizado_em = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.placa} - {self.cliente}"