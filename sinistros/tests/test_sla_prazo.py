"""
Testes para cálculo de SLA com base em Prazo Limite (retornar_ate).

Verifica:
1. Quando retornar_ate existe, SLA é calculado como diferença entre retornar_ate e data_ultima_mudanca_setor
2. Fallback: sem histórico, usa created_at/updated_at do sinistro
3. Quando não há retornar_ate, mantém regras padrão por setor
"""
from datetime import date, timedelta
from django.test import TestCase
from django.contrib.auth.models import User
from sinistros.models import Sinistro, HistoricoSinistro


class SLAPrazoLimiteTestCase(TestCase):
    """Testes para cálculo de SLA com Prazo Limite"""

    def setUp(self):
        """Configuração inicial dos testes"""
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        
        # Criar um sinistro base
        self.sinistro = Sinistro.objects.create(
            placa='ABC1234',
            cliente='Cliente Teste',
            modelo_ativo='Modelo Teste',
            n_chamado='CH001',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            segmento='PESADOS',
            setor_atual='CLIENTE',
            criado_por=self.user
        )

    def test_sla_com_prazo_limite_e_historico(self):
        """
        Testa SLA quando retornar_ate existe e há histórico.
        SLA deve ser calculado como: (retornar_ate - data_ultima_mudanca_setor).days
        """
        # Criar histórico de mudança para CLIENTE há 3 dias
        from django.utils import timezone
        from django.db import connection
        
        historico = HistoricoSinistro.objects.create(
            sinistro=self.sinistro,
            setor_anterior='ABERTURA',
            setor_novo='CLIENTE',
            alterado_por=self.user,
            comentario='Mudança para Cliente'
        )
        
        # Atualizar data_mudanca diretamente no banco
        data_passada = timezone.now() - timedelta(days=3)
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE sinistros_historicosinistro SET data_mudanca = %s WHERE id = %s",
                [data_passada, historico.id]
            )
        
        # Recarregar histórico
        historico.refresh_from_db()
        
        # Definir prazo limite para daqui a 2 dias (total: 5 dias desde mudança)
        self.sinistro.retornar_ate = date.today() + timedelta(days=2)
        self.sinistro.save()
        
        # Calcular SLA
        dias, label = self.sinistro.sla_por_setor()
        
        # Deve retornar 5 dias (diferença entre prazo e data da última mudança)
        self.assertEqual(dias, 5)
        self.assertEqual(label, '5 dias corridos')

    def test_sla_com_prazo_limite_sem_historico(self):
        """
        Testa SLA quando retornar_ate existe mas não há histórico.
        Deve usar criado_em como fallback.
        """
        # Remover qualquer histórico existente
        HistoricoSinistro.objects.filter(sinistro=self.sinistro).delete()
        
        # Definir prazo limite para daqui a 7 dias
        self.sinistro.retornar_ate = date.today() + timedelta(days=7)
        self.sinistro.save()
        
        # Calcular SLA
        dias, label = self.sinistro.sla_por_setor()
        
        # Deve retornar aproximadamente 7 dias (pode variar se criado_em não é exatamente hoje)
        self.assertIsNotNone(dias)
        self.assertGreaterEqual(dias, 6)  # Pelo menos 6 dias
        self.assertLessEqual(dias, 8)  # No máximo 8 dias
        self.assertIn('dias corridos', label)

    def test_sla_sem_prazo_limite_usa_regras_padrao(self):
        """
        Testa que quando não há retornar_ate, usa regras padrão por setor.
        """
        # Garantir que retornar_ate está vazio
        self.sinistro.retornar_ate = None
        self.sinistro.setor_atual = 'CLIENTE'
        self.sinistro.save()
        
        # Calcular SLA
        dias, label = self.sinistro.sla_por_setor()
        
        # Deve usar regra padrão: CLIENTE = 5 dias
        self.assertEqual(dias, 5)
        self.assertEqual(label, '5 dias corridos')

    def test_sla_sem_prazo_limite_manutencao_normal(self):
        """
        Testa regra padrão para MANUTENCAO sem aguardar aprovação.
        """
        self.sinistro.retornar_ate = None
        self.sinistro.setor_atual = 'MANUTENCAO'
        self.sinistro.aguarda_aprovacao_os = False
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        
        # Deve usar regra padrão: MANUTENCAO = 15 dias
        self.assertEqual(dias, 15)
        self.assertEqual(label, '15 dias corridos')

    def test_sla_sem_prazo_limite_manutencao_aguardando(self):
        """
        Testa regra padrão para MANUTENCAO aguardando aprovação.
        """
        self.sinistro.retornar_ate = None
        self.sinistro.setor_atual = 'MANUTENCAO'
        self.sinistro.aguarda_aprovacao_os = True
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        
        # Deve usar regra padrão: MANUTENCAO aguardando = 2 dias
        self.assertEqual(dias, 2)
        self.assertEqual(label, '2 dias corridos')

    def test_sla_prazo_vencido_retorna_zero_dias(self):
        """
        Testa que quando prazo está vencido, retorna 0 dias (não negativo).
        """
        # Criar histórico há 3 dias usando update direto no DB
        from django.utils import timezone
        from django.db import connection
        
        historico = HistoricoSinistro.objects.create(
            sinistro=self.sinistro,
            setor_anterior='ABERTURA',
            setor_novo='CLIENTE',
            alterado_por=self.user
        )
        
        # Atualizar data_mudanca diretamente no banco para 3 dias atrás
        data_passada = timezone.now() - timedelta(days=3)
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE sinistros_historicosinistro SET data_mudanca = %s WHERE id = %s",
                [data_passada, historico.id]
            )
        
        # Recarregar histórico
        historico.refresh_from_db()
        
        # Definir prazo limite para 5 dias atrás (vencido E anterior à mudança de setor)
        # Isso resulta em um valor negativo que deve ser convertido a 0
        self.sinistro.retornar_ate = date.today() - timedelta(days=5)
        self.sinistro.save()
        
        # Calcular SLA
        dias, label = self.sinistro.sla_por_setor()
        
        # Deve retornar 0 (max(0, valor_negativo))
        # Prazo é 5 dias atrás, mudança foi 3 dias atrás, então prazo < mudança = negativo
        self.assertEqual(dias, 0)
        self.assertEqual(label, '0 dias corridos')

    def test_data_ultima_mudanca_setor_com_historico(self):
        """
        Testa método data_ultima_mudanca_setor quando há histórico.
        """
        # Criar múltiplos históricos
        from django.utils import timezone
        from django.db import connection
        
        # Histórico antigo
        h1 = HistoricoSinistro.objects.create(
            sinistro=self.sinistro,
            setor_anterior='ABERTURA',
            setor_novo='PRECIFICACAO',
            alterado_por=self.user
        )
        
        # Histórico mais recente (mudança para CLIENTE)
        h2 = HistoricoSinistro.objects.create(
            sinistro=self.sinistro,
            setor_anterior='PRECIFICACAO',
            setor_novo='CLIENTE',
            alterado_por=self.user
        )
        
        # Atualizar datas diretamente no banco
        data_antiga = timezone.now() - timedelta(days=10)
        data_recente = timezone.now() - timedelta(days=3)
        
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE sinistros_historicosinistro SET data_mudanca = %s WHERE id = %s",
                [data_antiga, h1.id]
            )
            cursor.execute(
                "UPDATE sinistros_historicosinistro SET data_mudanca = %s WHERE id = %s",
                [data_recente, h2.id]
            )
        
        # Recarregar históricos
        h1.refresh_from_db()
        h2.refresh_from_db()
        
        # Obter data da última mudança para CLIENTE
        data_mudanca = self.sinistro.data_ultima_mudanca_setor()
        
        # Deve ser a data do histórico mais recente
        expected_date = data_recente.date()
        self.assertEqual(data_mudanca, expected_date)

    def test_data_ultima_mudanca_setor_sem_historico(self):
        """
        Testa método data_ultima_mudanca_setor quando não há histórico.
        Deve usar fallback para ultima_interacao ou criado_em.
        """
        # Remover histórico
        HistoricoSinistro.objects.filter(sinistro=self.sinistro).delete()
        
        # Obter data da última mudança
        data_mudanca = self.sinistro.data_ultima_mudanca_setor()
        
        # Deve retornar uma data válida (ultima_interacao, criado_em ou today)
        self.assertIsNotNone(data_mudanca)
        self.assertIsInstance(data_mudanca, date)

    def test_sla_setores_sem_prazo(self):
        """
        Testa que setores sem prazo retornam (None, 'SEM PRAZO').
        """
        setores_sem_prazo = ['DESMOBILIZACAO_MEDICAO', 'FINALIZADO', 'ABERTURA']
        
        for setor in setores_sem_prazo:
            self.sinistro.setor_atual = setor
            self.sinistro.retornar_ate = None
            self.sinistro.save()
            
            dias, label = self.sinistro.sla_por_setor()
            
            self.assertIsNone(dias, f"Setor {setor} deveria retornar None para dias")
            self.assertEqual(label, 'SEM PRAZO', f"Setor {setor} deveria retornar 'SEM PRAZO'")

    def test_sla_com_prazo_limite_sobrepoe_sem_prazo(self):
        """
        Testa que mesmo para setores sem prazo, se houver retornar_ate,
        o cálculo é feito com base nele.
        """
        # Setor sem prazo (FINALIZADO)
        self.sinistro.setor_atual = 'FINALIZADO'
        self.sinistro.retornar_ate = date.today() + timedelta(days=10)
        self.sinistro.save()
        
        # Calcular SLA
        dias, label = self.sinistro.sla_por_setor()
        
        # Mesmo sendo FINALIZADO, deve calcular com base no prazo limite
        self.assertIsNotNone(dias)
        self.assertGreaterEqual(dias, 9)
        self.assertLessEqual(dias, 11)
        self.assertIn('dias corridos', label)

    def test_sla_precificacao_padrao(self):
        """Testa SLA padrão para PRECIFICACAO (5 dias)"""
        self.sinistro.setor_atual = 'PRECIFICACAO'
        self.sinistro.retornar_ate = None
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        self.assertEqual(dias, 5)
        self.assertEqual(label, '5 dias corridos')

    def test_sla_financeiro_padrao(self):
        """Testa SLA padrão para FINANCEIRO (4 dias)"""
        self.sinistro.setor_atual = 'FINANCEIRO'
        self.sinistro.retornar_ate = None
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        self.assertEqual(dias, 4)
        self.assertEqual(label, '4 dias corridos')

    def test_sla_depto_sinistro_padrao(self):
        """Testa SLA padrão para DEPTO_SINISTRO (60 dias)"""
        self.sinistro.setor_atual = 'DEPTO_SINISTRO'
        self.sinistro.retornar_ate = None
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        self.assertEqual(dias, 60)
        self.assertEqual(label, '60 dias corridos')
