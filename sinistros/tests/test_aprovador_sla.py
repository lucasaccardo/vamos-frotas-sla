"""
Testes para o campo aprovador_os e SLA por setor.

Verifica:
1. Cálculo correto de retornar_ate ao mudar de setor
2. Ajuste de SLA quando aguarda_aprovacao_os é marcado
3. Persistência do campo aprovador_os
"""
from datetime import date, timedelta
from django.test import TestCase
from django.contrib.auth.models import User
from sinistros.models import Sinistro, HistoricoSinistro


class AprovadorSLATestCase(TestCase):
    """Testes para aprovador_os e cálculo automático de SLA"""

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
            setor_atual='ABERTURA',
            criado_por=self.user
        )

    def test_sla_por_setor_cliente(self):
        """Testa SLA para setor CLIENTE (5 dias corridos)"""
        self.sinistro.setor_atual = 'CLIENTE'
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        self.assertEqual(dias, 5)
        self.assertEqual(label, '5 dias corridos')

    def test_sla_por_setor_precificacao(self):
        """Testa SLA para setor PRECIFICACAO (5 dias corridos)"""
        self.sinistro.setor_atual = 'PRECIFICACAO'
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        self.assertEqual(dias, 5)
        self.assertEqual(label, '5 dias corridos')

    def test_sla_por_setor_financeiro(self):
        """Testa SLA para setor FINANCEIRO (4 dias corridos)"""
        self.sinistro.setor_atual = 'FINANCEIRO'
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        self.assertEqual(dias, 4)
        self.assertEqual(label, '4 dias corridos')

    def test_sla_por_setor_desmobilizacao(self):
        """Testa SLA para setor DESMOBILIZACAO_MEDICAO (sem prazo)"""
        self.sinistro.setor_atual = 'DESMOBILIZACAO_MEDICAO'
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        self.assertIsNone(dias)
        self.assertEqual(label, 'SEM PRAZO')

    def test_sla_por_setor_depto_sinistro(self):
        """Testa SLA para setor DEPTO_SINISTRO (60 dias corridos)"""
        self.sinistro.setor_atual = 'DEPTO_SINISTRO'
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        self.assertEqual(dias, 60)
        self.assertEqual(label, '60 dias corridos')

    def test_sla_por_setor_manutencao_normal(self):
        """Testa SLA para setor MANUTENCAO sem aguardar aprovação (15 dias corridos)"""
        self.sinistro.setor_atual = 'MANUTENCAO'
        self.sinistro.aguarda_aprovacao_os = False
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        self.assertEqual(dias, 15)
        self.assertEqual(label, '15 dias corridos')

    def test_sla_por_setor_manutencao_aguardando(self):
        """Testa SLA para setor MANUTENCAO aguardando aprovação (2 dias corridos)"""
        self.sinistro.setor_atual = 'MANUTENCAO'
        self.sinistro.aguarda_aprovacao_os = True
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        self.assertEqual(dias, 2)
        self.assertEqual(label, '2 dias corridos')

    def test_sla_por_setor_finalizado(self):
        """Testa SLA para setor FINALIZADO (sem prazo)"""
        self.sinistro.setor_atual = 'FINALIZADO'
        self.sinistro.save()
        
        dias, label = self.sinistro.sla_por_setor()
        self.assertIsNone(dias)
        self.assertEqual(label, 'SEM PRAZO')

    def test_retornar_ate_calculado_ao_mudar_setor(self):
        """Testa que retornar_ate é calculado ao mudar de ABERTURA para MANUTENCAO"""
        # Simular mudança de setor
        self.sinistro.setor_atual = 'MANUTENCAO'
        self.sinistro.aguarda_aprovacao_os = False
        
        # Calcular retornar_ate manualmente (como a view faz)
        dias, label = self.sinistro.sla_por_setor()
        if dias is not None:
            self.sinistro.retornar_ate = date.today() + timedelta(days=dias)
        
        self.sinistro.save()
        
        # Verificar que retornar_ate foi definido corretamente (15 dias)
        expected_date = date.today() + timedelta(days=15)
        self.assertEqual(self.sinistro.retornar_ate, expected_date)

    def test_retornar_ate_ajustado_ao_marcar_aguarda_aprovacao(self):
        """Testa que retornar_ate é ajustado ao marcar aguarda_aprovacao_os"""
        # Primeiro configurar como MANUTENCAO normal
        self.sinistro.setor_atual = 'MANUTENCAO'
        self.sinistro.aguarda_aprovacao_os = False
        
        dias, label = self.sinistro.sla_por_setor()
        if dias is not None:
            self.sinistro.retornar_ate = date.today() + timedelta(days=dias)
        
        self.sinistro.save()
        
        # Agora marcar aguarda_aprovacao_os como True
        self.sinistro.aguarda_aprovacao_os = True
        
        # Recalcular retornar_ate
        dias, label = self.sinistro.sla_por_setor()
        if dias is not None:
            self.sinistro.retornar_ate = date.today() + timedelta(days=dias)
        
        self.sinistro.save()
        
        # Verificar que retornar_ate foi ajustado para 2 dias
        expected_date = date.today() + timedelta(days=2)
        self.assertEqual(self.sinistro.retornar_ate, expected_date)

    def test_aprovador_os_persistencia(self):
        """Testa persistência do campo aprovador_os quando setor é MANUTENCAO"""
        self.sinistro.setor_atual = 'MANUTENCAO'
        self.sinistro.aguarda_aprovacao_os = True
        self.sinistro.aprovador_os = 'João Silva'
        self.sinistro.save()
        
        # Recarregar do banco de dados
        sinistro_recarregado = Sinistro.objects.get(pk=self.sinistro.pk)
        
        # Verificar que aprovador_os foi salvo corretamente
        self.assertEqual(sinistro_recarregado.aprovador_os, 'João Silva')
        self.assertTrue(sinistro_recarregado.aguarda_aprovacao_os)
        self.assertEqual(sinistro_recarregado.setor_atual, 'MANUTENCAO')

    def test_aprovador_os_vazio_quando_nao_aguarda(self):
        """Testa que aprovador_os pode ser vazio quando não aguarda aprovação"""
        self.sinistro.setor_atual = 'MANUTENCAO'
        self.sinistro.aguarda_aprovacao_os = False
        self.sinistro.aprovador_os = ''
        self.sinistro.save()
        
        # Recarregar do banco de dados
        sinistro_recarregado = Sinistro.objects.get(pk=self.sinistro.pk)
        
        # Verificar estado
        self.assertEqual(sinistro_recarregado.aprovador_os, '')
        self.assertFalse(sinistro_recarregado.aguarda_aprovacao_os)

    def test_retornar_ate_none_para_setores_sem_prazo(self):
        """Testa que retornar_ate é None para setores sem prazo"""
        self.sinistro.setor_atual = 'FINALIZADO'
        
        # Calcular retornar_ate
        dias, label = self.sinistro.sla_por_setor()
        if dias is not None:
            self.sinistro.retornar_ate = date.today() + timedelta(days=dias)
        else:
            self.sinistro.retornar_ate = None
        
        self.sinistro.save()
        
        # Verificar que retornar_ate é None
        self.assertIsNone(self.sinistro.retornar_ate)
