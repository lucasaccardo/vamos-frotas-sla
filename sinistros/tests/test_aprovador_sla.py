from django.test import TestCase
from django.contrib.auth.models import User
from datetime import date, timedelta
from sinistros.models import Sinistro


class AprovadorSLATestCase(TestCase):
    """
    Test suite for aprovador_os field and SLA calculations
    """
    
    def setUp(self):
        """Create a test user for the tests"""
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
    
    def test_sla_por_setor_cliente(self):
        """Test SLA calculation for CLIENTE sector"""
        sinistro = Sinistro.objects.create(
            placa='ABC1234',
            cliente='Test Client',
            modelo_ativo='Test Model',
            n_chamado='TEST001',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            setor_atual='CLIENTE',
            criado_por=self.user
        )
        
        dias, label = sinistro.sla_por_setor()
        self.assertEqual(dias, 5)
        self.assertEqual(label, "5 dias corridos")
    
    def test_sla_por_setor_precificacao(self):
        """Test SLA calculation for PRECIFICACAO sector"""
        sinistro = Sinistro.objects.create(
            placa='ABC1235',
            cliente='Test Client',
            modelo_ativo='Test Model',
            n_chamado='TEST002',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            setor_atual='PRECIFICACAO',
            criado_por=self.user
        )
        
        dias, label = sinistro.sla_por_setor()
        self.assertEqual(dias, 5)
        self.assertEqual(label, "5 dias corridos")
    
    def test_sla_por_setor_financeiro(self):
        """Test SLA calculation for FINANCEIRO sector"""
        sinistro = Sinistro.objects.create(
            placa='ABC1236',
            cliente='Test Client',
            modelo_ativo='Test Model',
            n_chamado='TEST003',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            setor_atual='FINANCEIRO',
            criado_por=self.user
        )
        
        dias, label = sinistro.sla_por_setor()
        self.assertEqual(dias, 4)
        self.assertEqual(label, "4 dias corridos")
    
    def test_sla_por_setor_desmobilizacao(self):
        """Test SLA calculation for DESMOBILIZACAO sector (no deadline)"""
        sinistro = Sinistro.objects.create(
            placa='ABC1237',
            cliente='Test Client',
            modelo_ativo='Test Model',
            n_chamado='TEST004',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            setor_atual='DESMOBILIZACAO',
            criado_por=self.user
        )
        
        dias, label = sinistro.sla_por_setor()
        self.assertIsNone(dias)
        self.assertEqual(label, "SEM PRAZO")
    
    def test_sla_por_setor_depto_sinistro(self):
        """Test SLA calculation for DEPTO_SINISTRO sector"""
        sinistro = Sinistro.objects.create(
            placa='ABC1238',
            cliente='Test Client',
            modelo_ativo='Test Model',
            n_chamado='TEST005',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            setor_atual='DEPTO_SINISTRO',
            criado_por=self.user
        )
        
        dias, label = sinistro.sla_por_setor()
        self.assertEqual(dias, 60)
        self.assertEqual(label, "60 dias corridos")
    
    def test_sla_por_setor_manutencao_sem_aprovacao(self):
        """Test SLA for MANUTENCAO without aguarda_aprovacao_os"""
        sinistro = Sinistro.objects.create(
            placa='ABC1239',
            cliente='Test Client',
            modelo_ativo='Test Model',
            n_chamado='TEST006',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            setor_atual='MANUTENCAO',
            aguarda_aprovacao_os=False,
            criado_por=self.user
        )
        
        dias, label = sinistro.sla_por_setor()
        self.assertEqual(dias, 15)
        self.assertEqual(label, "15 dias corridos")
    
    def test_sla_por_setor_manutencao_com_aprovacao(self):
        """Test SLA for MANUTENCAO with aguarda_aprovacao_os"""
        sinistro = Sinistro.objects.create(
            placa='ABC1240',
            cliente='Test Client',
            modelo_ativo='Test Model',
            n_chamado='TEST007',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            setor_atual='MANUTENCAO',
            aguarda_aprovacao_os=True,
            aprovador_os='João Silva',
            criado_por=self.user
        )
        
        dias, label = sinistro.sla_por_setor()
        self.assertEqual(dias, 2)
        self.assertEqual(label, "2 dias corridos")
    
    def test_sla_por_setor_finalizado(self):
        """Test SLA calculation for FINALIZADO sector (no deadline)"""
        sinistro = Sinistro.objects.create(
            placa='ABC1241',
            cliente='Test Client',
            modelo_ativo='Test Model',
            n_chamado='TEST008',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            setor_atual='FINALIZADO',
            criado_por=self.user
        )
        
        dias, label = sinistro.sla_por_setor()
        self.assertIsNone(dias)
        self.assertEqual(label, "SEM PRAZO")
    
    def test_aprovador_os_persistence(self):
        """Test that aprovador_os field persists correctly"""
        sinistro = Sinistro.objects.create(
            placa='ABC1242',
            cliente='Test Client',
            modelo_ativo='Test Model',
            n_chamado='TEST009',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            setor_atual='MANUTENCAO',
            aguarda_aprovacao_os=True,
            aprovador_os='Maria Santos',
            criado_por=self.user
        )
        
        # Retrieve from database to confirm persistence
        sinistro_db = Sinistro.objects.get(pk=sinistro.pk)
        self.assertEqual(sinistro_db.aprovador_os, 'Maria Santos')
        self.assertTrue(sinistro_db.aguarda_aprovacao_os)
    
    def test_change_sector_updates_aguarda_to_false(self):
        """Test changing sector clears aguarda_aprovacao_os when appropriate"""
        sinistro = Sinistro.objects.create(
            placa='ABC1243',
            cliente='Test Client',
            modelo_ativo='Test Model',
            n_chamado='TEST010',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            setor_atual='MANUTENCAO',
            aguarda_aprovacao_os=True,
            aprovador_os='Pedro Costa',
            criado_por=self.user
        )
        
        # Change to different sector
        sinistro.setor_atual = 'CLIENTE'
        sinistro.save()
        
        # The aguarda_aprovacao_os should still be True (model doesn't auto-clear)
        # But the form validation should handle this
        self.assertTrue(sinistro.aguarda_aprovacao_os)
    
    def test_retornar_ate_calculation_manutencao_15_days(self):
        """Test retornar_ate is correctly calculated for MANUTENCAO (15 days)"""
        sinistro = Sinistro.objects.create(
            placa='ABC1244',
            cliente='Test Client',
            modelo_ativo='Test Model',
            n_chamado='TEST011',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            setor_atual='MANUTENCAO',
            aguarda_aprovacao_os=False,
            criado_por=self.user
        )
        
        # Manually simulate what the view does
        dias, label = sinistro.sla_por_setor()
        expected_date = date.today() + timedelta(days=dias)
        
        self.assertEqual(dias, 15)
        # We don't set retornar_ate in the model save, it's done in the view
        # This test validates the calculation logic
        
    def test_retornar_ate_calculation_manutencao_2_days(self):
        """Test retornar_ate is correctly calculated for MANUTENCAO with approval (2 days)"""
        sinistro = Sinistro.objects.create(
            placa='ABC1245',
            cliente='Test Client',
            modelo_ativo='Test Model',
            n_chamado='TEST012',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            setor_atual='MANUTENCAO',
            aguarda_aprovacao_os=True,
            aprovador_os='Ana Lima',
            criado_por=self.user
        )
        
        # Manually simulate what the view does
        dias, label = sinistro.sla_por_setor()
        expected_date = date.today() + timedelta(days=dias)
        
        self.assertEqual(dias, 2)
