"""
Testes para a funcionalidade de exportação de relatórios (XLSX e CSV).

Verifica:
1. Acesso protegido (apenas is_staff)
2. Geração de arquivo XLSX com Content-Type correto
3. Geração de arquivo CSV com Content-Type correto
4. Aplicação de filtros
"""
from datetime import date, timedelta
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from sinistros.models import Sinistro


class ExportXLSXTestCase(TestCase):
    """Testes para exportação de relatórios em Excel e CSV"""

    def setUp(self):
        """Configuração inicial dos testes"""
        # Criar usuário staff
        self.staff_user = User.objects.create_user(
            username='staffuser',
            password='testpass123',
            is_staff=True
        )
        
        # Criar usuário normal (sem staff)
        self.normal_user = User.objects.create_user(
            username='normaluser',
            password='testpass123',
            is_staff=False
        )
        
        # Criar alguns sinistros para testar
        self.sinistro1 = Sinistro.objects.create(
            placa='ABC1234',
            cliente='Cliente A',
            modelo_ativo='Modelo X',
            n_chamado='CH001',
            data_ocorrencia=date.today(),
            motivo='COLISAO',
            segmento='PESADOS',
            setor_atual='CLIENTE',
            criado_por=self.staff_user,
            total_pago=1000.00
        )
        
        self.sinistro2 = Sinistro.objects.create(
            placa='DEF5678',
            cliente='Cliente B',
            modelo_ativo='Modelo Y',
            n_chamado='CH002',
            data_ocorrencia=date.today() - timedelta(days=10),
            motivo='FURTO_ROUBO',
            segmento='AGRO',
            setor_atual='MANUTENCAO',
            criado_por=self.staff_user,
            aguarda_aprovacao_os=True,
            aprovador_os='João Silva',
            total_pago=0.00
        )
        
        self.client = Client()

    def test_exportar_relatorios_view_requires_staff(self):
        """Testa que a view de relatórios requer usuário staff"""
        # Sem login
        response = self.client.get(reverse('exportar_relatorios'))
        self.assertEqual(response.status_code, 302)  # Redirect to login
        
        # Com usuário normal (não staff)
        self.client.login(username='normaluser', password='testpass123')
        response = self.client.get(reverse('exportar_relatorios'))
        self.assertEqual(response.status_code, 302)  # Redirect (sem permissão)
        
        # Com usuário staff
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('exportar_relatorios'))
        self.assertEqual(response.status_code, 200)  # OK

    def test_exportar_xlsx_requires_staff(self):
        """Testa que a exportação XLSX requer usuário staff"""
        # Sem login
        response = self.client.get(reverse('exportar_xlsx'))
        self.assertEqual(response.status_code, 302)  # Redirect to login
        
        # Com usuário normal (não staff)
        self.client.login(username='normaluser', password='testpass123')
        response = self.client.get(reverse('exportar_xlsx'))
        self.assertEqual(response.status_code, 302)  # Redirect (permission denied)

    def test_exportar_xlsx_content_type(self):
        """Testa que a exportação XLSX retorna Content-Type correto"""
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('exportar_xlsx'))
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response['Content-Type'],
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        self.assertIn('attachment', response['Content-Disposition'])
        self.assertIn('.xlsx', response['Content-Disposition'])

    def test_exportar_csv_content_type(self):
        """Testa que a exportação CSV retorna Content-Type correto"""
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('exportar_csv'))
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/csv', response['Content-Type'])
        self.assertIn('attachment', response['Content-Disposition'])
        self.assertIn('.csv', response['Content-Disposition'])

    def test_exportar_xlsx_with_setor_filter(self):
        """Testa exportação XLSX com filtro de setor"""
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('exportar_xlsx'), {'setor': 'MANUTENCAO'})
        
        self.assertEqual(response.status_code, 200)
        # Verificar que retornou dados (não vazio)
        self.assertGreater(len(response.content), 0)

    def test_exportar_csv_with_apenas_pagos_filter(self):
        """Testa exportação CSV com filtro apenas_pagos"""
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('exportar_csv'), {'apenas_pagos': 'on'})
        
        self.assertEqual(response.status_code, 200)
        content = response.content.decode('utf-8')
        
        # Verificar que o cabeçalho está presente
        self.assertIn('Placa', content)
        # Verificar que tem dados (sinistro1 tem total_pago > 0)
        self.assertIn('ABC1234', content)

    def test_exportar_xlsx_with_cliente_filter(self):
        """Testa exportação XLSX com filtro de cliente"""
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('exportar_xlsx'), {'cliente': 'Cliente A'})
        
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.content), 0)

    def test_exportar_csv_with_date_range_filter(self):
        """Testa exportação CSV com filtro de intervalo de datas"""
        self.client.login(username='staffuser', password='testpass123')
        
        # Filtrar últimos 5 dias
        data_inicio = (date.today() - timedelta(days=5)).strftime('%Y-%m-%d')
        data_fim = date.today().strftime('%Y-%m-%d')
        
        response = self.client.get(reverse('exportar_csv'), {
            'campo_data': 'data_ocorrencia',
            'data_inicio': data_inicio,
            'data_fim': data_fim
        })
        
        self.assertEqual(response.status_code, 200)
        content = response.content.decode('utf-8')
        
        # Sinistro1 está dentro do intervalo
        self.assertIn('ABC1234', content)
        # Sinistro2 está fora do intervalo (10 dias atrás)
        self.assertNotIn('DEF5678', content)

    def test_exportar_xlsx_with_segmento_filter(self):
        """Testa exportação XLSX com filtro de segmento"""
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('exportar_xlsx'), {'segmento': 'PESADOS'})
        
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.content), 0)

    def test_exportar_csv_includes_aprovador_os(self):
        """Testa que CSV inclui campo aprovador_os"""
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('exportar_csv'))
        
        content = response.content.decode('utf-8')
        
        # Verificar header
        self.assertIn('Aprovador O.S.', content)
        # Verificar dados
        self.assertIn('João Silva', content)

    def test_exportar_csv_includes_sla_label(self):
        """Testa que CSV inclui label de SLA"""
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('exportar_csv'))
        
        content = response.content.decode('utf-8')
        
        # Verificar header
        self.assertIn('SLA', content)
        # Verificar que tem algum label de SLA (ex: "5 dias corridos" ou "2 dias corridos")
        self.assertTrue(
            'dias corridos' in content or 'SEM PRAZO' in content
        )

    def test_exportar_relatorios_view_renders_template(self):
        """Testa que a view de relatórios renderiza o template correto"""
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('exportar_relatorios'))
        
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'sinistros/exportar.html')
        # Verificar que contexto tem setores e segmentos
        self.assertIn('setores', response.context)
        self.assertIn('segmentos', response.context)

    def test_exportar_xlsx_filename_includes_timestamp(self):
        """Testa que o nome do arquivo XLSX inclui timestamp"""
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('exportar_xlsx'))
        
        content_disposition = response['Content-Disposition']
        self.assertIn('relatorio_sinistros_', content_disposition)
        self.assertIn('.xlsx', content_disposition)

    def test_exportar_csv_filename_includes_timestamp(self):
        """Testa que o nome do arquivo CSV inclui timestamp"""
        self.client.login(username='staffuser', password='testpass123')
        response = self.client.get(reverse('exportar_csv'))
        
        content_disposition = response['Content-Disposition']
        self.assertIn('relatorio_sinistros_', content_disposition)
        self.assertIn('.csv', content_disposition)
