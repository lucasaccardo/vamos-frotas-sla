from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from .models import Ticket


class LGPDViewsTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="lgpduser",
            email="lgpd@example.com",
            password="SenhaSegura123!",
        )
        self.client.force_login(self.user)

    def test_termos_aceite_registra_versao_e_hash(self):
        response = self.client.post(reverse("termos_uso"))
        self.assertEqual(response.status_code, 302)

        self.user.refresh_from_db()
        self.assertIsNotNone(self.user.perfil.termos_aceitos_em)
        self.assertTrue(self.user.perfil.termos_versao)
        self.assertEqual(len(self.user.perfil.termos_hash), 64)

    def test_exportar_meus_dados_retorna_json(self):
        response = self.client.get(reverse("exportar_meus_dados"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response["Content-Type"])
        self.assertIn("lgpduser", response.content.decode("utf-8"))

    def test_solicitar_exclusao_cria_ticket(self):
        response = self.client.post(reverse("solicitar_exclusao_dados"))
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            Ticket.objects.filter(
                usuario=self.user,
                titulo="Solicitação LGPD - exclusão de dados",
            ).exists()
        )
