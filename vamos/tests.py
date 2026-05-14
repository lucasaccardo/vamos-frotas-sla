import json
import logging
import os
import tempfile

from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse

from .logging_handlers import HashChainAuditHandler
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

    def test_revogar_consentimento_limpa_registro(self):
        self.client.post(reverse("termos_uso"))
        response = self.client.post(reverse("revogar_consentimento"))
        self.assertEqual(response.status_code, 302)

        self.user.refresh_from_db()
        self.assertIsNone(self.user.perfil.termos_aceitos_em)
        self.assertEqual(self.user.perfil.termos_versao, "")
        self.assertEqual(self.user.perfil.termos_hash, "")

    def test_excluir_meus_dados_remove_conta(self):
        response = self.client.post(reverse("excluir_meus_dados"))
        self.assertEqual(response.status_code, 302)
        self.assertFalse(User.objects.filter(id=self.user.id).exists())

    def test_login_route_redirects_to_two_factor(self):
        self.client.logout()
        response = self.client.get(reverse("login"))
        self.assertEqual(response.status_code, 302)
        self.assertNotEqual(response.url, reverse("login"))

    def test_password_reset_token_invalido_exibe_erro(self):
        self.client.logout()
        response = self.client.get(
            reverse(
                "password_reset_confirm",
                kwargs={"uidb64": "MQ", "token": "invalid-token"},
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Link inválido ou expirado")


class SecurityAuditLogChainTestCase(TestCase):
    def test_hash_chain_audit_handler_creates_linked_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, "audit.log")
            handler = HashChainAuditHandler(logfile)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger = logging.getLogger("vamos.security.test")
            logger.setLevel(logging.INFO)
            logger.handlers = [handler]
            logger.propagate = False

            logger.info("evento-1")
            logger.info("evento-2")

            with open(logfile, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["previous_hash"], "GENESIS")
            self.assertEqual(rows[1]["previous_hash"], rows[0]["hash"])
