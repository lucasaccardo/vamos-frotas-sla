# Testes de segurança e evidências

## Execução automatizada

Comando (focado nas mudanças de segurança/LGPD):

```bash
python manage.py test vamos.tests
```

## Cenários cobertos

- Registro de aceite de termos com versão/hash.
- Exportação de dados do titular em JSON.
- Registro de solicitação de exclusão (ticket LGPD).
- Revogação de consentimento.
- Exclusão de dados pessoais (remoção da conta do titular).
- Rota `/login/` redirecionando para fluxo 2FA.

## Evidências complementares de auditoria

Eventos registrados em `security_audit.log`:

- `auth_login_success`, `auth_login_failed`, `auth_logout`
- `auth_2fa_verified`
- `password_reset_requested`, `password_reset_success`, `password_reset_invalid_or_expired`
- `lgpd_terms_accepted`, `lgpd_consent_revoked`, `lgpd_data_export`, `lgpd_data_deletion_requested`, `lgpd_data_deleted`
