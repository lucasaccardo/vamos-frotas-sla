# Evidências de Segurança e LGPD

## 1) HTTPS/TLS (produção)

Exemplo de validação (ajuste domínio):

```bash
curl -I https://seu-dominio.exemplo
```

Verificar:
- resposta HTTPS;
- cabeçalhos HSTS (`Strict-Transport-Security`);
- redirecionamento automático de HTTP para HTTPS.

## 2) Exemplo de análise de logs de segurança

Logs de segurança são escritos em console e no arquivo encadeado (`SECURITY_AUDIT_LOG_PATH`).

```bash
tail -n 20 logs/security_audit.log
```

Exemplos de eventos esperados:
- `auth_login_success`
- `auth_login_failed`
- `auth_2fa_verified`
- `password_reset_requested`
- `password_reset_invalid_or_expired`
- `password_reset_success`
- `lgpd_terms_accepted`
- `lgpd_terms_revoked`
- `lgpd_data_export`
- `lgpd_data_erasure_started/completed`

## 3) Verificação de integridade (hash encadeado)

Cada linha inclui `previous_hash` e `hash`.  
Para validação, conferir se o `previous_hash` do evento N+1 coincide com o `hash` do evento N.
