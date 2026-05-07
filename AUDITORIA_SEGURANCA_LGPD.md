# Auditoria de Segurança e LGPD

Status: **Atendido / Parcial / Ausente**  
Escopo: requisitos acadêmicos 1.x a 6.x.

| Requisito | Status | Evidência |
|---|---|---|
| 1.1–1.4 Hash seguro + salt + armazenamento | Atendido | `vamos_frotas_sla/settings.py` (`PASSWORD_HASHERS`, `ARGON2_*`) |
| 1.5–1.6 2FA implementado e validado após login primário | Atendido | `vamos_frotas_sla/settings.py` (`django_otp`, `two_factor`, `OTPMiddleware`), `vamos_frotas_sla/urls.py` |
| 1.7 Fluxo de autenticação documentado | Parcial | Este relatório + `docs/LGPD.md` (resumo operacional) |
| 1.8 Evidências funcionais (prints/logs/testes) | Parcial | Logs configurados e testes adicionados; prints dependem coleta manual |
| 1.9 Sessão com expiração | Atendido | `vamos_frotas_sla/settings.py` (`SESSION_COOKIE_AGE`, `SESSION_EXPIRE_AT_BROWSER_CLOSE`) |
| 1.10 Invalidação no logout | Atendido | `vamos/views.py` (`logout_view`) + logs em `vamos/security_logging.py` |
| 1.11 Proteção força bruta | Atendido | `vamos_frotas_sla/settings.py` (`axes`, `AXES_FAILURE_LIMIT`, `AXES_COOLOFF_TIME`) |
| 1.12 Justificativas técnicas | Atendido | Comentários e parâmetros de hash em `settings.py` + este relatório |
| 2.1 Recuperação de senha implementada | Atendido | `vamos/urls.py` e templates `vamos/templates/vamos/password_reset*.html` |
| 2.2 Token criptograficamente seguro | Atendido | Fluxo nativo Django (`PasswordResetConfirmView`) |
| 2.3 Token com expiração | Atendido | `vamos_frotas_sla/settings.py` (`PASSWORD_RESET_TIMEOUT=3600`) |
| 2.4 Token invalidado após uso | Atendido | Fluxo nativo Django (`PasswordResetConfirmView`) |
| 2.5 Falha para token expirado | Atendido | `vamos/auth_views.py` (`password_reset_invalid_or_expired`) |
| 2.6 Log de solicitação de recuperação | Atendido | `vamos/auth_views.py` (`password_reset_requested`) |
| 2.7 Log de sucesso/falha do processo | Atendido | `vamos/auth_views.py` (`password_reset_success`, `invalid_or_expired`) |
| 3.1–3.2 TLS/HTTPS + bloqueio não seguro | Atendido | `vamos_frotas_sla/settings.py` (`SECURE_SSL_REDIRECT`, `SECURE_PROXY_SSL_HEADER`, HSTS) |
| 3.3 Evidência de tráfego cifrado | Parcial | Requer print manual no browser (cadeado HTTPS e aba Network) |
| 3.4–3.6 Criptografia em repouso e proteção de chaves | Parcial | `sinistros/models.py` (campos com `encrypt`) e S3 AES256 em `settings.py`; rotação/gestão de chaves depende infraestrutura |
| 3.7–3.8 Estratégia e justificativas documentadas | Atendido | `SECURITY_SUMMARY.md`, este relatório e comentários no `settings.py` |
| 4.1 Lista de dados pessoais coletados | Atendido | `docs/LGPD.md` |
| 4.2 Associação dado × finalidade | Atendido | `docs/LGPD.md` |
| 4.3 Evidência de minimização | Atendido | `docs/LGPD.md` + views de dados do titular (`vamos/views.py`) |
| 4.4–4.5 Consentimento explícito e associado à finalidade | Atendido | `vamos/templates/vamos/termos.html`, `vamos/views.py` (`termos_uso_view`) |
| 4.6 Revogação do consentimento | Parcial | Fluxo via solicitação administrativa (`solicitar_exclusao_dados_view`) |
| 4.7 Registro de data e versão do consentimento | Atendido | `vamos/models.py` (`termos_aceitos_em`, `termos_versao`, `termos_hash`) |
| 4.8 Consulta dos dados do titular | Atendido | `vamos/urls.py` + `vamos/views.py` (`meus_dados_view`) |
| 4.9 Exportação dos dados | Atendido | `vamos/views.py` (`exportar_meus_dados_view`) |
| 4.10 Exclusão de dados pessoais | Parcial | `vamos/views.py` (`solicitar_exclusao_dados_view`) com fila administrativa |
| 4.11 Fluxo de direitos documentado | Atendido | `docs/LGPD.md` |
| 5.1 Logs de autenticação | Atendido | `vamos/security_logging.py` + `LOGGING` em `settings.py` |
| 5.2 Logs de falhas e 2FA | Atendido | `vamos/security_logging.py` |
| 5.3 Proteção contra alteração de logs | Parcial | Saída em console; recomendada centralização externa imutável |
| 5.4 Exemplo de análise de logs | Parcial | Recomendado extrair evidência em ambiente de execução (manual) |
| 6.1 Visão geral do sistema | Atendido | `IMPLEMENTATION_SUMMARY.md`, `SECURITY_SUMMARY.md` |
| 6.2 Diagrama de arquitetura | Atendido | `LAYOUT_DIAGRAM.md` |
| 6.3 Fluxos de autenticação e dados | Parcial | Parcialmente em docs; recomendado diagrama dedicado |
| 6.4 Gestão de credenciais documentada | Atendido | `settings.py` + este relatório |
| 6.5 Uso de criptografia documentado | Atendido | `SECURITY_SUMMARY.md`, `settings.py`, `sinistros/models.py` |
| 6.6 Ativos do sistema identificados | Parcial | Recomendado inventário formal de ativos |
| 6.7 Ameaças e vulnerabilidades identificadas | Parcial | Evidências pontuais; recomendado documento de threat model |
| 6.8 Associação risco × contramedida | Parcial | Parcial neste relatório; recomendado matriz dedicada |
| 6.9–6.10 Testes de segurança realizados/documentados | Parcial | Testes automatizados existentes; segurança requer suíte dedicada |
| 6.11 Uso de normas/artigos técnicos | Parcial | Recomendado anexar referências formais no relatório científico |
| 6.12 Referências normalizadas | Ausente | Recomendado incluir seção ABNT |

## Evidências manuais recomendadas (prints)

1. Cadeado HTTPS no navegador em produção.
2. Aba Network com requisições `https://`.
3. Tela de lockout do `django-axes` após tentativas inválidas.
4. Fluxo de reset com token inválido/expirado.
5. Tela **Meus dados (LGPD)** com ações de consulta/exportação/solicitação de exclusão.
