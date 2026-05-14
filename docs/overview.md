# Visão geral do sistema

O **Vamos Frotas SLA** é uma aplicação Django para gestão operacional com autenticação reforçada, auditoria de segurança e funcionalidades LGPD.

## Componentes principais

- **Aplicação web Django** (`vamos_frotas_sla`, `vamos`, `sinistros`, `accounts`, `tickets`, `analyses`, `procedures`)
- **Autenticação**: login com 2FA (`django-two-factor-auth` + `django-otp`)
- **Proteção de credenciais**: Argon2 configurável por ambiente (`vamos/hashers.py`)
- **Proteção contra brute-force**: `django-axes` com limites configuráveis
- **Recuperação de senha segura**: fluxo nativo Django com token temporário e auditoria (`vamos/auth_views.py`)
- **LGPD**: consentimento versionado, consulta/exportação/exclusão e revogação de consentimento (`vamos/views.py`, `vamos/models.py`)
- **Criptografia em repouso**: campos sensíveis com `django-cryptography` (`sinistros/models.py`)
- **Auditoria**: logs de segurança em console e arquivo append-only restrito (`vamos/security_logging.py`, `vamos/apps.py`, `settings.py`)

## Estrutura de diretórios (alto nível)

- `vamos_frotas_sla/`: configuração global
- `vamos/`: autenticação, páginas principais, fluxo LGPD
- `sinistros/`: módulo de sinistros e dados sensíveis
- `docs/`: documentação técnica e científica
- `infra/`: artefatos de infraestrutura (TLS local)

## Fluxo de segurança resumido

1. Usuário inicia login em `/login/` e é redirecionado ao fluxo 2FA.
2. Senha é validada (Argon2 + salt implícito por hash da biblioteca).
3. 2FA é validado antes do acesso autenticado.
4. Sessão tem expiração e logout invalida sessão.
5. Eventos críticos são auditados em log de segurança.
