# Visão Geral Técnica-Científica de Segurança

## 1. Arquitetura do sistema

- **Backend:** Django 4.2 (apps `vamos`, `sinistros`, `accounts`, `procedures`).
- **Autenticação:** `django-two-factor-auth` + `django-otp`.
- **Proteção de brute force:** `django-axes`.
- **Dados sensíveis em repouso:** `django-cryptography` (`encrypt(...)`) em campos críticos de sinistros/frota.
- **Armazenamento de senha:** Argon2 configurável (`vamos.hashers.ConfigurableArgon2PasswordHasher`).

Fluxo macro: Usuário → HTTPS/Reverse proxy → Django (sessão + 2FA) → Banco relacional + trilha de auditoria.

## 2. Gestão de credenciais e autenticação

- **Hash de senha:** Argon2 (primário), com parâmetros configuráveis por ambiente:
  - `DJANGO_ARGON2_TIME_COST` (padrão 3)
  - `DJANGO_ARGON2_MEMORY_COST` (padrão 102400)
  - `DJANGO_ARGON2_PARALLELISM` (padrão 8)
- **Salt:** gerenciado automaticamente pelo hasher do Django/Argon2, único por credencial.
- **2FA:** login via fluxo do `two_factor` (rota `login` redireciona para `two_factor:login`).
- **Sessão:** expiração em 30 minutos (`SESSION_COOKIE_AGE=1800`) e invalidação via logout.
- **Brute force:** bloqueio após 5 falhas e cooloff de 1h (`AXES_FAILURE_LIMIT`, `AXES_COOLOFF_TIME`).

## 3. Recuperação de senha

- Fluxo com token de alta entropia do Django (`PasswordResetTokenGenerator`).
- Expiração de token em 1 hora (`PASSWORD_RESET_TIMEOUT=3600`).
- Invalidação automática após uso (comportamento nativo).
- Logs de solicitação, sucesso e token inválido/expirado em `vamos.auth_views`.

## 4. Criptografia e comunicação segura

- **Em trânsito:** HTTPS obrigatório em produção (`SECURE_SSL_REDIRECT`, HSTS e proxy header).
- **Em repouso:** campos sensíveis de domínio com `encrypt()`; upload em S3 com `ServerSideEncryption: AES256`.
- **Chaves:** segredos via variáveis de ambiente (`DJANGO_SECRET_KEY` e credenciais de infraestrutura).

## 5. LGPD: dados, finalidades e direitos do titular

- Mapeamento de dados e finalidades: `docs/LGPD.md`.
- Consentimento explícito com data, versão e hash dos termos.
- Revogação de consentimento disponível via endpoint dedicado.
- Direitos implementados:
  - Consulta dos próprios dados;
  - Exportação JSON;
  - Exclusão por ticket administrativo;
  - Exclusão imediata de conta/dados no app.

## 6. Ativos, ameaças, vulnerabilidades e contramedidas

| Ativo | Ameaça/Vulnerabilidade | Contramedida |
|---|---|---|
| Credenciais de usuários | Quebra de senha por força bruta | Argon2 + `django-axes` |
| Sessão autenticada | Sequestro de sessão | Cookies seguros, HttpOnly, expiração curta |
| Dados pessoais | Exposição em trânsito | HTTPS obrigatório + HSTS |
| Dados sensíveis no banco | Vazamento em repouso | Criptografia em campo (`encrypt`) |
| Processo de reset | Reuso/token indevido | Token temporário + expiração + invalidação |
| Trilha de segurança | Alteração maliciosa de logs | Handler append-only com hash encadeado (`HashChainAuditHandler`) |

## 7. Testes de segurança executados

- Testes automatizados em `vamos/tests.py`:
  - aceite/revogação de consentimento;
  - exportação e exclusão de dados;
  - redirecionamento para fluxo 2FA;
  - token inválido/expirado em reset;
  - encadeamento de hash nos logs de auditoria.

## 8. Fundamentação técnico-científica (APA)

- OWASP Foundation. (2023). *OWASP ASVS 4.0.3*. https://owasp.org/www-project-application-security-verification-standard/
- National Institute of Standards and Technology. (2017). *NIST SP 800-63B: Digital Identity Guidelines – Authentication and Lifecycle Management*. https://doi.org/10.6028/NIST.SP.800-63b
- International Organization for Standardization. (2022). *ISO/IEC 27001:2022 Information security management systems*.
- International Organization for Standardization. (2022). *ISO/IEC 27002:2022 Information security controls*.
- Krawczyk, H., Bellare, M., & Canetti, R. (1997). *HMAC: Keyed-Hashing for Message Authentication (RFC 2104)*.
