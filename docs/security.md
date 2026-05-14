# Segurança de autenticação e credenciais

## Hash de senha e salt

- Algoritmo: **Argon2** (`vamos.hashers.ConfigurableArgon2PasswordHasher`)
- Parâmetros configuráveis via ambiente:
  - `DJANGO_ARGON2_TIME_COST`
  - `DJANGO_ARGON2_MEMORY_COST`
  - `DJANGO_ARGON2_PARALLELISM`
- Justificativa: Argon2 é resistente a ataques de força bruta por custo de memória/CPU ajustável.
- Salt por usuário: gerado automaticamente pelo hasher (armazenado no hash serializado no campo de senha do Django).

## 2FA

- Biblioteca: `django-two-factor-auth` + `django-otp`
- Fluxo obrigatório: rota `/login/` redireciona para `two_factor:login`.
- Validação 2FA ocorre após autenticação primária e antes da sessão final.

## Sessão

- `SESSION_COOKIE_AGE=1800`
- `SESSION_EXPIRE_AT_BROWSER_CLOSE=True`
- `SESSION_SAVE_EVERY_REQUEST=True`
- Logout invalida sessão (`logout_view`).

## Força bruta

- `django-axes` habilitado em middleware/backend.
- Configurável por ambiente:
  - `AXES_FAILURE_LIMIT`
  - `AXES_COOLOFF_TIME`

## Evidências automatizadas

- Testes em `vamos/tests.py` validam:
  - aceite e revogação de consentimento
  - exportação de dados
  - exclusão de dados
  - redirecionamento de login para fluxo 2FA
