# Análise de riscos

## Ativos

- Credenciais de usuário
- Dados pessoais de cadastro/perfil
- Dados operacionais de sinistros
- Sessões autenticadas
- Logs de segurança

## Ameaças e vulnerabilidades

| Ameaça | Vulnerabilidade explorada | Contramedida |
|---|---|---|
| Brute-force de senha | tentativas ilimitadas | `django-axes` com lockout configurável |
| Roubo de sessão | cookie inseguro e sessão longa | cookies seguros em produção + expiração + logout |
| Bypass de MFA | login direto sem 2FA | redirecionamento obrigatório para fluxo 2FA |
| Vazamento de credenciais | hash fraco | Argon2 parametrizável |
| Interceptação de tráfego | HTTP sem proteção | TLS obrigatório + HSTS |
| Exposição de dados sensíveis em banco | armazenamento em claro | campos com `encrypt` |
| Repúdio de operações críticas | ausência de auditoria | logs de autenticação, 2FA, reset, LGPD |
| Manipulação de logs | arquivo aberto ou sobrescrito | escrita append-only + permissão restrita (0700/0600) |
| Não atendimento LGPD | ausência de direitos do titular | endpoints de consulta/exportação/revogação/exclusão |

## Matriz risco × contramedida (resumo)

- **Alto**: brute-force, interceptação de tráfego, bypass de MFA.
- **Médio**: manipulação de logs, repúdio, exposição de dados em repouso.
- **Baixo/Médio**: minimização/documentação incompleta.
