# Checklist de conformidade (1.1–8.7)

## 1. Autenticação e Gestão de Credenciais
- [x] **1.1** Hash seguro: `vamos_frotas_sla/settings.py`, `vamos/hashers.py`
- [x] **1.2** Custo configurável: `vamos/hashers.py`, `.env.example`, `docs/security.md`
- [x] **1.3** Salt único por usuário: `docs/security.md` (hasher Argon2/Django)
- [x] **1.4** Armazenamento hash+salt: `docs/security.md`
- [x] **1.5** 2FA implementado: `vamos_frotas_sla/settings.py`, `vamos_frotas_sla/urls.py`
- [x] **1.6** 2FA após autenticação primária: `vamos/views.py` (`login_view`)
- [x] **1.7** Fluxo documentado: `docs/architecture.md`, `docs/security.md`
- [x] **1.8** Evidências funcionais: `vamos/tests.py`, `docs/security-tests.md`
- [x] **1.9** Expiração de sessão: `vamos_frotas_sla/settings.py`
- [x] **1.10** Logout invalida sessão: `vamos/views.py` (`logout_view`)
- [x] **1.11** Proteção força bruta: `vamos_frotas_sla/settings.py`
- [x] **1.12** Justificativas técnicas: `docs/security.md`

## 2. Recuperação de Senha
- [x] **2.1** Funcionalidade implementada: `vamos/urls.py`, `vamos/auth_views.py`
- [x] **2.2** Token seguro: fluxo nativo Django (`PasswordReset*`), `docs/security.md`
- [x] **2.3** Expiração de token: `vamos_frotas_sla/settings.py` (`PASSWORD_RESET_TIMEOUT`)
- [x] **2.4** Invalidação após uso: fluxo nativo Django
- [x] **2.5** Falha para token expirado: `vamos/auth_views.py`
- [x] **2.6** Log de solicitação: `vamos/auth_views.py`
- [x] **2.7** Log de sucesso/falha: `vamos/auth_views.py`

## 3. Criptografia e Comunicação Segura
- [x] **3.1** TLS/HTTPS: `vamos_frotas_sla/settings.py`, `infra/nginx/local-https.conf`
- [x] **3.2** Bloqueio não seguro: `SECURE_SSL_REDIRECT` em `settings.py`
- [x] **3.3** Evidência de tráfego cifrado: `docs/crypto.md`
- [x] **3.4** Dados sensíveis em repouso: `sinistros/models.py`
- [x] **3.5** Algoritmo adequado (AES): `docs/crypto.md`
- [x] **3.6** Chaves protegidas: `.env.example`, `docs/crypto.md`
- [x] **3.7** Estratégia documentada: `docs/crypto.md`
- [x] **3.8** Justificativas técnicas: `docs/crypto.md`

## 4. Conformidade LGPD
- [x] **4.1** Dados pessoais listados: `docs/lgpd.md`
- [x] **4.2** Dado × finalidade: `docs/lgpd.md`
- [x] **4.3** Minimização: `docs/lgpd.md`
- [x] **4.4** Consentimento explícito: `vamos/views.py` (`termos_uso_view`)
- [x] **4.5** Consentimento associado à finalidade: `docs/lgpd.md`
- [x] **4.6** Revogação de consentimento: `vamos/views.py` (`revogar_consentimento_view`)
- [x] **4.7** Data e versão do consentimento: `vamos/models.py`
- [x] **4.8** Consulta de dados: `vamos/views.py` (`meus_dados_view`)
- [x] **4.9** Exportação de dados: `vamos/views.py` (`exportar_meus_dados_view`)
- [x] **4.10** Exclusão de dados pessoais: `vamos/views.py` (`excluir_meus_dados_view`)
- [x] **4.11** Fluxo documentado: `docs/lgpd.md`

## 5. Auditoria e Logs
- [x] **5.1** Logs de autenticação: `vamos/security_logging.py`
- [x] **5.2** Logs falhas e 2FA: `vamos/security_logging.py`
- [x] **5.3** Proteção de logs: `vamos/apps.py`, `settings.py` (append-only + permissão restrita)
- [x] **5.4** Exemplo de análise: `docs/security-tests.md`

## 6. Documentação Técnico-Científica
- [x] **6.1** Visão geral: `docs/overview.md`
- [x] **6.2** Diagrama de arquitetura: `docs/architecture.md`
- [x] **6.3** Fluxos documentados: `docs/architecture.md`
- [x] **6.4** Gestão de credenciais: `docs/security.md`
- [x] **6.5** Criptografia documentada: `docs/crypto.md`
- [x] **6.6** Ativos identificados: `docs/risk-analysis.md`
- [x] **6.7** Ameaças/vulnerabilidades: `docs/risk-analysis.md`
- [x] **6.8** Risco × contramedida: `docs/risk-analysis.md`
- [x] **6.9** Testes de segurança realizados: `docs/security-tests.md`
- [x] **6.10** Resultados documentados: `docs/security-tests.md`
- [x] **6.11** Uso de normas/artigos: `docs/references.md`
- [x] **6.12** Referências normalizadas: `docs/references.md`

## 7. Resumo Científico
- [x] **7.1** 200–300 palavras: `docs/abstract.md`
- [x] **7.2** Objetivo definido: `docs/abstract.md`
- [x] **7.3** Metodologia descrita: `docs/abstract.md`
- [x] **7.4** Mecanismos de segurança: `docs/abstract.md`
- [x] **7.5** Conformidade LGPD: `docs/abstract.md`
- [x] **7.6** Terminologia adequada: `docs/abstract.md`
- [x] **7.7** Qualidade textual científica: `docs/abstract.md`

## 8. Pôster Científico e Apresentação
- [x] **8.1** Estrutura científica do pôster: `docs/poster.md`
- [x] **8.2** Arquitetura/fluxos visuais: `docs/poster.md`, `docs/architecture.md`
- [x] **8.3** Conformidade LGPD evidenciada: `docs/poster.md`, `docs/lgpd.md`
- [x] **8.4** Qualidade técnica dos diagramas: `docs/architecture.md`
- [x] **8.5** Coerência com sistema: `docs/checklist.md`
- [x] **8.6** Domínio técnico na apresentação: `docs/poster.md` (roteiro de fala)
- [x] **8.7** Respostas a perguntas: `docs/poster.md` (seção de mensagens-chave)
