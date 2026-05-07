# LGPD - Tratamento de dados pessoais

## 1) Dados pessoais coletados

- **Conta/autenticação:** username, e-mail, hash de senha, status de usuário, último login.
- **Perfil:** matrícula, foto de perfil (opcional), data/hora de aceite dos termos, versão e hash do termo aceito.
- **Operação do sistema:** análises vinculadas ao usuário, tickets e solicitações administrativas.
- **Segurança/auditoria:** eventos de login/logout/falha, eventos de 2FA, solicitações e resultado de reset de senha.

## 2) Finalidade por dado

- **Autenticação e controle de acesso:** login, sessão, autorização por perfil.
- **Operação de negócio:** execução e histórico de análises/tickets.
- **Conformidade e responsabilização:** aceite de termos com rastreabilidade (data, versão e hash).
- **Segurança:** trilha de auditoria para incidentes, brute force e recuperação de senha.

## 3) Minimização

- Coleta restrita ao necessário para autenticação, operação e auditoria.
- Campos opcionais (ex.: foto de perfil) permanecem não obrigatórios.
- Exportação do titular retorna somente dados do próprio usuário autenticado.

## 4) Consentimento (coleta e revogação)

- O aceite é coletado na rota `termos-de-uso/` com registro em `Perfil`:
  - `termos_aceitos_em`
  - `termos_versao`
  - `termos_hash`
- A revogação/exclusão é tratada via solicitação formal do titular (fluxo administrativo) pela UI em `perfil/meus-dados/solicitar-exclusao/`.

## 5) Direitos do titular (consulta, exportação, exclusão)

- **Consulta:** `GET /perfil/meus-dados/`
- **Exportação:** `GET /perfil/meus-dados/exportar/` (JSON)
- **Exclusão:** `POST /perfil/meus-dados/solicitar-exclusao/` (abre ticket LGPD para tratativa)

## 6) Como solicitar atendimento

1. Usuário autenticado acessa **Minha Conta**.
2. Clica em **Consultar dados pessoais (LGPD)**.
3. Escolhe:
   - Exportar JSON dos dados;
   - Solicitar exclusão de dados (ticket de atendimento).

## 7) Base legal e boas práticas adotadas

- Lei nº 13.709/2018 (LGPD).
- Segurança por padrão com variáveis de ambiente para segredos e hosts.
- Logs de segurança sem exposição de dados sensíveis em texto claro.
