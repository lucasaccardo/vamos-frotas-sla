# LGPD (Lei 13.709/2018)

## Inventário de dados pessoais e finalidades

| Dado | Origem | Finalidade |
|---|---|---|
| username | cadastro | autenticação e rastreabilidade |
| email | cadastro | recuperação de senha e contato de segurança |
| hash de senha | autenticação | proteção de credenciais |
| matrícula (opcional) | perfil | identificação interna mínima |
| aceite de termos (data/versão/hash) | consentimento | prova de consentimento e accountability |
| dados operacionais vinculados ao usuário | uso da plataforma | operação e suporte |
| logs de segurança | eventos de autenticação/LGPD | auditoria e resposta a incidentes |

## Minimização

- Apenas dados necessários para autenticação, operação e auditoria são coletados.
- Campos opcionais permanecem opcionais (ex.: matrícula/foto).
- Funcionalidades LGPD expõem apenas dados do titular autenticado.

## Consentimento

- Captura explícita em `termos_uso_view`.
- Armazena data/hora, versão e hash do termo aceito.
- Revogação disponível em `revogar_consentimento_view`.

## Direitos do titular

- **Consulta**: `meus_dados_view`
- **Exportação**: `exportar_meus_dados_view` (JSON)
- **Revogação de consentimento**: `revogar_consentimento_view`
- **Exclusão**:
  - fluxo assistido: `solicitar_exclusao_dados_view`
  - autoatendimento: `excluir_meus_dados_view`

## Fluxo de atendimento

1. Titular autenticado acessa “Meus dados”.
2. Solicita consulta/exportação/revogação/exclusão.
3. Sistema executa ação e registra evento de auditoria.
4. Em exclusão assistida, equipe administrativa recebe ticket.
