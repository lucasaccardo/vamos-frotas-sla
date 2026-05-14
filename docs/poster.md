# Pôster científico (roteiro)

## 1. Título
Sistema Seguro de Autenticação, Comunicação e Gestão de Credenciais com Conformidade LGPD

## 2. Problema
Risco de incidentes por autenticação fraca, má gestão de credenciais e ausência de mecanismos de direitos do titular.

## 3. Objetivo
Implementar e demonstrar controles técnicos de segurança e conformidade LGPD em aplicação real.

## 4. Metodologia
- Diagnóstico do repositório
- Refatoração orientada a requisitos 1.1–8.7
- Testes automatizados + evidências de logs
- Documentação técnico-científica

## 5. Arquitetura (figuras)
- Figura A: arquitetura lógica (`docs/architecture.md`)
- Figura B: fluxo autenticação + 2FA (`docs/architecture.md`)
- Figura C: fluxo reset de senha (`docs/architecture.md`)
- Figura D: fluxo de direitos LGPD (`docs/architecture.md`)

## 6. Mecanismos de segurança
- Argon2 parametrizável
- 2FA obrigatório
- Sessão expirada + logout invalidando sessão
- Lockout anti brute-force
- Reset seguro com token temporário
- TLS obrigatório em produção
- Criptografia de dados sensíveis em repouso
- Auditoria de eventos críticos

## 7. Conformidade LGPD
- Inventário de dados e finalidades
- Minimização
- Consentimento (versão/data/hash)
- Revogação de consentimento
- Consulta/exportação/exclusão de dados

## 8. Resultados
- Requisitos técnicos cobertos e mapeados em checklist
- Testes automatizados para fluxos críticos
- Trilhas de auditoria para autenticação e direitos do titular

## 9. Conclusão
A solução reduz superfície de ataque, melhora rastreabilidade e oferece mecanismos práticos de conformidade legal.

## 10. Instrução de exportação
Gerar versão visual do pôster em ferramenta de apresentação (PowerPoint/Google Slides) usando esta estrutura e exportar em PDF público para apresentação.

## 11. Mensagens-chave para apresentação oral (8.6/8.7)

- Justificar escolhas de segurança (Argon2, 2FA, TLS, criptografia em repouso) com base em risco.
- Explicar como os direitos LGPD foram operacionalizados em endpoints verificáveis.
- Demonstrar leitura de logs de auditoria para responder perguntas sobre rastreabilidade.
