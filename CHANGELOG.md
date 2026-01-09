# Changelog

All notable changes to this project will be documented in this file.

## [2026-01-09] - Adicionado campo aprovador_os, SLAs por setor e exportação de relatórios

### Added
- Adicionado campo `aprovador_os` ao model Sinistro para armazenar nome do aprovador da O.S.
- Implementado método `sla_por_setor()` no model Sinistro que retorna SLA máxima (dias corridos) e label descritivo para cada setor
- Adicionado cálculo automático de `retornar_ate` baseado na SLA do setor ao salvar/alterar setor
- Adicionada coluna "SLA (dias)" na listagem principal de sinistros (templates/sinistros/home.html)
- Adicionado display dinâmico da SLA ao lado do select de setor no formulário de edição
- **Implementada funcionalidade de exportação de relatórios (.xlsx e .csv) com filtros avançados**
- **Criada página de relatórios (templates/sinistros/exportar.html) acessível por usuários staff**
- **Adicionadas views `exportar_xlsx` e `exportar_csv` com proteção `@user_passes_test(is_staff)`**
- Criados testes unitários abrangentes em `sinistros/tests/test_aprovador_sla.py`
- **Criados testes de exportação em `sinistros/tests/test_export_xlsx.py`**

### Changed
- Atualizada lista SETORES no model para incluir: PRECIFICACAO, DESMOBILIZACAO_MEDICAO, DEPTO_SINISTRO
- Atualizada view `editar_sinistro_view` para recalcular `retornar_ate` quando setor ou `aguarda_aprovacao_os` mudar
- Atualizado template `editar_sinistro.html` com display dinâmico de SLA via JavaScript
- Atualizado template `home.html` para exibir SLA de cada processo na tabela
- **Adicionadas rotas de exportação em `sinistros/urls.py`**

### SLA Rules Implemented
- CLIENTE → 5 dias corridos
- PRECIFICACAO → 5 dias corridos
- FINANCEIRO → 4 dias corridos
- DESMOBILIZACAO_MEDICAO → SEM PRAZO
- DEPTO_SINISTRO → 60 dias corridos
- MANUTENCAO → 15 dias corridos (ou 2 dias se aguarda_aprovacao_os == True)
- FINALIZADO → SEM PRAZO
- ABERTURA → SEM PRAZO

### Export Features
- **Filtros disponíveis**: setor, segmento, cliente (busca parcial), intervalo de datas (com seleção do campo a filtrar), apenas processos pagos
- **Formato Excel (.xlsx)**: Formatação automática de datas (dd/mm/YYYY), valores com 2 decimais, colunas auto-ajustadas
- **Formato CSV**: Fallback para volumes grandes, compatível com todas as ferramentas
- **Colunas incluídas**: ID, Nº Chamado, Nº Contrato, Placa, Chassi, Cliente, Segmento, Modelo, Setor Atual, Responsável Setor, SLA, Prazo, Data Ocorrência, Data Início Tratativa, Última Interação, Motivo, Observações, Aguardando Aprovação, Aprovador O.S., valores financeiros, dados de auditoria
- **Proteção**: Acesso restrito a usuários staff (`is_staff=True`)
- **Performance**: Otimizado para volumes grandes com opção CSV

### Technical Details
- Migration já existente: `0004_sinistro_aguarda_aprovacao_os_sinistro_aprovador_os.py`
- Campo `aprovador_os` configurado com `max_length=255`, `blank=True`, `null=True`
- Cálculo de `retornar_ate` usa `date.today() + timedelta(days=dias)` para dias corridos
- JavaScript atualiza display de SLA sem necessidade de chamadas AJAX
- **Dependências**: pandas e openpyxl (já presentes em requirements.txt)
- **Arquivos gerados**: Nome inclui timestamp para facilitar organização
