# Changelog

All notable changes to this project will be documented in this file.

## [2026-08-07] - Auditoria de bugs e preparação para deploy no Railway

### Fixed (bugs de runtime)
- `sinistros/views.py`: `sinistro_timeline_api` usava a variável inexistente `events`
  em vez de `eventos`, causando `NameError` (500) em toda chamada à API de timeline.
- `vamos/views.py`: 7 views staff-only (`dashboard_view`, `usuario_list_view`,
  `usuario_detail_view`, `usuario_toggle_status_view`, `usuario_delete_view`,
  `delete_request_list_view`, `admin_upload_base_view`) redirecionavam para a URL
  `"home"`, que não existe no projeto — usuário não-staff acessando essas rotas
  recebia `NoReverseMatch` (500) em vez do redirecionamento esperado. Corrigido
  para `"portal"`.
- `vamos/views.py`: o assistente de I.A. formatava negrito markdown com
  `.replace('**','<b>').replace('**','</b>')`, que substituía AMBAS as ocorrências
  de `**` pela tag de abertura (a segunda chamada não encontrava mais nada),
  gerando HTML quebrado. Trocado por regex que faz a substituição em pares.
- `templates/two_factor/_base.html` (novo arquivo): a tela de login (2FA) estava
  renderizando o template de fallback do pacote `django-two-factor-auth`, com o
  aviso "Provide a template named two_factor/_base.html..." visível para todo
  usuário. Criado o override reaproveitando `static/css/auth.css` (que já
  existia no projeto mas nunca foi conectado a nenhuma página).
- `sinistros/views.py`: `exportar_csv` filtrava `cliente__icontains` diretamente
  no banco — campo que passou a ser criptografado (ver migração abaixo) só
  suporta o lookup `isnull`. Alterado para filtrar em Python após a busca
  (mesmo padrão já usado em `exportar_xlsx`).
- `vamos_frotas_sla/settings.py`: se `SECURITY_LOG_FILE` apontasse para um
  diretório inexistente (ex.: valor sugerido `/app/logs/...` no `.env.example`
  fora de um container Docker), o `logging.config.dictConfig()` chamado por
  `django.setup()` derrubava a aplicação com `FileNotFoundError` antes mesmo do
  `AppConfig.ready()` rodar. Agora o diretório é criado no próprio `settings.py`.
- `build.sh`: removida chamada a `python manage.py load_postgres_data`, comando
  de management que não existe no projeto — quebrava o build sempre que
  `LOAD_INITIAL_DATA=True` fosse configurado.
- `templates/vamos/delete_request_status.html`, `delete_request_detail.html`,
  `reset_password_confirm.html`: `{% url %}` apontando para nomes de rota
  inexistentes (`delete_request_review`, `reset_password_confirm`).
- Removidos dois arquivos vazios com nome inválido
  (`accounts/management/inicialização.py.py` e
  `accounts/management/commands/inicialização.py.py`), resquício de erro de
  criação de arquivo.

### Security
- `vamos/views.py`: `criar_admin_secreto` (rota `/segredo-admin/`) criava um
  superusuário com senha fixa e conhecida (`MudarAgora123`) para qualquer
  visitante não autenticado, desde que nenhum superusuário existisse ainda.
  Agora exige um token (`ADMIN_SETUP_TOKEN`) via querystring.
- Migração `sinistros/migrations/0010_...`: os campos sensíveis do model
  `Sinistro`/`Frota` (`placa`, `cliente`, `chassi`, `n_contrato`, `contrato`,
  `telefone_contato`) já estavam declarados com
  `django_cryptography.fields.encrypt` no `models.py`, mas a migração que
  aplicava a criptografia no banco nunca tinha sido gerada — ou seja, os dados
  estavam sendo salvos em texto plano apesar do código/documentação (`docs/crypto.md`)
  afirmarem que eram criptografados em repouso.

### Added (deploy)
- `railway.json`: configuração de build/deploy para o Railway (Railpack) —
  `collectstatic` no build, `migrate` como pre-deploy e `gunicorn` como start
  command usando `$PORT`.
- `.python-version`: fixa a versão do Python (3.12) usada no build.
- `docs/deploy-railway.md`: passo a passo completo de deploy no Railway,
  incluindo variáveis de ambiente necessárias e o aviso de que o Railway não
  tem mais um plano gratuito ilimitado (apenas trial de 30 dias + US$1/mês).
- `vamos_frotas_sla/settings.py`: detecção automática do ambiente Railway
  (`RAILWAY_ENVIRONMENT`/`RAILWAY_PUBLIC_DOMAIN`) para `DEBUG`, `ALLOWED_HOSTS`
  e `CSRF_TRUSTED_ORIGINS`, no mesmo padrão já existente para o Render.
- `requirements.txt`: adicionado `google-generativeai`, dependência do
  assistente de I.A. que já era importada em `vamos/ai_services.py` mas nunca
  tinha sido declarada — a funcionalidade ficava sempre indisponível mesmo com
  `GOOGLE_API_KEY` configurada.

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
