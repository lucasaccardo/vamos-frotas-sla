# Guia de Testes - Nova Tela de Login

## Visão Geral

Este documento descreve como testar as mudanças no layout da tela de login após a implementação do PR.

## Arquivos Modificados/Criados

1. **templates/account/login.html** - Novo template de login com layout dividido
2. **static/css/auth.css** - Estilos CSS para autenticação com centralização vertical
3. **static/images/README.md** - Documentação sobre imagens necessárias
4. **vamos/views.py** - Atualização para usar o novo template

## Como Testar Localmente

### 1. Preparar o Ambiente

```bash
# Clone o repositório e acesse a branch
git checkout fix/login-layout

# Ative o ambiente virtual (se necessário)
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt

# Execute as migrações (se necessário)
python manage.py migrate

# Colete os arquivos estáticos
python manage.py collectstatic --noinput
```

### 2. Iniciar o Servidor de Desenvolvimento

```bash
python manage.py runserver
```

### 3. Acessar a Tela de Login

Abra o navegador e acesse: `http://localhost:8000/login/`

## Checklist de QA

### Layout Desktop (≥1200px)

- [ ] **Centralização Vertical**: O painel de autenticação está centralizado verticalmente na tela
- [ ] **Espaçamento Direito**: O painel está afastado do canto direito (margin-right: 6%)
- [ ] **Hero Visível**: A imagem hero aparece no lado esquerdo ocupando o espaço restante
- [ ] **Proporções**: O painel tem largura fixa de 380px
- [ ] **Sombras e Bordas**: O painel tem sombra e borda arredondada visíveis
- [ ] **Logo**: A logo está visível no topo do painel
- [ ] **Campos de Input**: Os campos de usuário e senha estão bem espaçados
- [ ] **Botão de Login**: O botão está centralizado e com cor vermelha (#dc2626)
- [ ] **Links**: Links de "Esqueceu a senha?" e "Criar conta" estão visíveis e alinhados

### Layout Tablet (900px - 1200px)

- [ ] **Painel reduzido**: O painel tem 360px de largura
- [ ] **Margin ajustada**: Margin-right reduzida para 4%
- [ ] **Hero redimensionado**: Hero ainda visível mas proporcionalmente menor
- [ ] **Texto do Hero**: Título e subtítulo redimensionados apropriadamente

### Layout Tablet Pequeno (768px - 900px)

- [ ] **Painel reduzido**: O painel tem 340px de largura
- [ ] **Margin ajustada**: Margin-right reduzida para 3%
- [ ] **Funcionalidade**: Todos os elementos ainda são clicáveis e funcionais

### Layout Mobile (≤768px)

- [ ] **Hero escondido**: A imagem hero não aparece em mobile
- [ ] **Painel centralizado**: O painel está centralizado horizontalmente
- [ ] **Largura responsiva**: O painel ocupa 90% da largura da tela
- [ ] **Layout vertical**: Todo conteúdo empilhado verticalmente
- [ ] **Touch targets**: Botões e links são facilmente clicáveis em touchscreen

### Layout Mobile Pequeno (≤375px)

- [ ] **Painel ajustado**: Painel ocupa 95% da largura
- [ ] **Padding reduzido**: Padding interno reduzido para 1.5rem
- [ ] **Logo menor**: Logo reduzida para 120px
- [ ] **Texto legível**: Todo texto permanece legível

## Validação de Funcionalidades

### Formulário de Login

- [ ] **Submit**: Formulário envia corretamente ao clicar em "Entrar"
- [ ] **CSRF Token**: Token CSRF está presente no formulário
- [ ] **Validação**: Campos requerem preenchimento (required)
- [ ] **Autofocus**: Campo de usuário recebe foco automaticamente
- [ ] **Mensagens de Erro**: Erros de autenticação são exibidos corretamente

### Links

- [ ] **Esqueceu a senha**: Link funciona e redireciona para a página correta
- [ ] **Criar conta**: Link funciona e redireciona para a página de signup
- [ ] **URLs Django**: Todos os {% url %} tags funcionam corretamente

### Estados Visuais

- [ ] **Hover no Botão**: Botão muda de cor ao passar o mouse (#dc2626 → #b91c1c)
- [ ] **Focus nos Inputs**: Inputs mostram borda vermelha e sombra ao receber foco
- [ ] **Hover nos Links**: Links mudam de cor ao passar o mouse

## Resoluções de Teste Recomendadas

Teste nas seguintes larguras de tela:

1. **1920px** - Desktop Full HD
2. **1366px** - Laptop padrão
3. **1024px** - Tablet landscape
4. **768px** - Tablet portrait (breakpoint mobile)
5. **375px** - Mobile padrão (iPhone SE, etc)

### Como Testar Diferentes Resoluções

**Chrome DevTools:**
1. Pressione F12 ou Ctrl+Shift+I
2. Clique no ícone de dispositivo móvel (Toggle device toolbar)
3. Selecione a resolução desejada ou digite manualmente

**Firefox DevTools:**
1. Pressione F12
2. Clique no ícone de Responsive Design Mode (Ctrl+Shift+M)
3. Ajuste a largura conforme necessário

## Ajustes Personalizados

Se necessário ajustar o espaçamento do painel:

1. Abra `static/css/auth.css`
2. Localize a linha `margin-right: 6%;` na classe `.auth-panel`
3. Ajuste o valor conforme necessário (recomendado: 3% a 10%)
4. Recarregue a página (Ctrl+Shift+R para forçar recarga)

## Imagens Estáticas

### Imagens Referenciadas

O layout faz referência às seguintes imagens:

- **Hero**: `static/images/hero-truck.jpg` (fallback: `static/img/background.png`)
- **Logo**: `static/images/logo-vamos.svg` (fallback: `static/img/logo.png`)

### Adicionar Imagens Customizadas

Para usar imagens customizadas:

1. Adicione `hero-truck.jpg` em `static/images/`
2. Adicione `logo-vamos.svg` em `static/images/`
3. Execute `python manage.py collectstatic`
4. Recarregue a página

## Problemas Conhecidos

- Se as imagens customizadas não existirem, o sistema usa as imagens de fallback
- Em navegadores muito antigos, o layout pode não funcionar perfeitamente (requer suporte a Flexbox)

## Notas Adicionais

- O layout usa Flexbox para centralização vertical
- Todos os estilos estão isolados no arquivo `auth.css`
- O template é independente e não herda de `base.html` para evitar carregar navbar e footer
- Os estilos são compatíveis com os navegadores modernos (Chrome, Firefox, Safari, Edge)
