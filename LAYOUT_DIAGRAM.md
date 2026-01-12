<!-- Layout Visual do Login -->

# Layout da Tela de Login - Diagrama Visual

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              TELA COMPLETA (100vw x 100vh)                      │
│  Background: #0f172a                                                           │
│                                                                                 │
│  ┌─────────────────────────────────────┬───────────────────────────────────┐  │
│  │                                     │                                   │  │
│  │      HERO SECTION (flex: 1)        │    AUTH PANEL (380px)            │  │
│  │                                     │                                   │  │
│  │  ┌──────────────────────────────┐  │    ┌──────────────────────┐      │  │
│  │  │                              │  │    │                      │      │  │
│  │  │  Background Image:           │  │    │  [LOGO]             │      │  │
│  │  │  hero-truck.jpg              │  │    │                      │  ↑  │  │
│  │  │  (fallback: background.png)  │  │    │  Bem-vindo de volta │  │  │  │
│  │  │                              │  │    │  Faça login para... │  │  │  │
│  │  │  Overlay: gradient           │  │    │                      │  │  │  │
│  │  │  (escuro → transparente)     │  │    │  ┌────────────────┐ │  │  │  │
│  │  │                              │  │    │  │ Usuário:       │ │  │  │  │
│  │  │                              │  │    │  │ [___________] │ │  │  │  │
│  │  │                              │  │    │  └────────────────┘ │  │  │  │
│  │  │                         ↓    │  │    │                      │  │  │  │
│  │  │  ┌─────────────────────┐     │  │    │  ┌────────────────┐ │  │  │  │
│  │  │  │ Gestão Inteligente │     │  │    │  │ Senha:         │ │  │  │  │
│  │  │  │ de Frotas          │     │  │    │  │ [___________] │ │  │  │  │
│  │  │  │                    │     │  │    │  └────────────────┘ │  │  │  │
│  │  │  │ Controle de SLA... │     │  │    │                      │ Centro │
│  │  │  └─────────────────────┘     │  │    │  ┌────────────────┐ │  │  │  │
│  │  │  (Texto sobre overlay)       │  │    │  │   [ ENTRAR ]  │ │  │  │  │
│  │  │                              │  │    │  └────────────────┘ │  │  │  │
│  │  │                              │  │    │                      │  │  │  │
│  │  └──────────────────────────────┘  │    │  Esqueceu? | Criar │  │  │  │
│  │                                     │    │                      │  ↓  │  │
│  │                                     │    └──────────────────────┘      │  │
│  │                                     │                                   │  │
│  │                                     │    ←─── margin-right: 6% ────→   │  │
│  └─────────────────────────────────────┴───────────────────────────────────┘  │
│                                                                                 │
│  ↑─────────────────── align-items: center (vertical) ─────────────────────↑   │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Breakpoints e Mudanças

### Desktop Grande (> 1200px)
```
┌───────────────────────────────────────────────┐
│  HERO (flex: 1)    │  PANEL (380px + 6%)     │
│  Imagem visível    │  Bem espaçado           │
└───────────────────────────────────────────────┘
```

### Desktop Médio (900px - 1200px)
```
┌─────────────────────────────────────────┐
│  HERO (flex: 1)  │  PANEL (360px + 4%) │
│  Imagem visível  │  Pouco reduzido     │
└─────────────────────────────────────────┘
```

### Tablet (768px - 900px)
```
┌──────────────────────────────────┐
│  HERO (flex: 1) │ PANEL (340px + 3%) │
│  Imagem visível │ Mais compacto   │
└──────────────────────────────────┘
```

### Mobile (< 768px)
```
┌──────────────────┐
│                  │
│   ╔════════════╗ │
│   ║   PANEL   ║ │ ← 90% largura
│   ║  (centro)  ║ │ ← Hero escondido
│   ║            ║ │
│   ╚════════════╝ │
│                  │
└──────────────────┘
```

## Anatomia do Painel de Autenticação

```
┌─────────────────────────────────┐
│ padding: 2.5rem                 │
│                                 │
│  ╔═══════════╗                 │
│  ║   LOGO   ║                  │ ← width: 140px
│  ╚═══════════╝                 │
│                                 │
│  Bem-vindo de volta            │ ← font-size: 1.5rem
│  Faça login para acessar       │ ← color: #94a3b8
│                                 │
│  ┌─────────────────────────┐   │
│  │ Label: Usuário          │   │
│  ├─────────────────────────┤   │ ← height: 45px
│  │ [input text]            │   │ ← background: #0f172a
│  └─────────────────────────┘   │
│                                 │
│  ┌─────────────────────────┐   │
│  │ Label: Senha            │   │
│  ├─────────────────────────┤   │
│  │ [input password]        │   │
│  └─────────────────────────┘   │
│                                 │
│  ┌─────────────────────────┐   │
│  │      [ ENTRAR ]        │   │ ← background: #dc2626
│  └─────────────────────────┘   │ ← hover: #b91c1c
│                                 │
│  Esqueceu?    |    Criar conta │
│                                 │
└─────────────────────────────────┘
     ↑                         ↑
  border-radius: 12px    box-shadow: rgba(0,0,0,0.5)
```

## Cores Utilizadas

```css
Backgrounds:
  • Body/Page: #0f172a (Cinza azulado muito escuro)
  • Auth Panel: #1e293b (Cinza azulado escuro)
  • Inputs: #0f172a (Mais escuro)

Borders:
  • Panel: #334155 (Cinza médio)
  • Inputs: #475569 (Cinza claro)
  • Input Focus: #dc2626 (Vermelho)

Text:
  • Títulos: #ffffff (Branco)
  • Subtítulos: #94a3b8 (Cinza azul claro)
  • Labels: #cbd5e1 (Cinza bem claro)
  • Links: #94a3b8 → #dc2626 (hover)

Buttons:
  • Primary: #dc2626 (Vermelho Vamos)
  • Hover: #b91c1c (Vermelho mais escuro)

Overlay Hero:
  • Gradient: rgba(15, 23, 42, 0.85) → rgba(15, 23, 42, 0.3)
```

## Especificações Técnicas

### Dimensões do Painel (Auth Panel)

| Breakpoint | Largura | Margin-Right | Padding |
|------------|---------|--------------|---------|
| > 1200px   | 380px   | 6%          | 2.5rem  |
| ≤ 1200px   | 360px   | 4%          | 2rem    |
| ≤ 900px    | 340px   | 3%          | 1.75rem |
| ≤ 768px    | 90%     | auto        | 2.5rem  |
| ≤ 375px    | 95%     | auto        | 1.5rem  |

### Elementos do Formulário

| Elemento | Altura | Border Radius | Font Size |
|----------|--------|---------------|-----------|
| Input    | 45px   | 8px          | 0.95rem   |
| Button   | 45px   | 8px          | 1rem      |
| Logo     | auto   | -            | width: 140px (120px mobile) |

### Espaçamentos

| Tipo | Valor |
|------|-------|
| Margin entre logo e título | 2rem |
| Margin entre título e subtitle | 0.5rem |
| Margin entre subtitle e form | 2rem |
| Margin entre inputs | 1.25rem |
| Margin entre button e links | 1.5rem |

## Fluxo de Centralização Vertical

```
HTML/Body height: 100%
       ↓
.login-page min-height: 100vh + display: flex
       ↓
align-items: center (centraliza filhos verticalmente)
       ↓
.auth-panel align-self: center (garante centralização própria)
       ↓
RESULTADO: Painel sempre centralizado verticalmente
```

## Comportamento Responsivo

```
Desktop (1920px):
┌─────────────────────────────────────────────────────┐
│ [========HERO========] [PANEL────────]   ← 6%  │
└─────────────────────────────────────────────────────┘

Laptop (1366px):
┌──────────────────────────────────────────┐
│ [======HERO======] [PANEL────]  ← 4% │
└──────────────────────────────────────────┘

Tablet (1024px):
┌─────────────────────────────────┐
│ [====HERO====] [PANEL──] ← 3%  │
└─────────────────────────────────┘

Mobile (375px):
┌──────────┐
│          │
│ [PANEL]  │ ← 95% width, centralizado
│          │
│  (hero   │
│ oculto)  │
└──────────┘
```

## Efeitos e Transições

### Botão ENTRAR
```
Estado Normal:
  background: #dc2626
  transform: none

Estado Hover:
  background: #b91c1c
  transform: translateY(-1px)
  box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4)

Estado Active:
  transform: translateY(0)
```

### Input Fields
```
Estado Normal:
  border: 1px solid #475569
  background: #0f172a

Estado Focus:
  border: 1px solid #dc2626
  background: #1e293b
  box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.15)
```

### Links
```
Estado Normal:
  color: #94a3b8

Estado Hover:
  color: #dc2626
  transition: color 0.2s ease
```
