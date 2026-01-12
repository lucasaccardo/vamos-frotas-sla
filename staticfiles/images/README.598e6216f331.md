# Imagens para Login

## Imagens Referenciadas

Este layout de login referencia as seguintes imagens:

1. **hero-truck.jpg** - `static/images/hero-truck.jpg`
   - Imagem hero para o lado esquerdo da tela de login
   - Sugestão: Imagem de caminhão ou frota da Vamos
   - Dimensões recomendadas: 1920x1080 ou superior
   - Formato: JPG
   - Fallback: Atualmente usa `static/img/background.png` caso não exista

2. **logo-vamos.svg** - `static/images/logo-vamos.svg`
   - Logo da Vamos em formato SVG para melhor qualidade
   - Dimensões: Escalável (SVG)
   - Fallback: Atualmente usa `static/img/logo.png` caso não exista

## Notas

- O CSS em `static/css/auth.css` já está configurado para usar estas imagens
- Se as imagens não existirem, o sistema usa os fallbacks mencionados acima
- Para melhor resultado visual, adicione as imagens nos caminhos especificados
