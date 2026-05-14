# Criptografia e comunicação segura

## Em trânsito (TLS/HTTPS)

- Em produção (`DJANGO_DEBUG=False`):
  - `SECURE_SSL_REDIRECT=True`
  - `SECURE_PROXY_SSL_HEADER=('HTTP_X_FORWARDED_PROTO', 'https')`
  - HSTS habilitado (`SECURE_HSTS_*`)
- Conexões não seguras são bloqueadas por redirecionamento obrigatório para HTTPS.

## Em repouso

- Dados sensíveis no módulo de sinistros usam `django_cryptography.fields.encrypt`.
- Estratégia de storage com S3 utiliza SSE (`AES256`) quando configurado.

## Gestão de chaves

- Chaves e segredos são externos ao código-fonte (`.env`/variáveis de ambiente).
- `DJANGO_SECRET_KEY` obrigatório em produção.
- Nenhuma chave criptográfica deve ser versionada.

## Teste local de TLS

Foi adicionado exemplo de reverse proxy em `infra/nginx/local-https.conf`.

Passos resumidos:
1. Suba Django local em `http://127.0.0.1:8000`.
2. Configure NGINX com certificado local para `https://localhost`.
3. Acesse via HTTPS e valide cadeado + requests HTTPS na aba Network do navegador.
