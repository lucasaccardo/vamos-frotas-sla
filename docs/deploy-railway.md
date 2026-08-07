# Deploy no Railway

Guia passo a passo para hospedar este projeto no [Railway](https://railway.com).

## ⚠️ Sobre o "grátis" do Railway

O Railway **não tem mais um plano gratuito ilimitado**. Hoje ele oferece:

- **Trial**: US$ 5 em créditos, válidos por 30 dias, sem precisar cartão de crédito.
- **Plano gratuito permanente**: US$ 1/mês em créditos de uso (bem limitado — um
  serviço web pequeno + Postgres rodando 24/7 costuma consumir mais que isso
  por mês).

Ou seja: dá para colocar o site no ar e testar completamente de graça, mas
para deixá-lo rodando o mês inteiro sem pausar, o mais realista é esperar
usar uma fração de crédito paga (geralmente poucos dólares/mês para uma app
pequena como esta). Isso é uma característica da Railway, não um problema do
projeto.

## Arquivos já preparados neste repositório

- [`railway.json`](../railway.json): define build (`collectstatic`), o comando
  que roda as migrations antes de cada deploy (`preDeployCommand`) e o comando
  de start (`gunicorn`).
- [`.python-version`](../.python-version): fixa a versão do Python usada pelo
  builder (Railpack) em `3.12`, a mesma testada localmente.
- `requirements.txt`: já inclui `gunicorn` e `whitenoise` (arquivos estáticos)
  e `dj-database-url`/`psycopg2-binary` (Postgres).
- `vamos_frotas_sla/settings.py`: já detecta automaticamente quando está
  rodando no Railway (`RAILWAY_ENVIRONMENT`) e ajusta `DEBUG`, `ALLOWED_HOSTS`
  e `CSRF_TRUSTED_ORIGINS` usando `RAILWAY_PUBLIC_DOMAIN`.

## Passo a passo

1. **Suba o código para o GitHub** (Railway faz deploy a partir de um repositório
   Git).

2. **Crie um projeto no Railway** → "Deploy from GitHub repo" → selecione este
   repositório.

3. **Adicione um banco Postgres**: no projeto, clique em "New" → "Database" →
   "Add PostgreSQL". O Railway cria a variável `DATABASE_URL` automaticamente
   nesse serviço de banco.

4. **Conecte o Postgres ao serviço web**: no serviço da aplicação, aba
   "Variables", adicione:
   ```
   DATABASE_URL=${{Postgres.DATABASE_URL}}
   DATABASE_SSL_REQUIRE=False
   ```
   (o `${{Postgres.DATABASE_URL}}` referencia a variável do serviço de banco
   automaticamente — não precisa copiar e colar a URL manualmente). Essa é a
   URL **interna** (rede privada do projeto), por isso `DATABASE_SSL_REQUIRE`
   pode ficar `False`. Se em vez disso você usar `DATABASE_PUBLIC_URL` (acesso
   externo ao banco, fora do projeto), troque para `True`.

5. **Configure as variáveis de ambiente obrigatórias** no serviço web (aba
   "Variables"):

   | Variável | Valor sugerido |
   |---|---|
   | `DJANGO_SECRET_KEY` | gere uma chave forte, ex.: `python -c "import secrets; print(secrets.token_urlsafe(50))"` |
   | `DJANGO_DEBUG` | `False` |
   | `DJANGO_ALLOWED_HOSTS` | o domínio público do serviço, ex.: `meu-app.up.railway.app` (gere o domínio primeiro — passo 6) |
   | `DJANGO_CSRF_TRUSTED_ORIGINS` | `https://meu-app.up.railway.app` |
   | `TERMOS_VERSAO` | `2026-05` (ou a versão vigente dos termos de uso) |
   | `ADMIN_SETUP_TOKEN` | um token secreto qualquer, usado só para criar o primeiro superusuário (veja passo 8) |

   Opcionais (funcionalidades específicas):

   | Variável | Para quê |
   |---|---|
   | `EMAIL_HOST_USER` / `EMAIL_HOST_PASSWORD` / `DEFAULT_FROM_EMAIL` | envio de e-mail (reset de senha, notificação de tickets) via Gmail SMTP |
   | `GOOGLE_API_KEY` | assistente de I.A. (Gemini) |
   | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_STORAGE_BUCKET_NAME` | armazenar uploads (fotos de perfil, PDFs) em S3 em vez do disco local do container (recomendado, já que o filesystem do Railway não é persistente entre deploys) |
   | `SECURITY_LOG_FILE` | caminho customizado do log de auditoria; se omitido, usa `<projeto>/logs/security_audit.log` (criado automaticamente) |

   > Não é preciso setar `DJANGO_ALLOWED_HOSTS`/`DJANGO_CSRF_TRUSTED_ORIGINS`
   > toda vez que o domínio mudar — o `settings.py` já adiciona
   > `RAILWAY_PUBLIC_DOMAIN` automaticamente. Definir essas duas variáveis
   > explicitamente é só uma rede de segurança para o primeiro deploy (antes
   > do domínio público existir).

6. **Gere o domínio público**: aba "Settings" → "Networking" → "Generate
   Domain". Copie o domínio gerado e ajuste `DJANGO_ALLOWED_HOSTS` /
   `DJANGO_CSRF_TRUSTED_ORIGINS` acima se ainda não tiver feito.

7. **Deploy**: o Railway builda e sobe automaticamente a cada push. O
   `railway.json` já cuida de `collectstatic`, `migrate` e subir o Gunicorn.
   Acompanhe os logs de build/deploy pela aba "Deployments".

8. **Crie o primeiro superusuário**. Duas opções:
   - Pela CLI do Railway (`npm i -g @railway/cli`, depois `railway login` e
     `railway link`): `railway run python manage.py createsuperuser`.
   - Ou acessando `https://SEU-DOMINIO/segredo-admin/?token=SEU_ADMIN_SETUP_TOKEN`
     uma única vez (cria o usuário `admin` com senha `MudarAgora123` — troque a
     senha imediatamente depois do primeiro login). Essa rota só funciona
     enquanto não existir nenhum superusuário e exige o token correto.

9. **Teste o site**: acesse o domínio gerado, faça login com o superusuário
   criado e confira o portal, o módulo de Manutenção e o de Sinistros.

## Observação sobre armazenamento de arquivos

Sem um bucket S3 configurado (`AWS_ACCESS_KEY_ID` etc.), uploads (fotos de
perfil, PDFs gerados, planilhas importadas) são gravados no disco do próprio
container. O Railway **não garante que esse disco persista entre deploys**
— a cada novo deploy o filesystem é recriado do zero. Para produção de
verdade, configure as variáveis AWS acima (S3 ou compatível) para que esses
arquivos não se percam a cada deploy.
