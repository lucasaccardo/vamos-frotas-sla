# Vamos Frotas SLA

## Execução local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py runserver
```

## Testes

```bash
python manage.py test vamos accounts tickets procedures analyses
```

> Observação: no estado atual do repositório, `python manage.py test` global pode falhar por conflito de descoberta no app `sinistros`.

## Documentação

- `docs/LGPD.md`
- `docs/SEGURANCA_E_ARQUITETURA.md`
- `docs/EVIDENCIAS_SEGURANCA.md`
- `docs/RESUMO_CIENTIFICO.md`
- `docs/POSTER_TEMPLATE.md`
