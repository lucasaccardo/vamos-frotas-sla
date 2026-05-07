# Documentação do projeto

- `docs/LGPD.md`: políticas e fluxo mínimo de atendimento aos direitos do titular.
- `AUDITORIA_SEGURANCA_LGPD.md`: checklist técnico-acadêmico de segurança e LGPD.

## Observações de segurança de repositório

- `db.sqlite3` não deve ser versionado (contém dados de aplicação). O arquivo foi removido do versionamento.
- `venv_stable/` está no histórico do repositório como legado e não deve ser usado em produção. O ambiente deve ser criado localmente com:
- O hasher principal de senha é Argon2 e depende de `argon2-cffi` (definido em `requirements.txt`).

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
