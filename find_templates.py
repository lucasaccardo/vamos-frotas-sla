# find_templates.py
# Executar: python find_templates.py
import os, re

pat = re.compile(r"\{\{\s*[^}]+?\s+or\s+['\"]")
found = []
for root, _, files in os.walk("templates"):
    for f in files:
        if f.endswith(".html"):
            p = os.path.join(root, f)
            try:
                s = open(p, encoding='utf-8').read()
            except:
                continue
            if pat.search(s):
                found.append(p)
if not found:
    print("Nenhum template com padrão '{{ ... or '...' }}' encontrado.")
else:
    print("Templates com uso inválido encontrado:")
    for p in found:
        print(p)