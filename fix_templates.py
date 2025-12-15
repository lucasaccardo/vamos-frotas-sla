# fix_templates.py
# Executar: python fix_templates.py
# Esse script substitui ocorrências tipo: {{ some_var or '-' }}  por  {{ some_var|default:"-" }}
# Ele faz backup dos arquivos originais com sufixo .bak antes de alterar.

import os, re, shutil

# Regex para capturar: {{ <expressão> or 'default' }} ou with double quotes
pattern = re.compile(r"\{\{\s*([^\}]+?)\s+or\s+(['\"])(.*?)\2\s*\}\}")

def replace_content(text):
    # substitui todas as ocorrências mapeando grupo1 (expressão) e grupo3 (default)
    def repl(m):
        expr = m.group(1).strip()
        default = m.group(3).replace('"', '\\"')
        return "{{ " + expr + '|default:"' + default + '" }}'
    new_text, n = pattern.subn(repl, text)
    return new_text, n

changed_files = []
for root, _, files in os.walk("templates"):
    for f in files:
        if f.endswith(".html"):
            path = os.path.join(root, f)
            try:
                s = open(path, encoding='utf-8').read()
            except Exception as e:
                print("Erro lendo", path, e)
                continue
            new_s, count = replace_content(s)
            if count > 0:
                bak = path + ".bak"
                # cria backup apenas se ainda não existir
                if not os.path.exists(bak):
                    shutil.copy2(path, bak)
                with open(path, "w", encoding='utf-8') as fh:
                    fh.write(new_s)
                changed_files.append((path, count))
                print(f"Corrigido {count} ocorrências em: {path} (backup: {bak})")

if not changed_files:
    print("Nenhum arquivo alterado — não foi encontrado o padrão.")
else:
    print("Arquivos atualizados:")
    for p, c in changed_files:
        print(f" - {p}: {c} ocorrências substituídas")