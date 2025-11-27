import google.generativeai as genai

# Sua chave (que você me passou antes)
genai.configure(api_key="AIzaSyA821yX6bOVatN5bf2BNikhAhngRSlo6p4")

print("\n--- BUSCANDO MODELOS DISPONÍVEIS PARA SUA CHAVE ---")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Nome: {m.name}")
except Exception as e:
    print(f"❌ Erro ao listar: {e}")

print("\n---------------------------------------------------")