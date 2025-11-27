import google.generativeai as genai
from django.conf import settings
from .models import Analise, Ticket, User
import pandas as pd
import os

# Pega a chave do settings
GENAI_API_KEY = getattr(settings, "GOOGLE_API_KEY", "")

def get_gemini_model():
    """Configura e retorna o modelo Gemini 2.0 Flash."""
    if not GENAI_API_KEY:
        print("ERRO IA: API Key não encontrada.")
        return None

    try:
        genai.configure(api_key=GENAI_API_KEY)
        
        system_instruction = (
            "Você é o Assistente Oficial do sistema 'Vamos Frotas SLA'. "
            "Você tem ACESSO TOTAL aos dados de faturamento e clientes da pasta 'data'. "
            "Use esses dados para responder perguntas específicas sobre valores, clientes, placas e totais. "
            "Se precisar somar ou analisar tendências, use os dados fornecidos abaixo. "
            "Responda sempre em português do Brasil."
        )

        # Usando o modelo capaz de ler grandes contextos
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_instruction,
            generation_config={"temperature": 0.3, "top_p": 0.95, "top_k": 40}, # Temperatura mais baixa para ser mais exato nos dados
        )
        return model
        
    except Exception as e:
        print(f"Erro crítico ao configurar Gemini: {e}")
        return None

def load_folder_data():
    """
    Lê TODOS os arquivos Excel ou CSV que estiverem na pasta 'vamos/data'.
    Retorna o conteúdo completo sem limites de linhas.
    """
    full_text_data = ""
    
    try:
        # Caminho da pasta 'data' dentro do app 'vamos'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')

        if not os.path.exists(data_dir):
            return " (Pasta 'data' não encontrada)"

        # Lista todos os arquivos da pasta
        arquivos = os.listdir(data_dir)
        arquivos_lidos = 0

        for arquivo in arquivos:
            file_path = os.path.join(data_dir, arquivo)
            
            # Se for Excel
            if arquivo.endswith('.xlsx') or arquivo.endswith('.xls'):
                try:
                    df = pd.read_excel(file_path)
                    # Converte TODO o dataframe para CSV string (mais compacto que to_string)
                    csv_text = df.to_csv(index=False)
                    full_text_data += f"\n\n--- ARQUIVO: {arquivo} ---\n{csv_text}"
                    arquivos_lidos += 1
                except Exception as e:
                    print(f"Erro ao ler {arquivo}: {e}")

            # Se for CSV
            elif arquivo.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    csv_text = df.to_csv(index=False)
                    full_text_data += f"\n\n--- ARQUIVO: {arquivo} ---\n{csv_text}"
                    arquivos_lidos += 1
                except Exception as e:
                    print(f"Erro ao ler {arquivo}: {e}")

        if arquivos_lidos == 0:
            return " (Nenhum arquivo de dados compatível encontrado na pasta data)"
            
        return full_text_data

    except Exception as e:
        return f" (Erro ao acessar pasta de dados: {str(e)})"

def get_ia_context_summary():
    """
    Cria o prompt com TODA a informação disponível.
    """
    lines = ["=== RAIO-X DO SISTEMA VAMOS FROTAS ==="]

    # 1. Dados do Banco (Resumidos)
    try:
        total_users = User.objects.count()
        tickets_abertos = Ticket.objects.filter(status='Pendente').count()
        lines.append(f"- Usuários cadastrados: {total_users}")
        lines.append(f"- Tickets Pendentes: {tickets_abertos}")
    except:
        pass

    # 2. DADOS COMPLETOS DA PASTA DATA
    # Aqui entra a mágica: ele vai injetar todo o conteúdo dos Excels
    dados_pasta = load_folder_data()
    lines.append(f"\n=== BASE DE CONHECIMENTO COMPLETA (PASTA DATA) ==={dados_pasta}")

    # 3. Regras
    lines.append("\n=== REGRAS ===")
    lines.append("- Preventiva: 2 dias úteis")
    lines.append("- Corretiva: 3 dias úteis")
    lines.append("- Motor: 15 dias úteis")
    
    return "\n".join(lines)