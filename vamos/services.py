import os
import io
import json
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from django.conf import settings
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from io import BytesIO

# --- Cores da Marca (Novidade) ---
VAMOS_RED = colors.HexColor("#DC2626")
VAMOS_DARK = colors.HexColor("#1e293b")
VAMOS_GRAY = colors.HexColor("#f3f4f6")

# --- 1. Funções Auxiliares ---

def resource_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'data', filename)

def formatar_moeda(valor):
    if not valor:
        return "R$ 0,00"
    if isinstance(valor, str):
        try:
            valor = float(valor.replace("R$", "").replace(".", "").replace(",", ".").strip())
        except:
            return valor
    return f"R${valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def carregar_base():
    caminho = resource_path("Base De Clientes Faturamento.xlsx")
    try:
        return pd.read_excel(caminho, engine='openpyxl')
    except Exception as e:
        print(f"Erro ao carregar base: {e}")
        return None

def moeda_para_float(valor_str):
    if isinstance(valor_str, (int, float)):
        return float(valor_str)
    if isinstance(valor_str, str):
        valor_str = valor_str.replace("R$", "").replace(".", "").replace(",", ".").strip()
        try:
            return float(valor_str)
        except:
            return 0.0
    return 0.0

# --- 2. Lógica de Cálculo ---

def calcular_sla_simples(data_entrada, data_saida, prazo_sla, valor_mensalidade, feriados):
    if isinstance(data_entrada, str):
        data_entrada = datetime.strptime(data_entrada, "%Y-%m-%d").date()
    if isinstance(data_saida, str):
        data_saida = datetime.strptime(data_saida, "%Y-%m-%d").date()
        
    dias = np.busday_count(
        np.datetime64(data_entrada), 
        np.datetime64(data_saida + timedelta(days=1))
    )
    dias -= int(feriados or 0)
    dias = max(dias, 0)
    
    if dias <= prazo_sla:
        status = "Dentro do prazo"
        desconto = 0.0
        dias_excedente = 0
    else:
        status = "Fora do prazo"
        dias_excedente = dias - prazo_sla
        desconto = (float(valor_mensalidade) / 30) * dias_excedente
        
    return dias, status, desconto, dias_excedente

def calcular_cenario_comparativo(cliente, placa, entrada, saida, feriados, servico, pecas, mensalidade):
    if isinstance(entrada, str):
        entrada = datetime.strptime(entrada, "%Y-%m-%d").date()
    if isinstance(saida, str):
        saida = datetime.strptime(saida, "%Y-%m-%d").date()

    dias = np.busday_count(
        np.datetime64(entrada), 
        np.datetime64(saida + timedelta(days=1))
    )
    dias_uteis = max(dias - int(feriados or 0), 0)
    
    sla_dict = {
        "Preventiva – 2 dias úteis": 2, 
        "Corretiva – 3 dias úteis": 3,
        "Preventiva + Corretiva – 5 dias úteis": 5, 
        "Motor – 15 dias úteis": 15
    }
    sla_dias = sla_dict.get(servico, 0)
    
    excedente = max(0, dias_uteis - sla_dias)
    desconto = (float(mensalidade) / 30) * excedente if excedente > 0 else 0.0
    total_pecas = sum(float(p.get("valor", 0) or 0) for p in (pecas or []))
    total_final = (float(mensalidade) - desconto) + total_pecas
    
    # CHAVES SIMPLIFICADAS (Para não dar erro no HTML do Django)
    return {
        "cliente": cliente, 
        "placa": placa,
        "data_entrada": entrada.strftime("%d/%m/%Y"),
        "data_saida": saida.strftime("%d/%m/%Y"),
        "servico": servico, 
        "dias_uteis": int(dias_uteis),
        "sla_dias": int(sla_dias), 
        "excedente": int(excedente),
        "mensalidade_float": float(mensalidade), # Novo campo para o PDF saber o valor
        "mensalidade": formatar_moeda(mensalidade),
        "desconto": formatar_moeda(round(desconto, 2)),
        "valor_desconto": round(desconto, 2), # Novo campo para o PDF
        "pecas_total": formatar_moeda(round(total_pecas, 2)),
        "total_pecas": round(total_pecas, 2), # Novo campo para o PDF
        "total_final": formatar_moeda(round(total_final, 2)), 
        "total_final_float": round(total_final, 2), # Novo campo para o PDF
        "detalhe_pecas": pecas or []
    }

# --- 3. Geração de PDFs (NOVA LÓGICA) ---

def gerar_pdf_moderno(dados, tipo_relatorio, protocolo):
    """Gera um PDF estilizado para SLA ou Cenários usando ReportLab avançado."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    elements = []
    styles = getSampleStyleSheet()

    # --- 1. CABEÇALHO (Marca e Protocolo) ---
    header_data = [
        [
            Paragraph(f"<b>VAMOS</b> <font color='{colors.grey}'>Frotas</font>", styles['Title']), 
            Paragraph(f"<b>PROTOCOLO:</b> {protocolo}<br/>Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal'])
        ]
    ]
    header_table = Table(header_data, colWidths=[10*cm, 6*cm])
    header_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TEXTCOLOR', (0,0), (0,0), VAMOS_RED),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 0.5*cm))
    
    # Linha divisória vermelha
    elements.append(Table([[""]], colWidths=[17*cm], rowHeights=[2], style=TableStyle([('BACKGROUND', (0,0), (-1,-1), VAMOS_RED)])))
    elements.append(Spacer(1, 1*cm))

    # --- 2. TÍTULO DO RELATÓRIO ---
    title_style = ParagraphStyle('TitleCustom', parent=styles['Heading1'], textColor=VAMOS_DARK, fontSize=16, spaceAfter=10)
    elements.append(Paragraph(tipo_relatorio.upper(), title_style))
    
    # --- 3. DADOS PRINCIPAIS (Tabela Zebrada) ---
    data_rows = []
    
    if tipo_relatorio == "RELATÓRIO DE SLA MENSAL":
        # Formata os dados do SLA
        data_rows = [
            ["OS / Chamado", dados.get('os_chamado', '-')],
            ["Cliente", dados.get('cliente', '-')],
            ["Placa", dados.get('placa', '-').upper()],
            ["Ferramenta", dados.get('ferramenta', '-')],
            ["Tipo de Serviço", dados.get('tipo_servico', '-')],
            ["Data Entrada", dados.get('data_entrada', '')],
            ["Data Saída", dados.get('data_saida', '')],
            ["Prazo SLA", f"{dados.get('prazo_sla', 0)} dias úteis"],
            ["Dias Utilizados", f"{dados.get('dias_uteis_manut', 0)} dias"],
        ]
    elif tipo_relatorio == "ANÁLISE DE CENÁRIOS":
        # Formata os dados do Cenário Vencedor
        melhor = dados.get('melhor', {})
        # O ReportLab precisa do float, então pegamos as chaves criadas em calcular_cenario_comparativo
        
        # Tentativa de converter string formatada (se vier do JSON) para float
        total_pecas_float = moeda_para_float(melhor.get('total_pecas', 0))
        valor_desconto_float = moeda_para_float(melhor.get('valor_desconto', 0))
        total_final_float = moeda_para_float(melhor.get('total_final', 0))

        data_rows = [
            ["OS / Chamado", dados.get('os_chamado', '-')],
            ["Ferramenta", dados.get('ferramenta', '-')],
            ["Cenário Escolhido", f"{melhor.get('placa', '-')} ({melhor.get('cliente', '-')})"],
            ["Serviço", melhor.get('servico', '-')],
            ["Data Entrada/Saída", f"{melhor.get('data_entrada', '-')} a {melhor.get('data_saida', '-')}"],
            ["Valor Mensalidade", formatar_moeda(melhor.get('mensalidade_float', 0))],
            ["Valor Peças", formatar_moeda(total_pecas_float)],
            ["Desconto SLA", formatar_moeda(valor_desconto_float)],
            ["Peças Detalhadas", ""], # Linha para lista de Peças
        ]
        
        # Adicionar as peças detalhadas
        pecas_rows = []
        for peca in melhor.get('detalhe_pecas', []):
            pecas_rows.append([Paragraph(peca.get('nome', '-'), styles['Normal']), formatar_moeda(peca.get('valor', 0))])
        
        if pecas_rows:
             # Célula multi-coluna para as peças
            data_rows[-1][1] = Table(pecas_rows, colWidths=[6*cm, 5*cm], style=TableStyle([
                ('FONTSIZE', (0,0), (-1,-1), 8),
                ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                ('BACKGROUND', (0,0), (-1,-1), colors.white),
            ]))

    # Estilo da Tabela de Dados
    t = Table(data_rows, colWidths=[6*cm, 11*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), VAMOS_GRAY), # Coluna esquerda cinza
        ('TEXTCOLOR', (0,0), (0,-1), VAMOS_DARK),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.white), # Linhas brancas
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 1*cm))

    # --- 4. RESULTADO FINANCEIRO (Caixa de Destaque) ---
    
    valor_final_display = "R$ 0,00"
    status_display = "CONCLUÍDO"
    
    if tipo_relatorio == "RELATÓRIO DE SLA MENSAL":
        desconto = dados.get('desconto', 0)
        valor_final_display = formatar_moeda(desconto)
        status_display = dados.get('status', 'NORMAL').upper()
        label_valor = "VALOR DO DESCONTO"
    else:
        # Usa o float do cenário
        total = total_final_float
        valor_final_display = formatar_moeda(total)
        status_display = "MELHOR CENÁRIO"
        label_valor = "CUSTO TOTAL FINAL"


    # Caixa de Resultado
    resumo_data = [
        [Paragraph("<b>STATUS DA ANÁLISE</b>", styles['Normal']), Paragraph(f"<b>{label_valor}</b>", styles['Normal'])],
        [Paragraph(f"<font size=14 color='{VAMOS_RED}'><b>{status_display}</b></font>", styles['Normal']), 
         Paragraph(f"<font size=14><b>{valor_final_display}</b></font>", styles['Normal'])]
    ]
    
    resumo_table = Table(resumo_data, colWidths=[8.5*cm, 8.5*cm])
    resumo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), VAMOS_GRAY), # Cabeçalho cinza
        ('BOX', (0,0), (-1,-1), 1, VAMOS_DARK), # Borda em volta
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ]))
    elements.append(resumo_table)
    
    # --- 5. RODAPÉ ---
    elements.append(Spacer(1, 2*cm))
    elements.append(Paragraph(f"Documento gerado eletronicamente por <b>{dados.get('gerado_por', 'Sistema')}</b>.", styles['Italic']))
    elements.append(Paragraph("Vamos Frotas - Sistema de Gestão de SLA", styles['Italic']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- 4. Funções Wrapper para o Views.py (Manutenção de Compatibilidade) ---

def gerar_pdf_sla_simples_buffer(dados):
    """Wrapper que chama a função moderna para SLA."""
    protocolo = dados.get('protocolo') or dados.get('protocolo_id', 'PREVIA')
    # Garantindo que data_entrada e data_saida estejam formatadas para exibição no PDF
    if not isinstance(dados.get('data_entrada'), str):
        dados['data_entrada'] = dados['data_entrada'].strftime('%d/%m/%Y')
        dados['data_saida'] = dados['data_saida'].strftime('%d/%m/%Y')

    return gerar_pdf_moderno(dados, "RELATÓRIO DE SLA MENSAL", protocolo)

def gerar_pdf_comparativo_buffer(lista_cenarios, melhor_cenario, metadados):
    """Wrapper que chama a função moderna para Cenários."""
    protocolo = metadados.get('protocolo', 'PREVIA')
    
    # Prepara o objeto 'dados' para a função moderna, que espera um dicionário simples.
    dados = {
        **metadados, 
        'melhor': melhor_cenario,
        'cenarios': lista_cenarios # Incluído apenas para referência, mas não usado diretamente no PDF Moderno
    }
    return gerar_pdf_moderno(dados, "ANÁLISE DE CENÁRIOS", protocolo)