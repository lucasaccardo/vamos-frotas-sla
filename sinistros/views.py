import os
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from .models import Sinistro, HistoricoSinistro
from .forms import SinistroForm

@login_required(login_url='login')
def sinistros_home_view(request):
    request.session['modulo_ativo'] = 'sinistros'
    sinistros = Sinistro.objects.all().order_by('-ultima_interacao')
    return render(request, "sinistros/home.html", {'sinistros': sinistros})

@login_required(login_url='login')
def novo_sinistro_view(request):
    if request.method == 'POST':
        form = SinistroForm(request.POST)
        if form.is_valid():
            sinistro = form.save(commit=False)
            sinistro.criado_por = request.user
            sinistro.save()
            
            # Registra no histórico
            HistoricoSinistro.objects.create(
                sinistro=sinistro,
                setor_anterior='-',
                setor_novo=sinistro.setor_atual,
                alterado_por=request.user,
                comentario="Abertura do processo"
            )
            
            messages.success(request, "Sinistro aberto com sucesso!")
            return redirect('sinistros_home')
    else:
        form = SinistroForm()
    
    return render(request, "sinistros/novo_sinistro.html", {'form': form})

# === API DE BUSCA INTELIGENTE ===
@login_required(login_url='login')
def api_buscar_dados_sinistro(request):
    placa = request.GET.get('placa', '').strip().upper()
    if not placa: return JsonResponse({'encontrado': False})

    try:
        # Sobe um nível para achar a pasta 'vamos/data' onde está o arquivo
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        file_path = os.path.join(base_dir, 'vamos', 'data', 'Base De Clientes Total.xlsx') # Tenta Excel primeiro
        
        # Se não achar xlsx, tenta csv
        if not os.path.exists(file_path):
            file_path = file_path.replace('.xlsx', '.csv')
        
        if not os.path.exists(file_path):
            return JsonResponse({'encontrado': False, 'msg': 'Base Total não encontrada.'})

        # Lê o arquivo
        if file_path.endswith('.csv'):
            try: df = pd.read_csv(file_path, sep=';', encoding='latin1', on_bad_lines='skip')
            except: df = pd.read_csv(file_path, sep=',', encoding='utf-8', on_bad_lines='skip')
        else:
            df = pd.read_excel(file_path)
            
        df.columns = df.columns.astype(str).str.strip().str.upper()
        
        # Procura a coluna PLACA
        col_placa = next((c for c in df.columns if 'PLACA' in c), None)
        if not col_placa: return JsonResponse({'encontrado': False, 'msg': 'Coluna PLACA não encontrada.'})

        # Busca a linha
        row = df[df[col_placa].astype(str).str.strip().str.upper() == placa]

        if not row.empty:
            data = row.iloc[0]
            
            # --- REGRA 1: SEGMENTO (Pelo Centro de Custo) ---
            col_cc = next((c for c in df.columns if 'CENTRO' in c and 'CUSTO' in c), '')
            cc = str(data.get(col_cc, '')).upper()
            segmento = 'OUTROS'
            if cc.startswith('G'): segmento = 'AGRO'
            elif cc.startswith('H15') or cc.startswith('H16'): segmento = 'PESADOS'
            elif cc.startswith('H30') or cc.startswith('H60'): segmento = 'INTRA'
            
            # --- REGRA 2: PROTEÇÃO DO CASCO ---
            col_prot = next((c for c in df.columns if 'PROTECAO' in c or 'CASCO' in c), None)
            tem_protecao = False
            if col_prot:
                val_prot = str(data[col_prot]).upper()
                if 'SIM' in val_prot or 'S' == val_prot: tem_protecao = True

            return JsonResponse({
                'encontrado': True,
                'cliente': str(data.get('CLIENTE', '') or data.get('NOME', '')),
                'chassi': str(data.get('CHASSI', '')),
                'modelo': str(data.get('MODELO', '') or data.get('MODELO DO ATIVO', '')),
                'contrato': str(data.get('CONTRATO', '')),
                'segmento': segmento,
                'tem_protecao': tem_protecao,
                'msg': 'Dados encontrados!'
            })
            
        return JsonResponse({'encontrado': False, 'msg': 'Placa não encontrada.'})

    except Exception as e:
        return JsonResponse({'encontrado': False, 'msg': str(e)})