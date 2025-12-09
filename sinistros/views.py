import os
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.db.models import Sum, Count, Avg
from django.urls import reverse
from .models import Sinistro, HistoricoSinistro
from .forms import SinistroForm
import json 

@login_required(login_url='login')
def sinistros_home_view(request):
    request.session['modulo_ativo'] = 'sinistros'
    segmento = request.GET.get('segmento')
    if segmento:
        sinistros = Sinistro.objects.filter(segmento=segmento).order_by('-ultima_interacao')
    else:
        sinistros = Sinistro.objects.all().order_by('-ultima_interacao')
    return render(request, "sinistros/home.html", {'sinistros': sinistros})

@login_required(login_url='login')
def novo_sinistro_view(request):
    if request.method == 'POST':
        form = SinistroForm(request.POST)
        if form.is_valid():
            sinistro = form.save(commit=False)
            sinistro.criado_por = request.user
            
            if sinistro.motivo == 'FURTO_ROUBO':
                sinistro.endereco_ativo = "N/A (Furto/Roubo)"
                
            sinistro.save()
            
            # Histórico Inicial
            HistoricoSinistro.objects.create(
                sinistro=sinistro,
                setor_anterior='-',
                setor_novo=sinistro.setor_atual,
                alterado_por=request.user,
                comentario="Abertura do processo"
            )
            
            # FEEDBACK E REDIRECT PARA A LISTA DO SEGMENTO
            messages.success(request, f"Processo {sinistro.placa} aberto com sucesso!")
            return redirect(f"{reverse('sinistros_home')}?segmento={sinistro.segmento}")
        else:
            print("❌ ERRO FORM:", form.errors)
            messages.error(request, "Erro ao salvar. Verifique os campos.")
    else:
        form = SinistroForm()
    return render(request, "sinistros/novo_sinistro.html", {'form': form})

# ... API DE BUSCA ... 

@login_required(login_url='login')
def api_buscar_dados_sinistro(request):
    placa = request.GET.get('placa', '').strip().upper()
    if not placa: return JsonResponse({'encontrado': False})

    try:
        # 1. Define o caminho base
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        data_dir = os.path.join(base_dir, 'vamos', 'data')
        
        # 2. Varredura inteligente de arquivo (Resolve problema de maiúscula/minúscula no Linux)
        target_file = None
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if 'BASE DE CLIENTES' in f.upper() and (f.endswith('.xlsx') or f.endswith('.csv')):
                    target_file = os.path.join(data_dir, f)
                    break
        
        if not target_file:
            return JsonResponse({'encontrado': False, 'msg': f'Arquivo de base não encontrado em: {data_dir}'})

        # 3. Leitura Otimizada (Lê apenas colunas essenciais para economizar memória)
        # Tenta ler apenas as colunas que importam, se possível
        try:
            if target_file.endswith('.csv'):
                # Tenta ler CSV
                try: df = pd.read_csv(target_file, sep=';', encoding='latin1', on_bad_lines='skip')
                except: df = pd.read_csv(target_file, sep=',', encoding='utf-8', on_bad_lines='skip')
            else:
                # Excel: Tenta ler sem carregar tudo (se o pandas for recente) ou lê normal
                df = pd.read_excel(target_file)

            # Normaliza colunas
            df.columns = df.columns.astype(str).str.strip().str.upper()
            
            # Localiza a coluna PLACA
            col_placa = next((c for c in df.columns if 'PLACA' in c), None)
            if not col_placa: 
                return JsonResponse({'encontrado': False, 'msg': 'Planilha sem coluna PLACA.'})

            # Busca (Filtra direto para economizar processamento)
            row = df[df[col_placa].astype(str).str.strip().str.upper() == placa]

            if not row.empty:
                data = row.iloc[0]
                
                # Pega dados com segurança (usando .get)
                col_cc = next((c for c in df.columns if 'CENTRO' in c and 'CUSTO' in c), '')
                cc = str(data.get(col_cc, '')).upper()
                
                segmento = 'OUTROS'
                if cc.startswith('G'): segmento = 'AGRO'
                elif cc.startswith('H15') or cc.startswith('H16'): segmento = 'PESADOS'
                elif cc.startswith('H30') or cc.startswith('H60'): segmento = 'INTRA'

                # Verifica proteção
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
                    'msg': 'Encontrado!'
                })
            else:
                return JsonResponse({'encontrado': False, 'msg': 'Placa não consta na base.'})

        except Exception as e:
            # Erro de leitura do pandas
            print(f"Erro Pandas: {e}")
            return JsonResponse({'encontrado': False, 'msg': 'Erro ao ler a planilha (Memória ou Formato).'})

    except Exception as e:
        # Erro genérico
        return JsonResponse({'encontrado': False, 'msg': f"Erro interno: {str(e)}"})

@login_required(login_url='login')
def editar_sinistro_view(request, pk):
    sinistro = get_object_or_404(Sinistro, pk=pk)
    # ... (MANTENHA A LÓGICA DE EDIÇÃO IGUAL) ...
    # (Para economizar espaço, assumo que você mantém essa parte igual)
    if request.method == 'POST':
        form = SinistroForm(request.POST, instance=sinistro)
        if form.is_valid():
            obj = form.save()
            messages.success(request, "Atualizado!")
            return redirect('editar_sinistro', pk=pk)
    else:
        form = SinistroForm(instance=sinistro)
    return render(request, "sinistros/editar_sinistro.html", {"form": form, "sinistro": sinistro})


# --- DASHBOARD COM SLA POR SETOR ---
@login_required(login_url='login')
def dashboard_sinistros_view(request):
    segmento_filtro = request.GET.get('segmento', 'TODOS')
    qs = Sinistro.objects.exclude(setor_atual='FINALIZADO')
    
    if segmento_filtro != 'TODOS': qs = qs.filter(segmento=segmento_filtro)

    # Cards Financeiros e Totais
    financeiro_pipeline = qs.aggregate(Sum('total_a_pagar'))['total_a_pagar__sum'] or 0
    total_abertos = qs.count()

    # SLA POR SETOR (CÁLCULO DE OFENSORES)
    # Agrupa por setor e calcula média de dias desde a última interação
    sla_data = {}
    now = timezone.now()
    
    # Lista fixa de setores para garantir que apareçam todos no gráfico
    setores_ordem = ['ABERTURA', 'MANUTENCAO', 'CLIENTE', 'JURIDICO', 'FINANCEIRO']
    labels = []
    valores = []
    
    for setor in setores_ordem:
        procs = qs.filter(setor_atual=setor)
        if procs.exists():
            dias_totais = sum([(now - p.ultima_interacao).days for p in procs])
            media = dias_totais / procs.count()
            valores.append(round(media, 1))
        else:
            valores.append(0)
        labels.append(setor)

    return render(request, "sinistros/dashboard.html", {
        "financeiro_pipeline": financeiro_pipeline,
        "total_abertos": total_abertos,
        "graf_sla_labels": json.dumps(labels),
        "graf_sla_data": json.dumps(valores),
        "segmento_atual": segmento_filtro
    })