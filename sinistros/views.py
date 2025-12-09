import os
import pandas as pd 
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.db.models import Sum, Count
from django.urls import reverse
from .models import Sinistro, HistoricoSinistro
from .forms import SinistroForm
import json 

# --- HOME E LISTAGEM ---
@login_required(login_url='login')
def sinistros_home_view(request):
    request.session['modulo_ativo'] = 'sinistros'
    segmento = request.GET.get('segmento')
    
    if segmento:
        sinistros = Sinistro.objects.filter(segmento=segmento).order_by('-ultima_interacao')
    else:
        sinistros = Sinistro.objects.all().order_by('-ultima_interacao')

    return render(request, "sinistros/home.html", {'sinistros': sinistros})

# --- NOVO SINISTRO (COM AS CORREÇÕES DE REDIRECT E ROUBO) ---
@login_required(login_url='login')
def novo_sinistro_view(request):
    if request.method == 'POST':
        form = SinistroForm(request.POST)
        if form.is_valid():
            sinistro = form.save(commit=False)
            sinistro.criado_por = request.user
            
            # REGRA: Se for Roubo, força endereço vazio para não travar
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
            
            messages.success(request, f"Processo {sinistro.placa} incluído com sucesso!")
            
            # REDIRECIONA PARA A ABA DO SEGMENTO CORRETO
            return redirect(f"{reverse('sinistros_home')}?segmento={sinistro.segmento}")
        else:
            print("❌ ERRO FORM:", form.errors)
            messages.error(request, "Erro ao salvar. Verifique os campos.")
    else:
        form = SinistroForm()
    
    return render(request, "sinistros/novo_sinistro.html", {'form': form})

# --- API DE BUSCA (RESTAURADA COMO ERA ANTES) ---
@login_required(login_url='login')
def api_buscar_dados_sinistro(request):
    placa = request.GET.get('placa', '').strip().upper()
    if not placa: return JsonResponse({'encontrado': False})

    try:
        # Caminho da planilha
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        file_path = os.path.join(base_dir, 'vamos', 'data', 'Base De Clientes Total.xlsx')
        
        # Tenta achar CSV se não tiver XLSX
        if not os.path.exists(file_path): file_path = file_path.replace('.xlsx', '.csv')
        if not os.path.exists(file_path): return JsonResponse({'encontrado': False, 'msg': 'Base de dados não encontrada no servidor.'})

        # Lê o arquivo com Pandas
        if file_path.endswith('.csv'):
            try: df = pd.read_csv(file_path, sep=';', encoding='latin1', on_bad_lines='skip')
            except: df = pd.read_csv(file_path, sep=',', encoding='utf-8', on_bad_lines='skip')
        else:
            df = pd.read_excel(file_path)
            
        # Normaliza colunas
        df.columns = df.columns.astype(str).str.strip().str.upper()
        col_placa = next((c for c in df.columns if 'PLACA' in c), None)
        
        if not col_placa: return JsonResponse({'encontrado': False, 'msg': 'Coluna PLACA não encontrada na base.'})

        # Busca a placa
        row = df[df[col_placa].astype(str).str.strip().str.upper() == placa]

        if not row.empty:
            data = row.iloc[0]
            
            # Lógica de Segmento pelo Centro de Custo
            col_cc = next((c for c in df.columns if 'CENTRO' in c and 'CUSTO' in c), '')
            cc = str(data.get(col_cc, '')).upper()
            
            segmento = 'OUTROS'
            if cc.startswith('G'): segmento = 'AGRO'
            elif cc.startswith('H15') or cc.startswith('H16'): segmento = 'PESADOS'
            elif cc.startswith('H30') or cc.startswith('H60'): segmento = 'INTRA'
            
            # Lógica de Proteção
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
        return JsonResponse({'encontrado': False, 'msg': f"Erro interno: {str(e)}"})

# --- EDIÇÃO ---
@login_required(login_url='login')
def editar_sinistro_view(request, pk):
    sinistro = get_object_or_404(Sinistro, pk=pk)
    setor_anterior = sinistro.setor_atual

    if request.method == 'POST':
        form = SinistroForm(request.POST, instance=sinistro)
        if form.is_valid():
            obj = form.save(commit=False)
            if obj.setor_atual != setor_anterior:
                HistoricoSinistro.objects.create(
                    sinistro=obj,
                    setor_anterior=setor_anterior,
                    setor_novo=obj.setor_atual,
                    alterado_por=request.user,
                    comentario=f"Mudança de fase: {obj.get_setor_atual_display()}"
                )
                obj.ultima_interacao = timezone.now()
                messages.info(request, f"Processo movido para: {obj.get_setor_atual_display()}")
            obj.save()
            messages.success(request, "Processo atualizado com sucesso!")
            return redirect('editar_sinistro', pk=pk)
    else:
        form = SinistroForm(instance=sinistro)
    historico = sinistro.historico.all().order_by('-data_mudanca')
    return render(request, "sinistros/editar_sinistro.html", {"form": form, "sinistro": sinistro, "historico": historico})

# --- DASHBOARD ---
@login_required(login_url='login')
def dashboard_sinistros_view(request):
    segmento_filtro = request.GET.get('segmento', 'TODOS')
    qs = Sinistro.objects.all()
    if segmento_filtro != 'TODOS': qs = qs.filter(segmento=segmento_filtro)

    financeiro_pipeline = qs.exclude(setor_atual='FINALIZADO').aggregate(Sum('total_a_pagar'))['total_a_pagar__sum'] or 0
    financeiro_caixa = qs.aggregate(Sum('total_pago'))['total_pago__sum'] or 0
    total_abertos = qs.exclude(setor_atual='FINALIZADO').count()
    total_finalizados = qs.filter(setor_atual='FINALIZADO').count()

    # Lógica de SLA
    processos_ativos = qs.exclude(setor_atual='FINALIZADO')
    sla_por_setor = {}
    now = timezone.now()
    for p in processos_ativos:
        dias_parado = (now - p.ultima_interacao).days
        nome_setor = p.get_setor_atual_display()
        if nome_setor not in sla_por_setor: sla_por_setor[nome_setor] = []
        sla_por_setor[nome_setor].append(dias_parado)
    
    graf_sla_labels = []
    graf_sla_data = []
    for setor, lista_dias in sla_por_setor.items():
        media = sum(lista_dias) / len(lista_dias)
        graf_sla_labels.append(setor)
        graf_sla_data.append(round(media, 1))
        
    return render(request, "sinistros/dashboard.html", {
        "segmento_atual": segmento_filtro,
        "financeiro_pipeline": financeiro_pipeline,
        "financeiro_caixa": financeiro_caixa,
        "total_abertos": total_abertos,
        "total_finalizados": total_finalizados,
        "graf_sla_labels": json.dumps(graf_sla_labels),
        "graf_sla_data": json.dumps(graf_sla_data),
    })