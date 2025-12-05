import os
import json # Adicionado para tratar os dados dos gráficos
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.db.models import Sum, Count, Avg # Adicionado para cálculos matemáticos

# Importação dos modelos e forms do app de Sinistros
from .models import Sinistro, HistoricoSinistro
from .forms import SinistroForm

# --- DASHBOARD (TORRE DE CONTROLE) ---
@login_required(login_url='login')
def dashboard_sinistros_view(request):
    # 1. Filtros
    segmento_filtro = request.GET.get('segmento', 'TODOS')
    
    qs = Sinistro.objects.all()
    
    # Aplica filtro se não for "TODOS"
    if segmento_filtro != 'TODOS':
        qs = qs.filter(segmento=segmento_filtro)

    # 2. Financeiro
    # A Pagar (Pipeline) = Soma do total_a_pagar dos processos NÃO finalizados
    # Pago (Caixa) = Soma do total_pago de TODOS os processos (inclusive finalizados)
    financeiro_pipeline = qs.exclude(setor_atual='FINALIZADO').aggregate(Sum('total_a_pagar'))['total_a_pagar__sum'] or 0
    financeiro_caixa = qs.aggregate(Sum('total_pago'))['total_pago__sum'] or 0

    # 3. Volumetria
    total_abertos = qs.exclude(setor_atual='FINALIZADO').count()
    total_finalizados = qs.filter(setor_atual='FINALIZADO').count()

    # 4. Cálculo de SLA (Gargalos)
    # Vamos calcular a média de dias parado em cada setor (apenas ativos)
    processos_ativos = qs.exclude(setor_atual='FINALIZADO')
    
    sla_por_setor = {} # Ex: {'JURIDICO': [5, 10, 2], 'MANUTENCAO': [1, 2]}
    
    now = timezone.now()
    
    for p in processos_ativos:
        # Se ultima_interacao for None, usa a data de criação ou agora para não quebrar
        data_base = p.ultima_interacao or p.created_at or now
        dias_parado = (now - data_base).days
        
        nome_setor = p.get_setor_atual_display()
        
        if nome_setor not in sla_por_setor:
            sla_por_setor[nome_setor] = []
        sla_por_setor[nome_setor].append(dias_parado)
    
    # Calcula a média simples
    graf_sla_labels = []
    graf_sla_data = []
    
    for setor, lista_dias in sla_por_setor.items():
        if lista_dias:
            media = sum(lista_dias) / len(lista_dias)
            graf_sla_labels.append(setor)
            graf_sla_data.append(round(media, 1))

    # 5. Gráfico de Motivos (Pizza)
    motivos_qs = qs.values('motivo').annotate(total=Count('id'))
    
    # Tratamento para exibir nome bonito no gráfico
    graf_motivo_labels = []
    graf_motivo_data = []
    
    for m in motivos_qs:
        label = m['motivo']
        if label: # Se não estiver vazio
            label = label.replace('_', ' ').title()
        else:
            label = "Não Classificado"
            
        graf_motivo_labels.append(label)
        graf_motivo_data.append(m['total'])

    return render(request, "sinistros/dashboard.html", {
        "segmento_atual": segmento_filtro,
        "financeiro_pipeline": financeiro_pipeline,
        "financeiro_caixa": financeiro_caixa,
        "total_abertos": total_abertos,
        "total_finalizados": total_finalizados,
        
        # Dados Gráficos JSON (Para o Chart.js ler no HTML)
        "graf_sla_labels": json.dumps(graf_sla_labels),
        "graf_sla_data": json.dumps(graf_sla_data),
        "graf_motivo_labels": json.dumps(graf_motivo_labels),
        "graf_motivo_data": json.dumps(graf_motivo_data),
    })

# --- VIEWS EXISTENTES ---

@login_required(login_url='login')
def sinistros_home_view(request):
    # Garante que a sessão está marcada como sinistros
    request.session['modulo_ativo'] = 'sinistros'
    
    # Filtro Opcional por Segmento (clicou no botão da dashboard)
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
            sinistro.save()
            
            # Registra a abertura no histórico
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

# === API DE BUSCA INTELIGENTE (Lê a base Excel existente) ===
@login_required(login_url='login')
def api_buscar_dados_sinistro(request):
    placa = request.GET.get('placa', '').strip().upper()
    if not placa: return JsonResponse({'encontrado': False})

    try:
        # Sobe dois níveis para achar a pasta 'vamos/data' (sinistros/views.py -> sinistros -> raiz -> vamos)
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
        
        # Procura a coluna PLACA de forma flexível
        col_placa = next((c for c in df.columns if 'PLACA' in c), None)
        if not col_placa: return JsonResponse({'encontrado': False, 'msg': 'Coluna PLACA não encontrada na base.'})

        # Busca a linha correspondente
        row = df[df[col_placa].astype(str).str.strip().str.upper() == placa]

        if not row.empty:
            data = row.iloc[0]
            
            # --- REGRA 1: SEGMENTO (Define se é AGRO, PESADOS ou INTRA pelo Centro de Custo) ---
            col_cc = next((c for c in df.columns if 'CENTRO' in c and 'CUSTO' in c), '')
            cc = str(data.get(col_cc, '')).upper()
            segmento = 'OUTROS'
            if cc.startswith('G'): segmento = 'AGRO'
            elif cc.startswith('H15') or cc.startswith('H16'): segmento = 'PESADOS'
            elif cc.startswith('H30') or cc.startswith('H60'): segmento = 'INTRA'
            
            # --- REGRA 2: PROTEÇÃO DO CASCO (Verifica se o cliente tem seguro interno) ---
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
                'msg': 'Dados encontrados com sucesso!'
            })
            
        return JsonResponse({'encontrado': False, 'msg': 'Placa não encontrada na base.'})

    except Exception as e:
        return JsonResponse({'encontrado': False, 'msg': f'Erro ao processar arquivo: {str(e)}'})

# === SALVAR O HISTORICO AUTOMATICAMENTE SEMPRE QUE ALTERAR O SETOR ===
@login_required(login_url='login')
def editar_sinistro_view(request, pk):
    sinistro = get_object_or_404(Sinistro, pk=pk)
    
    # Guarda o estado anterior para comparar mudanças de fase
    setor_anterior = sinistro.setor_atual

    if request.method == 'POST':
        form = SinistroForm(request.POST, instance=sinistro)
        if form.is_valid():
            obj = form.save(commit=False)
            
            # Se mudou de setor, registra no histórico automaticamente
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
            messages.success(request, "Sinistro atualizado com sucesso!")
            return redirect('editar_sinistro', pk=pk)
    else:
        form = SinistroForm(instance=sinistro)

    # Pega o histórico para mostrar na timeline da tela de edição
    historico = sinistro.historico.all().order_by('-data_mudanca')

    return render(request, "sinistros/editar_sinistro.html", {
        "form": form, 
        "sinistro": sinistro,
        "historico": historico
    })