import pandas as pd
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.db.models import Sum
from django.urls import reverse
from .models import Sinistro, HistoricoSinistro, Frota
from .forms import SinistroForm, UploadBaseForm

# --- HOME ---
@login_required(login_url='login')
def sinistros_home_view(request):
    request.session['modulo_ativo'] = 'sinistros'
    segmento = request.GET.get('segmento')
    if segmento:
        sinistros = Sinistro.objects.filter(segmento=segmento).order_by('-ultima_interacao')
    else:
        sinistros = Sinistro.objects.all().order_by('-ultima_interacao')
    return render(request, "sinistros/home.html", {'sinistros': sinistros})

# --- IMPORTADOR DE BASE (O Grande Segredo: Sobe Excel -> Salva no Banco) ---
@login_required(login_url='login')
def importar_frota_view(request):
    if request.method == 'POST':
        form = UploadBaseForm(request.POST, request.FILES)
        if form.is_valid():
            arquivo = request.FILES['arquivo']
            try:
                # 1. Lê o arquivo (Excel ou CSV)
                if arquivo.name.endswith('.csv'):
                    df = pd.read_csv(arquivo, sep=';', encoding='latin1', on_bad_lines='skip')
                else:
                    df = pd.read_excel(arquivo)
                
                # 2. Limpa nomes das colunas (Maiúsculo e sem espaço)
                df.columns = df.columns.astype(str).str.strip().str.upper()
                
                # Verifica se tem a coluna principal
                col_placa = next((c for c in df.columns if 'PLACA' in c), None)
                if not col_placa:
                    messages.error(request, "A planilha precisa ter a coluna PLACA.")
                    return redirect('importar_frota')

                # 3. Limpa a base antiga (Substituição total para não duplicar)
                Frota.objects.all().delete()
                
                # 4. Prepara dados para salvar
                lista_frota = []
                
                # Função auxiliar para pegar valor seguro de colunas variadas
                def get_val(row, keys):
                    for k in df.columns:
                        for key in keys:
                            if key in k:
                                val = row.get(k)
                                if pd.notna(val): return str(val).strip().upper()
                    return None

                # Itera sobre cada linha da planilha
                for _, row in df.iterrows():
                    placa = str(row[col_placa]).strip().upper()
                    if not placa or placa == 'NAN': continue
                    
                    cliente = get_val(row, ['CLIENTE', 'NOME']) or ''
                    modelo = get_val(row, ['MODELO', 'VEICULO', 'BEM']) or ''
                    chassi = get_val(row, ['CHASSI', 'VIN']) or ''
                    contrato = get_val(row, ['CONTRATO']) or ''
                    cc = get_val(row, ['CENTRO', 'CUSTO']) or ''
                    seg_excel = get_val(row, ['SEGMENTO']) or ''
                    
                    # Lógica de Segmento Inteligente
                    segmento_final = 'OUTROS'
                    # Prioridade 1: O que está escrito na planilha
                    if 'AGRO' in seg_excel: segmento_final = 'AGRO'
                    elif 'PESADO' in seg_excel or 'CAMINHAO' in seg_excel: segmento_final = 'PESADOS'
                    elif 'INTRA' in seg_excel or 'EMPILHADEIRA' in seg_excel: segmento_final = 'INTRA'
                    
                    # Prioridade 2: Centro de Custo ou Modelo (caso a coluna Segmento esteja vazia)
                    elif cc.startswith('G'): segmento_final = 'AGRO'
                    elif cc.startswith('H1') or 'PESADO' in modelo or 'CAMINHAO' in modelo: segmento_final = 'PESADOS'
                    elif cc.startswith('H3') or cc.startswith('H6') or 'EMPILHADEIRA' in modelo: segmento_final = 'INTRA'
                    
                    lista_frota.append(Frota(
                        placa=placa,
                        cliente=cliente[:199],
                        modelo=modelo[:199],
                        chassi=chassi[:99],
                        contrato=contrato[:99],
                        centro_custo=cc[:49],
                        segmento=segmento_final
                    ))
                
                # 5. Salva no Banco (Super Rápido com bulk_create)
                Frota.objects.bulk_create(lista_frota)
                
                messages.success(request, f"Base atualizada com sucesso! {len(lista_frota)} veículos importados.")
                return redirect('sinistros_home')
                
            except Exception as e:
                messages.error(request, f"Erro ao processar arquivo: {str(e)}")
    else:
        form = UploadBaseForm()
    
    return render(request, "sinistros/importar_base.html", {'form': form})

# --- NOVA API (Busca Instantânea no Banco de Dados) ---
@login_required(login_url='login')
def api_buscar_dados_sinistro(request):
    placa = request.GET.get('placa', '').strip().upper()
    if not placa: return JsonResponse({'encontrado': False})

    # A mágica acontece aqui: Busca direto na tabela Frota
    veiculo = Frota.objects.filter(placa=placa).first()
    
    if veiculo:
        return JsonResponse({
            'encontrado': True,
            'cliente': veiculo.cliente,
            'modelo': veiculo.modelo,
            'chassi': veiculo.chassi,
            'contrato': veiculo.contrato,
            'segmento': veiculo.segmento,
            'msg': 'Encontrado!'
        })
    else:
        return JsonResponse({'encontrado': False, 'msg': 'Placa não encontrada na base atualizada. Por favor, importe a planilha mais recente.'})

# --- NOVO SINISTRO ---
@login_required(login_url='login')
def novo_sinistro_view(request):
    if request.method == 'POST':
        form = SinistroForm(request.POST)
        if form.is_valid():
            sinistro = form.save(commit=False)
            sinistro.criado_por = request.user
            if sinistro.motivo == 'FURTO_ROUBO': sinistro.endereco_ativo = "N/A (Furto/Roubo)"
            sinistro.save()
            
            # Cria histórico inicial
            HistoricoSinistro.objects.create(
                sinistro=sinistro, 
                setor_anterior='-', 
                setor_novo=sinistro.setor_atual, 
                alterado_por=request.user, 
                comentario="Abertura"
            )
            
            messages.success(request, f"Processo {sinistro.placa} aberto!")
            return redirect(f"{reverse('sinistros_home')}?segmento={sinistro.segmento}")
        else:
            messages.error(request, "Erro ao salvar. Verifique os campos.")
    else:
        form = SinistroForm()
    return render(request, "sinistros/novo_sinistro.html", {'form': form})

# --- EDIÇÃO ---
@login_required(login_url='login')
def editar_sinistro_view(request, pk):
    sinistro = get_object_or_404(Sinistro, pk=pk)
    if request.method == 'POST':
        form = SinistroForm(request.POST, instance=sinistro)
        if form.is_valid():
            form.save()
            messages.success(request, "Atualizado!")
            return redirect('editar_sinistro', pk=pk)
    else:
        form = SinistroForm(instance=sinistro)
    return render(request, "sinistros/editar_sinistro.html", {"form": form, "sinistro": sinistro})

# --- DASHBOARD ---
@login_required(login_url='login')
def dashboard_sinistros_view(request):
    segmento_filtro = request.GET.get('segmento', 'TODOS')
    qs = Sinistro.objects.exclude(setor_atual='FINALIZADO')
    if segmento_filtro != 'TODOS': qs = qs.filter(segmento=segmento_filtro)
    
    labels = ['ABERTURA', 'MANUTENCAO', 'CLIENTE', 'JURIDICO', 'FINANCEIRO']
    valores = []
    now = timezone.now()
    
    # Cálculo de SLA
    for setor in labels:
        procs = qs.filter(setor_atual=setor)
        if procs.exists():
            media = sum([(now - p.ultima_interacao).days for p in procs]) / procs.count()
            valores.append(round(media, 1))
        else:
            valores.append(0)

    return render(request, "sinistros/dashboard.html", {
        "financeiro_pipeline": qs.aggregate(Sum('total_a_pagar'))['total_a_pagar__sum'] or 0,
        "total_abertos": qs.count(),
        "graf_sla_labels": json.dumps(labels),
        "graf_sla_data": json.dumps(valores),
        "segmento_atual": segmento_filtro
    })