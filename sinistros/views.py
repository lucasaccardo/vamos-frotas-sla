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
# --- SUBSTITUA APENAS A FUNÇÃO api_buscar_dados_sinistro ---

@login_required(login_url='login')
def api_buscar_dados_sinistro(request):
    placa = request.GET.get('placa', '').strip().upper()
    if not placa: return JsonResponse({'encontrado': False})

    try:
        # 1. Localiza a pasta de dados
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        data_dir = os.path.join(base_dir, 'vamos', 'data')
        
        # 2. Procura o arquivo "Base De Clientes Total" (ignorando maiúsculas/minúsculas)
        arquivo_alvo = None
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if "BASE DE CLIENTES TOTAL" in f.upper() and f.endswith('.xlsx'):
                    arquivo_alvo = os.path.join(data_dir, f)
                    break
        
        if not arquivo_alvo:
            # Tenta pegar qualquer xlsx se não achar o nome exato
            if os.path.exists(data_dir):
                xlsx_files = [x for x in os.listdir(data_dir) if x.endswith('.xlsx')]
                if xlsx_files: arquivo_alvo = os.path.join(data_dir, xlsx_files[0])

        if not arquivo_alvo:
            return JsonResponse({'encontrado': False, 'msg': 'Planilha Base não encontrada.'})

        # 3. Lê o Excel
        try:
            df = pd.read_excel(arquivo_alvo)
            
            # Normaliza colunas: Remove espaços nas pontas e deixa tudo MAIÚSCULO
            df.columns = df.columns.astype(str).str.strip().str.upper()
            
            # --- MAPEAMENTO EXATO DAS SUAS COLUNAS ---
            # Sua lista: PLACA, CLIENTE, MODELO, CHASSI, CONTRATO, CENTRO DE CUSTO, SEGMENTO
            
            if 'PLACA' not in df.columns:
                return JsonResponse({'encontrado': False, 'msg': f'Coluna PLACA não encontrada. Colunas lidas: {list(df.columns)}'})

            # Busca a linha da placa
            # Converte para string para evitar erro se a placa for número no Excel
            row = df[df['PLACA'].astype(str).str.strip().str.upper() == placa]

            if not row.empty:
                data = row.iloc[0] # Pega a primeira linha
                
                # Função segura para pegar valor e evitar "nan"
                def get_val(col_name):
                    val = data.get(col_name)
                    if pd.isna(val) or str(val).lower() == 'nan': return ""
                    return str(val).strip().upper()

                # Puxa os dados EXATOS
                cliente = get_val('CLIENTE')
                modelo = get_val('MODELO')
                chassi = get_val('CHASSI')
                contrato = get_val('CONTRATO')
                
                # Lógica Inteligente de Segmento
                # 1º Tenta pegar da coluna SEGMENTO (que você disse que tem)
                segmento_lido = get_val('SEGMENTO')
                
                # 2º Se não tiver, calcula pelo CENTRO DE CUSTO
                cc = get_val('CENTRO DE CUSTO')
                
                segmento_final = 'OUTROS'
                
                # Prioridade para o que está escrito na coluna SEGMENTO
                if 'AGRO' in segmento_lido: segmento_final = 'AGRO'
                elif 'PESADO' in segmento_lido or 'CAMINHAO' in segmento_lido: segmento_final = 'PESADOS'
                elif 'INTRA' in segmento_lido or 'EMPILHADEIRA' in segmento_lido: segmento_final = 'INTRA'
                
                # Se ainda for OUTROS, tenta pelo Centro de Custo (H1, G, etc)
                elif cc:
                    if cc.startswith('G'): segmento_final = 'AGRO'
                    elif cc.startswith('H1') or 'PESADO' in modelo: segmento_final = 'PESADOS'
                    elif cc.startswith('H3') or cc.startswith('H6') or 'EMPILHADEIRA' in modelo: segmento_final = 'INTRA'

                # Proteção Casco (Não vi na sua lista, deixo False por segurança ou tento achar)
                tem_protecao = False
                # Se tiver alguma coluna de status que indique proteção, podemos ajustar aqui depois

                return JsonResponse({
                    'encontrado': True,
                    'cliente': cliente,
                    'chassi': chassi,
                    'modelo': modelo,
                    'contrato': contrato,
                    'segmento': segmento_final,
                    'tem_protecao': tem_protecao,
                    'msg': 'Encontrado!'
                })
            else:
                return JsonResponse({'encontrado': False, 'msg': 'Placa não encontrada na Base.'})

        except Exception as e:
            print(f"Erro Excel: {e}")
            return JsonResponse({'encontrado': False, 'msg': 'Erro ao ler arquivo Excel.'})

    except Exception as e:
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