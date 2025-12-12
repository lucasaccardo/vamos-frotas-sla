import os
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.db.models import Sum
from django.urls import reverse
from .models import Sinistro, HistoricoSinistro
from .forms import SinistroForm
import json 

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

# --- API DE BUSCA (MODO ROBUSTO - IGUAL MANUTENÇÃO) ---
@login_required(login_url='login')
def api_buscar_dados_sinistro(request):
    placa = request.GET.get('placa', '').strip().upper()
    if not placa: return JsonResponse({'encontrado': False})

    try:
        # 1. Caminho da pasta de dados
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        data_dir = os.path.join(base_dir, 'vamos', 'data')
        
        # 2. Varredura para achar "Base De Clientes Total"
        arquivo_alvo = None
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if "BASE DE CLIENTES TOTAL" in f.upper() and (f.endswith('.xlsx') or f.endswith('.csv')):
                    arquivo_alvo = os.path.join(data_dir, f)
                    break
        
        # Se não achar pelo nome exato, pega qualquer Excel (Fallback)
        if not arquivo_alvo and os.path.exists(data_dir):
             for f in os.listdir(data_dir):
                 if f.endswith('.xlsx'):
                     arquivo_alvo = os.path.join(data_dir, f)
                     break

        if not arquivo_alvo:
            return JsonResponse({'encontrado': False, 'msg': 'Base de dados não encontrada.'})

        # 3. Leitura Completa (Sem filtros para garantir que pegamos tudo)
        try:
            if arquivo_alvo.endswith('.csv'):
                try: df = pd.read_csv(arquivo_alvo, sep=';', encoding='latin1', on_bad_lines='skip')
                except: df = pd.read_csv(arquivo_alvo, sep=',', encoding='utf-8', on_bad_lines='skip')
            else:
                df = pd.read_excel(arquivo_alvo)

            # Normaliza colunas (Maiúsculo e sem espaços)
            df.columns = df.columns.astype(str).str.strip().str.upper()
            
            # Localiza a coluna PLACA
            col_placa = next((c for c in df.columns if 'PLACA' in c), None)
            if not col_placa: 
                return JsonResponse({'encontrado': False, 'msg': 'Coluna PLACA não encontrada na planilha.'})

            # 4. Busca a Linha
            row = df[df[col_placa].astype(str).str.strip().str.upper() == placa]

            if not row.empty:
                data = row.iloc[0]
                
                # --- FUNÇÃO FAREJADORA DE COLUNAS ---
                def obter_dado(palavras_chave):
                    # Tenta achar uma coluna que contenha uma das palavras
                    for col in df.columns:
                        for chave in palavras_chave:
                            if chave == col: # Prioridade exata
                                val = data.get(col)
                                if pd.notna(val) and str(val).strip() != "": return str(val).strip().upper()
                    
                    # Se não achou exato, tenta parcial (ex: "MODELO" dentro de "DESCRIÇÃO MODELO")
                    for col in df.columns:
                        for chave in palavras_chave:
                            if chave in col:
                                val = data.get(col)
                                if pd.notna(val) and str(val).strip() != "": return str(val).strip().upper()
                    return ""
                # -------------------------------------

                # Mapeamento com múltiplas tentativas de nome
                cliente = obter_dado(['CLIENTE', 'NOME', 'RAZAO'])
                modelo = obter_dado(['MODELO', 'VEICULO', 'BEM', 'ATIVO', 'DESCRIÇÃO'])
                chassi = obter_dado(['CHASSI', 'VIN', 'SERIE'])
                contrato = obter_dado(['CONTRATO', 'PEDIDO'])
                
                # Inteligência de Segmento
                cc = obter_dado(['CENTRO DE CUSTO', 'CENTRO CUSTO', 'CC'])
                seg_planilha = obter_dado(['SEGMENTO'])
                
                segmento_final = 'OUTROS'
                
                # Regra 1: O que está escrito na coluna Segmento
                if 'AGRO' in seg_planilha: segmento_final = 'AGRO'
                elif 'PESADO' in seg_planilha or 'CAMINHAO' in seg_planilha: segmento_final = 'PESADOS'
                elif 'INTRA' in seg_planilha or 'EMPILHADEIRA' in seg_planilha: segmento_final = 'INTRA'
                
                # Regra 2: Centro de Custo
                elif cc:
                    if cc.startswith('G'): segmento_final = 'AGRO'
                    elif cc.startswith('H1') or 'PESADO' in modelo: segmento_final = 'PESADOS'
                    elif cc.startswith('H3') or cc.startswith('H6'): segmento_final = 'INTRA'

                # Proteção Casco (Tenta achar coluna Proteção ou Casco)
                tem_protecao = False
                prot = obter_dado(['PROTECAO', 'CASCO', 'SEGURO'])
                if 'SIM' in prot or 'S' == prot: tem_protecao = True

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
                return JsonResponse({'encontrado': False, 'msg': 'Placa não encontrada.'})

        except Exception as e:
            print(f"Erro leitura Excel: {e}")
            return JsonResponse({'encontrado': False, 'msg': 'Erro ao ler arquivo Excel.'})

    except Exception as e:
        return JsonResponse({'encontrado': False, 'msg': f"Erro interno: {str(e)}"})


# --- NOVO SINISTRO ---
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
            
            HistoricoSinistro.objects.create(
                sinistro=sinistro,
                setor_anterior='-',
                setor_novo=sinistro.setor_atual,
                alterado_por=request.user,
                comentario="Abertura do processo"
            )
            
            messages.success(request, f"Processo {sinistro.placa} incluído!")
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