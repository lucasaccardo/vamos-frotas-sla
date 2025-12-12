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

        # 3. Leitura Completa
        try:
            if arquivo_alvo.endswith('.csv'):
                try: df = pd.read_csv(arquivo_alvo, sep=';', encoding='latin1', on_bad_lines='skip')
                except: df = pd.read_csv(arquivo_alvo, sep=',', encoding='utf-8', on_bad_lines='skip')
            else:
                df = pd.read_excel(arquivo_alvo)

            # Normaliza colunas
            df.columns = df.columns.astype(str).str.strip().str.upper()
            
            # Localiza a coluna PLACA
            col_placa = None
            if 'PLACA' in df.columns: col_placa = 'PLACA'
            elif 'PLACA / CHASSI' in df.columns: col_placa = 'PLACA / CHASSI'
            else:
                for c in df.columns:
                    if 'PLACA' in c: col_placa = c; break

            if not col_placa: 
                return JsonResponse({'encontrado': False, 'msg': 'Coluna PLACA não encontrada na planilha.'})

            # 4. Busca a Linha
            # Usa contains para achar mesmo se tiver texto misturado
            row = df[df[col_placa].astype(str).str.strip().str.upper().str.contains(placa, na=False)]

            if not row.empty:
                data = row.iloc[0]
                
                # --- FUNÇÃO FAREJADORA ---
                def obter(chaves):
                    for col in df.columns:
                        for k in chaves:
                            if k in col: 
                                val = data.get(col)
                                if pd.notna(val) and str(val).strip() != "": return str(val).strip().upper()
                    return ""
                
                cliente = obter(['CLIENTE', 'NOME'])
                modelo = obter(['MODELO', 'VEICULO'])
                chassi = obter(['CHASSI', 'VIN'])
                contrato = obter(['CONTRATO'])
                cc = obter(['CENTRO', 'CUSTO'])
                seg = obter(['SEGMENTO'])
                
                segmento_final = 'OUTROS'
                if 'AGRO' in seg: segmento_final = 'AGRO'
                elif 'PESADO' in seg or 'CAMINHAO' in seg: segmento_final = 'PESADOS'
                elif 'INTRA' in seg or 'EMPILHADEIRA' in seg: segmento_final = 'INTRA'
                elif cc.startswith('G'): segmento_final = 'AGRO'
                elif cc.startswith('H1'): segmento_final = 'PESADOS'
                elif cc.startswith('H3') or cc.startswith('H6'): segmento_final = 'INTRA'

                tem_protecao = False
                prot = obter(['PROTECAO', 'CASCO'])
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
            return JsonResponse({'encontrado': False, 'msg': f'Erro leitura Excel: {str(e)}'})

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
    setor_anterior = sinistro.setor_atual
    
    if request.method == 'POST':
        form = SinistroForm(request.POST, instance=sinistro)
        if form.is_valid():
            obj = form.save(commit=False)
            # Verifica mudança de setor
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
            messages.success(request, "Atualizado!")
            return redirect('editar_sinistro', pk=pk)
    else:
        form = SinistroForm(instance=sinistro)
        
    historico = sinistro.historico.all().order_by('-data_mudanca')
    return render(request, "sinistros/editar_sinistro.html", {"form": form, "sinistro": sinistro, "historico": historico})

# --- DASHBOARD (A FUNÇÃO QUE FALTAVA) ---
@login_required(login_url='login')
def dashboard_sinistros_view(request):
    segmento_filtro = request.GET.get('segmento', 'TODOS')
    
    # Base QuerySet
    qs_all = Sinistro.objects.all()
    if segmento_filtro != 'TODOS': 
        qs_all = qs_all.filter(segmento=segmento_filtro)
    
    # Métricas Financeiras
    financeiro_pipeline = qs_all.exclude(setor_atual='FINALIZADO').aggregate(Sum('total_a_pagar'))['total_a_pagar__sum'] or 0
    financeiro_caixa = qs_all.aggregate(Sum('total_pago'))['total_pago__sum'] or 0
    
    # Volumetria
    total_abertos = qs_all.exclude(setor_atual='FINALIZADO').count()
    total_finalizados = qs_all.filter(setor_atual='FINALIZADO').count()
    
    # SLA Calc
    labels = ['ABERTURA', 'MANUTENCAO', 'CLIENTE', 'JURIDICO', 'FINANCEIRO']
    valores = []
    now = timezone.now()
    
    # Usamos qs_all excluindo finalizados para calcular média de dias parado
    processos_ativos = qs_all.exclude(setor_atual='FINALIZADO')
    
    for setor in labels:
        procs = processos_ativos.filter(setor_atual=setor)
        if procs.exists():
            # Calcula média de dias desde a última interação
            media = sum([(now - p.ultima_interacao).days for p in procs]) / procs.count()
            valores.append(round(media, 1))
        else:
            valores.append(0)

    # Gráfico de Motivos
    motivos_qs = qs_all.values('motivo').annotate(total=Count('id'))
    graf_motivo_labels = [m['motivo'].replace('_', ' ') for m in motivos_qs]
    graf_motivo_data = [m['total'] for m in motivos_qs]

    return render(request, "sinistros/dashboard.html", {
        "financeiro_pipeline": financeiro_pipeline,
        "financeiro_caixa": financeiro_caixa,
        "total_abertos": total_abertos,
        "total_finalizados": total_finalizados,
        "graf_sla_labels": json.dumps(labels),
        "graf_sla_data": json.dumps(valores),
        "graf_motivo_labels": json.dumps(graf_motivo_labels),
        "graf_motivo_data": json.dumps(graf_motivo_data),
        "segmento_atual": segmento_filtro
    })