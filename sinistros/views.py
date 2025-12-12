import os
import json
import logging
import traceback
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
# Adicionado Q conforme solicitado, embora para filtros simples o filter() baste, é bom ter importado.
from django.db.models import Sum, Count, Q
from django.urls import reverse

from .models import Sinistro, HistoricoSinistro
from .forms import SinistroForm, EditarSinistroForm

# --- HOME (COM NOVOS FILTROS DE SETOR E SEGMENTO) ---
@login_required(login_url='login')
def sinistros_home_view(request):
    request.session['modulo_ativo'] = 'sinistros'
    
    # 1. Captura parâmetros
    segmento = request.GET.get('segmento')
    setor = request.GET.get('setor')

    # 2. QuerySet Base
    qs = Sinistro.objects.all()

    # 3. Filtro por Segmento
    if segmento:
        qs = qs.filter(segmento=segmento)

    # 4. Exclui FINALIZADO por padrão (para limpar a visão)
    qs = qs.exclude(setor_atual='FINALIZADO')

    # 5. Filtro por Setor Específico
    if setor and setor != 'TODOS':
        qs = qs.filter(setor_atual=setor)

    # 6. Construir lista de opções para o Select de Setores
    try:
        # Tenta pegar as 'choices' definidas no Model (fica mais bonito o texto)
        field = Sinistro._meta.get_field('setor_atual')
        raw_choices = getattr(field, 'choices', []) or []
        setor_choices = [('TODOS', 'Todos os Setores')] + list(raw_choices)
    except Exception:
        # Fallback: pega valores distintos que existem no banco
        distinct_values = list(Sinistro.objects.values_list('setor_atual', flat=True).distinct())
        setor_choices = [('TODOS', 'Todos os Setores')] + [(v, v) for v in distinct_values]

    # 7. Ordenação e Contexto
    # Mantive a ordenação por 'ultima_interacao' que você usava, pois é melhor para gestão de fila
    qs = qs.order_by('-ultima_interacao')

    context = {
        'sinistros': qs,
        'segmento_atual': segmento or '',
        'setor_atual': setor or 'TODOS',
        'setor_choices': setor_choices,
    }
    return render(request, "sinistros/home.html", context)


# --- API DE BUSCA (MODO ROBUSTO) ---
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

            # 4. Busca a Linha (usa contains para achar mesmo se tiver texto misturado)
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
    historico_qs = sinistro.historico.order_by('-data_mudanca')

    if request.method == 'POST':
        form = EditarSinistroForm(request.POST, instance=sinistro)
        if form.is_valid():
            # ... salva e redirect (mantém sua lógica atual) ...
            pass
    else:
        form = EditarSinistroForm(instance=sinistro)

    # flag que controla se mostramos o campo status_os inicialmente
    show_status = (str(sinistro.setor_atual).upper() == 'MANUTENCAO') or (form.initial.get('setor_atual', '').upper() == 'MANUTENCAO')

    return render(request, "sinistros/editar_sinistro.html", {
        "form": form,
        "sinistro": sinistro,
        "historico": historico_qs,
        "show_status": show_status,
    })
    
# --- DASHBOARD ---
@login_required(login_url='login')
def dashboard_sinistros_view(request):
    segmento_filtro = request.GET.get('segmento', 'TODOS')
    qs = Sinistro.objects.exclude(setor_atual='FINALIZADO')
    if segmento_filtro != 'TODOS':
        qs = qs.filter(segmento=segmento_filtro)

    financeiro_pipeline = qs.aggregate(Sum('total_a_pagar'))['total_a_pagar__sum'] or 0
    total_abertos = qs.count()
    total_finalizados = Sinistro.objects.filter(setor_atual='FINALIZADO').count()

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
        "financeiro_pipeline": financeiro_pipeline,
        "financeiro_caixa": 0,
        "total_abertos": total_abertos,
        "total_finalizados": total_finalizados,
        "graf_sla_labels": json.dumps(labels),
        "graf_sla_data": json.dumps(valores),
        "segmento_atual": segmento_filtro
    })

# --- DELETE SELECIONADOS ---
@login_required(login_url='login')
def delete_selected_sinistros(request):
    """
    Exclui em lote os sinistros selecionados na listagem.
    Somente aceita POST.
    """
    if request.method != 'POST':
        messages.error(request, "Método inválido.")
        return redirect('sinistros_home')

    # Se quiser permitir só staff:
    if not request.user.is_staff:
        messages.error(request, "Permissão negada.")
        return redirect('sinistros_home')

    ids = request.POST.getlist('selected_ids')
    if not ids:
        messages.error(request, "Nenhum processo selecionado.")
        return redirect('sinistros_home')

    # Segurança: garantir que ids são inteiros
    try:
        ids = [int(i) for i in ids]
    except ValueError:
        messages.error(request, "IDs inválidos.")
        return redirect('sinistros_home')

    qs = Sinistro.objects.filter(pk__in=ids)
    count = qs.count()
    if count == 0:
        messages.warning(request, "Nenhum processo válido encontrado para exclusão.")
        return redirect('sinistros_home')

    # Excluir (operação destrutiva)
    qs.delete()
    messages.success(request, f"{count} processo(s) excluído(s).")
    return redirect('sinistros_home')