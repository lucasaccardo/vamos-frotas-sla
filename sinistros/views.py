import os
import json
import logging
import traceback
import pandas as pd
from datetime import timedelta

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.db.models import Sum, Count, Q
from django.urls import reverse

from .models import Sinistro, HistoricoSinistro
from .forms import SinistroForm, EditarSinistroForm

# Logger configuration
logger = logging.getLogger(__name__)

# --- HELPER: FORMAT TIME ---
def format_timedelta(td: timedelta):
    """Returns readable string: Xd Yh Zm"""
    total_seconds = int(td.total_seconds())
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)

# --- HOME (WITH SECTOR AND SEGMENT FILTERS) ---
@login_required(login_url='login')
def sinistros_home_view(request):
    request.session['modulo_ativo'] = 'sinistros'
    
    # 1. Capture parameters
    segmento = request.GET.get('segmento')
    setor = request.GET.get('setor')

    # 2. Base QuerySet
    qs = Sinistro.objects.all()

    # 3. Filter by Segment
    if segmento:
        qs = qs.filter(segmento=segmento)

    # 4. Exclude FINALIZED by default (to clean up the view)
    qs = qs.exclude(setor_atual='FINALIZADO')

    # 5. Filter by Specific Sector
    if setor and setor != 'TODOS':
        qs = qs.filter(setor_atual=setor)

    # 6. Build options list for Sector Select
    try:
        field = Sinistro._meta.get_field('setor_atual')
        raw_choices = getattr(field, 'choices', []) or []
        setor_choices = [('TODOS', 'Todos os Setores')] + list(raw_choices)
    except Exception:
        distinct_values = list(Sinistro.objects.values_list('setor_atual', flat=True).distinct())
        setor_choices = [('TODOS', 'Todos os Setores')] + [(v, v) for v in distinct_values]

    # 7. Ordering
    qs = qs.order_by('-ultima_interacao')

    context = {
        'sinistros': qs,
        'segmento_atual': segmento or '',
        'setor_atual': setor or 'TODOS',
        'setor_choices': setor_choices,
    }
    return render(request, "sinistros/home.html", context)


# --- SEARCH API (ROBUST MODE) ---
@login_required(login_url='login')
def api_buscar_dados_sinistro(request):
    placa = request.GET.get('placa', '').strip().upper()
    if not placa: return JsonResponse({'encontrado': False})

    try:
        # 1. Path to data folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        data_dir = os.path.join(base_dir, 'vamos', 'data')
        
        # 2. Scan to find "Base De Clientes Total"
        arquivo_alvo = None
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if "BASE DE CLIENTES TOTAL" in f.upper() and (f.endswith('.xlsx') or f.endswith('.csv')):
                    arquivo_alvo = os.path.join(data_dir, f)
                    break
        
        # Fallback
        if not arquivo_alvo and os.path.exists(data_dir):
             for f in os.listdir(data_dir):
                 if f.endswith('.xlsx'):
                     arquivo_alvo = os.path.join(data_dir, f)
                     break

        if not arquivo_alvo:
            return JsonResponse({'encontrado': False, 'msg': 'Base de dados não encontrada.'})

        # 3. Full Read
        try:
            if arquivo_alvo.endswith('.csv'):
                try: df = pd.read_csv(arquivo_alvo, sep=';', encoding='latin1', on_bad_lines='skip')
                except: df = pd.read_csv(arquivo_alvo, sep=',', encoding='utf-8', on_bad_lines='skip')
            else:
                df = pd.read_excel(arquivo_alvo)

            # Normalize columns
            df.columns = df.columns.astype(str).str.strip().str.upper()
            
            # Locate PLATE column
            col_placa = None
            if 'PLACA' in df.columns: col_placa = 'PLACA'
            elif 'PLACA / CHASSI' in df.columns: col_placa = 'PLACA / CHASSI'
            else:
                for c in df.columns:
                    if 'PLACA' in c: col_placa = c; break

            if not col_placa: 
                return JsonResponse({'encontrado': False, 'msg': 'Coluna PLACA não encontrada na planilha.'})

            # 4. Search Row
            row = df[df[col_placa].astype(str).str.strip().str.upper().str.contains(placa, na=False)]

            if not row.empty:
                data = row.iloc[0]
                
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


# --- NEW SINISTRO ---
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

# --- UPDATED EDIT ---
@login_required(login_url='login')
def editar_sinistro_view(request, pk):
    sinistro = get_object_or_404(Sinistro, pk=pk)
    # Try getting history safely
    historico_qs = sinistro.historico.order_by('-data_mudanca') if hasattr(sinistro, 'historico') else []

    if request.method == 'POST':
        setor_antigo = sinistro.setor_atual
        form = EditarSinistroForm(request.POST, instance=sinistro)
        if form.is_valid():
            try:
                sinistro = form.save(commit=False)
                # Since fields are part of model, just save
                sinistro.save()

                # Create history if sector changed
                if setor_antigo != sinistro.setor_atual:
                    HistoricoSinistro.objects.create(
                        sinistro=sinistro,
                        setor_anterior=setor_antigo or '-',
                        setor_novo=sinistro.setor_atual,
                        alterado_por=request.user,
                        comentario=request.POST.get('observacoes', '') or 'Alteração via edição'
                    )

                messages.success(request, "Atualizado!")
                return redirect(f"{reverse('sinistros_home')}?segmento={sinistro.segmento}")
            except Exception as e:
                # Log and error message
                logger.exception("Erro ao salvar sinistro %s: %s", pk, e)
                traceback.print_exc()
                messages.error(request, f"Erro ao salvar: {str(e)}")
        else:
            messages.error(request, "Formulário inválido. Verifique os campos e mensagens de erro exibidas.")
    else:
        form = EditarSinistroForm(instance=sinistro)

    # Display flags (control visibility in template)
    setor_val = str(sinistro.setor_atual).upper() if sinistro.setor_atual is not None else ''
    
    # Shows approval block if MANUTENCAO
    show_aprovacao = ('MANUT' in setor_val)
    # Shows status block if MANUTENCAO (logic maintained)
    show_status = (setor_val == 'MANUTENCAO' or 'MANUT' in setor_val)

    return render(request, "sinistros/editar_sinistro.html", {
        "form": form,
        "sinistro": sinistro,
        "historico": historico_qs,
        "show_status": show_status,
        "show_aprovacao": show_aprovacao,
    })

# --- DETAILED HISTORY (ROBUST VERSION) ---
@login_required(login_url='login')
def sinistro_history_view(request, pk):
    """
    Detailed history of the claim with tolerance for missing fields and varied relation names.
    In case of error, logs traceback and shows a friendly message (does not trigger 500).
    """
    sinistro = get_object_or_404(Sinistro, pk=pk)
    now = timezone.now()
    try:
        # Try getting history queryset in different ways
        if hasattr(sinistro, 'historico'):
            eventos_qs = sinistro.historico.all().order_by('data_mudanca')
        elif hasattr(sinistro, 'historicos'):
            eventos_qs = sinistro.historicos.all().order_by('data_mudanca')
        else:
            # fallback: direct query to HistoricoSinistro assuming FK sinistro field
            eventos_qs = HistoricoSinistro.objects.filter(sinistro=sinistro).order_by('data_mudanca')

        eventos = list(eventos_qs)

        timeline = []
        per_sector = {}
        total_duration = timedelta(0)

        if not eventos:
            # No events: return empty view (no error)
            context = {
                'sinistro': sinistro,
                'timeline': [],
                'per_sector': [],
                'total_duration_human': format_timedelta(total_duration),
                'now': now,
            }
            return render(request, 'sinistros/history_detail.html', context)

        # Iterate analyzing each event
        for idx, ev in enumerate(eventos):
            # defensive: date retrieval
            start = getattr(ev, 'data_mudanca', None) or getattr(ev, 'created_at', None) or getattr(ev, 'criado_em', None)
            if not start:
                start = now

            # end is the date of the next event, or now if it's the last one
            if idx + 1 < len(eventos):
                end = getattr(eventos[idx + 1], 'data_mudanca', None) or now
            else:
                end = now

            # guarantee that start/end are compatible datetimes
            try:
                if (hasattr(end, 'tzinfo') and end.tzinfo) and (hasattr(start, 'tzinfo') and start.tzinfo):
                    duration = end - start
                else:
                    duration = end - start
            except Exception:
                duration = timedelta(0)

            total_duration += duration

            setor = getattr(ev, 'setor_novo', None) or getattr(ev, 'setor', None) or sinistro.setor_atual or '—'
            per_sector.setdefault(setor, timedelta(0))
            per_sector[setor] += duration

            # Retrieve user safely
            usuario = getattr(getattr(ev, 'alterado_por', None), 'username', None) or str(getattr(ev, 'alterado_por', '—'))

            timeline.append({
                'data_mudanca': start,
                'usuario': usuario,
                'setor_anterior': getattr(ev, 'setor_anterior', '') or '-',
                'setor_novo': getattr(ev, 'setor_novo', '') or '-',
                'comentario': getattr(ev, 'comentario', '') or '',
                'start': start,
                'end': end,
                'duration': duration,
                'duration_human': format_timedelta(duration)
            })

        per_sector_list = [
            {'setor': s, 'duration': d, 'duration_human': format_timedelta(d)}
            for s, d in per_sector.items()
        ]
        per_sector_list.sort(key=lambda x: x['duration'], reverse=True)

        context = {
            'sinistro': sinistro,
            'timeline': timeline,
            'per_sector': per_sector_list,
            'total_duration_human': format_timedelta(total_duration),
            'now': now,
        }
        return render(request, 'sinistros/history_detail.html', context)

    except Exception as exc:
        # Full log for debugging
        logger.exception("Erro ao gerar histórico detalhado do sinistro %s: %s", pk, exc)
        error_msgs = [
            "Ocorreu um erro ao carregar o histórico completo deste processo.",
            "Verifique os logs do servidor para mais detalhes."
        ]
        return render(request, 'sinistros/history_detail.html', {
            'sinistro': sinistro,
            'timeline': [],
            'per_sector': [],
            'total_duration_human': format_timedelta(timedelta(0)),
            'now': now,
            'errors': error_msgs
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

# --- DELETE SELECTED ---
@login_required(login_url='login')
def delete_selected_sinistros(request):
    if request.method != 'POST':
        messages.error(request, "Método inválido.")
        return redirect('sinistros_home')

    if not request.user.is_staff:
        messages.error(request, "Permissão negada.")
        return redirect('sinistros_home')

    ids = request.POST.getlist('selected_ids')
    if not ids:
        messages.error(request, "Nenhum processo selecionado.")
        return redirect('sinistros_home')

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

    qs.delete()
    messages.success(request, f"{count} processo(s) excluído(s).")
    return redirect('sinistros_home')