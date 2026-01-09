import os
import json
import logging
import traceback
import pandas as pd
import csv
import io
from datetime import timedelta

from django.core.cache import cache
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.views.decorators.http import require_POST
from django.http import JsonResponse, HttpResponse
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

# Helper: format timedelta in 'Xd Xh' for API usage
def format_timedelta_days_hours(td: timedelta):
    total_seconds = int(td.total_seconds())
    days, rem = divmod(total_seconds, 86400)
    hours, _ = divmod(rem, 3600)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or not parts:
        parts.append(f"{hours}h")
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
        aguarda_aprovacao_antigo = sinistro.aguarda_aprovacao_os
        form = EditarSinistroForm(request.POST, instance=sinistro)
        if form.is_valid():
            try:
                sinistro = form.save(commit=False)
                
                # Auto-calculate retornar_ate based on SLA if sector changed or aguarda_aprovacao_os changed
                setor_mudou = setor_antigo != sinistro.setor_atual
                aguarda_mudou = aguarda_aprovacao_antigo != sinistro.aguarda_aprovacao_os
                retornar_ate_vazio = not sinistro.retornar_ate
                
                if setor_mudou or aguarda_mudou or retornar_ate_vazio:
                    dias, label = sinistro.sla_por_setor()
                    if dias is not None:
                        # Calculate return date as today + dias corridos
                        from datetime import date, timedelta
                        sinistro.retornar_ate = date.today() + timedelta(days=dias)
                    else:
                        # No SLA deadline
                        sinistro.retornar_ate = None
                
                sinistro.save()

                # Create history if sector changed
                if setor_mudou:
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
    
    # Get current SLA info for display
    dias_sla, label_sla = sinistro.sla_por_setor()

    return render(request, "sinistros/editar_sinistro.html", {
        "form": form,
        "sinistro": sinistro,
        "historico": historico_qs,
        "show_status": show_status,
        "show_aprovacao": show_aprovacao,
        "sla_label": label_sla,
        "sla_dias": dias_sla,
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

# --- DASHBOARD STATS API ---
@login_required(login_url='login')
def dashboard_stats_api(request):
    """
    Returns aggregated metrics for the dashboard.
    Supports GET filters: segmento, period_start (YYYY-MM-DD), period_end (YYYY-MM-DD).
    Result is cached for 30s to reduce load.
    """
    segmento = request.GET.get('segmento')
    period_start = request.GET.get('period_start')
    period_end = request.GET.get('period_end')
    cache_key = f"dashboard_stats:{segmento}:{period_start}:{period_end}"
    cached = cache.get(cache_key)
    if cached:
        return JsonResponse(cached)

    now = timezone.now()
    base_qs = Sinistro.objects.all()
    if segmento:
        base_qs = base_qs.filter(segmento=segmento)

    if period_start:
        try:
            from django.utils.dateparse import parse_date
            ds = parse_date(period_start)
            if ds:
                base_qs = base_qs.filter(criado_em__date__gte=ds)
        except Exception:
            pass
    if period_end:
        try:
            from django.utils.dateparse import parse_date
            de = parse_date(period_end)
            if de:
                base_qs = base_qs.filter(criado_em__date__lte=de)
        except Exception:
            pass

    totals = base_qs.aggregate(total_pago=Sum('total_pago'), total_a_pagar=Sum('total_a_pagar'))
    total_pago = float(totals['total_pago'] or 0)
    total_a_pagar_agg = totals.get('total_a_pagar') or 0.0
    if total_a_pagar_agg and total_a_pagar_agg != 0:
        total_a_pagar = float(total_a_pagar_agg)
    else:
        total_a_pagar = 0.0
        for s in base_qs:
            soma = 0
            try:
                soma += float(s.valor_fipe or 0)
            except Exception:
                soma += 0
            try:
                soma += float(s.valor_implemento or 0)
            except Exception:
                soma += 0
            total_a_pagar += soma

    total_pendente = max(0.0, total_a_pagar - total_pago)

    counts_qs = base_qs.values('setor_atual').annotate(count=Count('id')).order_by('-count')
    counts_by_sector = []
    for item in counts_qs:
        code = item['setor_atual'] or '—'
        label = dict(Sinistro.SETORES).get(code, code) if hasattr(Sinistro, 'SETORES') else code
        counts_by_sector.append({'setor_code': code, 'setor_label': label, 'count': item['count']})

    finalizados_total = base_qs.filter(setor_atual='FINALIZADO').count()

    finalizados_hist = HistoricoSinistro.objects.filter(setor_novo='FINALIZADO', sinistro__in=base_qs)
    finalizados_by_sector_map = {}
    for h in finalizados_hist:
        setor = h.setor_anterior or '—'
        finalizados_by_sector_map[setor] = finalizados_by_sector_map.get(setor, 0) + 1
    finalizados_by_sector = [{'setor': k, 'count': v} for k, v in finalizados_by_sector_map.items()]

    offenders_avg = []
    setores_list = getattr(Sinistro, 'SETORES', [])
    for code, label in setores_list:
        procs = base_qs.filter(setor_atual=code).exclude(setor_atual='FINALIZADO')
        if procs.exists():
            total_days = 0.0
            for p in procs:
                diff = now - (p.ultima_interacao or p.criado_em or now)
                total_days += diff.total_seconds() / 86400.0
            avg = total_days / procs.count()
            offenders_avg.append({'setor_code': code, 'setor_label': label, 'avg_days': round(avg, 2), 'count': procs.count()})
    offenders_avg.sort(key=lambda x: x['avg_days'], reverse=True)

    per_sector_seconds = {}
    historicos = HistoricoSinistro.objects.filter(sinistro__in=base_qs).order_by('sinistro_id', 'data_mudanca')
    current_sid = None
    events = []
    for h in historicos:
        sid = h.sinistro_id
        if current_sid is None:
            current_sid = sid
            events = [h]
        elif sid == current_sid:
            events.append(h)
        else:
            for idx, ev in enumerate(events):
                start = ev.data_mudanca
                end = events[idx+1].data_mudanca if idx+1 < len(events) else now
                setor = ev.setor_novo or ev.setor_anterior or '—'
                per_sector_seconds[setor] = per_sector_seconds.get(setor, 0) + max(0, (end - start).total_seconds())
            current_sid = sid
            events = [h]
    for idx, ev in enumerate(events):
        start = ev.data_mudanca
        end = events[idx+1].data_mudanca if idx+1 < len(events) else now
        setor = ev.setor_novo or ev.setor_anterior or '—'
        per_sector_seconds[setor] = per_sector_seconds.get(setor, 0) + max(0, (end - start).total_seconds())

    offenders_sum = []
    for setor, secs in per_sector_seconds.items():
        days = secs / 86400.0
        offenders_sum.append({'setor': setor, 'sum_days': round(days, 2)})
    offenders_sum.sort(key=lambda x: x['sum_days'], reverse=True)

    non_final_qs = base_qs.exclude(setor_atual='FINALIZADO')
    if non_final_qs.exists():
        total_days = 0.0
        for p in non_final_qs:
            diff = now - (p.ultima_interacao or p.criado_em or now)
            total_days += diff.total_seconds() / 86400.0
        avg_sla_days = total_days / non_final_qs.count()
    else:
        avg_sla_days = 0.0

    avg_seconds = int(avg_sla_days * 86400)
    avg_days = avg_seconds // 86400
    avg_hours = (avg_seconds % 86400) // 3600
    avg_sla_display = f"{avg_days}d {avg_hours}h"

    payload = {
        'total_pago': round(total_pago, 2),
        'total_a_pagar': round(total_a_pagar, 2),
        'total_pendente': round(total_pendente, 2),
        'counts_by_sector': counts_by_sector,
        'finalizados_total': finalizados_total,
        'finalizados_by_sector': finalizados_by_sector,
        'offenders_avg': offenders_avg[:8],
        'offenders_sum': offenders_sum[:8],
        'avg_sla_days': round(avg_sla_days, 2),
        'avg_sla_display': avg_sla_display,
        'timestamp': now.isoformat(),
    }

    cache.set(cache_key, payload, 30)
    return JsonResponse(payload)


@login_required(login_url='login')
@user_passes_test(lambda u: u.is_staff)
def dashboard_export_csv(request):
    segmento = request.GET.get('segmento')
    base_qs = Sinistro.objects.all()
    if segmento:
        base_qs = base_qs.filter(segmento=segmento)

    now = timezone.now()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    header = ['id', 'placa', 'cliente', 'setor_atual', 'responsavel_setor', 'retornar_ate', 'valor_fipe', 'valor_implemento', 'total_a_pagar', 'total_pago', 'ultima_interacao', 'timeline_durations_json']
    writer.writerow(header)

    for s in base_qs.order_by('id'):
        events = list(s.historico.order_by('data_mudanca').all())
        timeline = []
        for idx, ev in enumerate(events):
            start = ev.data_mudanca
            end = events[idx+1].data_mudanca if idx+1 < len(events) else now
            dur_secs = max(0, int((end - start).total_seconds()))
            timeline.append({'setor': ev.setor_novo or ev.setor_anterior, 'start': start.isoformat(), 'end': end.isoformat(), 'seconds': dur_secs})
        import json
        timeline_json = json.dumps(timeline, ensure_ascii=False)
        writer.writerow([
            s.id,
            s.placa,
            s.cliente,
            s.setor_atual,
            s.responsavel_setor or '',
            s.retornar_ate.isoformat() if s.retornar_ate else '',
            str(s.valor_fipe or ''),
            str(s.valor_implemento or ''),
            str(s.total_a_pagar or ''),
            str(s.total_pago or ''),
            s.ultima_interacao.isoformat() if s.ultima_interacao else '',
            timeline_json
        ])

    resp = HttpResponse(buffer.getvalue(), content_type='text/csv')
    resp['Content-Disposition'] = 'attachment; filename="dashboard_export.csv"'
    return resp


@login_required(login_url='login')
def sinistro_timeline_api(request, pk):
    sinistro = get_object_or_404(Sinistro, pk=pk)
    now = timezone.now()
    eventos = list(sinistro.historico.order_by('data_mudanca').all())
    timeline = []
    for idx, ev in enumerate(eventos):
        start = ev.data_mudanca
        end = events[idx+1].data_mudanca if idx+1 < len(eventos) else now
        dur = max(0, int((end - start).total_seconds()))
        timeline.append({
            'data_mudanca': start.isoformat(),
            'setor_anterior': ev.setor_anterior,
            'setor_novo': ev.setor_novo,
            'usuario': getattr(ev.alterado_por, 'username', None) or '',
            'comentario': ev.comentario or '',
            'start': start.isoformat(),
            'end': end.isoformat(),
            'duration_seconds': dur,
            'duration_human': format_timedelta_days_hours(timedelta(seconds=dur))
        })
    return JsonResponse({'sinistro_id': sinistro.id, 'timeline': timeline})

# --- PING SESSION (FOR KEEP ALIVE) ---
@login_required
@require_POST
def ping_session(request):
    """
    Minimal view to update user session activity timestamp.
    Used by frontend idle timeout script.
    """
    request.session['last_activity'] = timezone.now().isoformat()
    return JsonResponse({'ok': True})
