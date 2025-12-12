import os
import pandas as pd
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.db.models import Sum
from django.urls import reverse
from .models import Sinistro, HistoricoSinistro
from .forms import SinistroForm

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

# --- API DE BUSCA (ATUALIZADA) ---
@login_required(login_url='login')
def api_buscar_dados_sinistro(request):
    placa_input = request.GET.get('placa', '').strip().upper()
    if not placa_input: return JsonResponse({'encontrado': False})

    try:
        # 1. Caminho da pasta
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        data_dir = os.path.join(base_dir, 'vamos', 'data')
        
        # 2. Busca o arquivo (Prioridade: BASE DE CLIENTES TOTAL)
        arquivo_alvo = None
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if "BASE DE CLIENTES TOTAL" in f.upper() and (f.endswith('.xlsx') or f.endswith('.csv')):
                    arquivo_alvo = os.path.join(data_dir, f)
                    break
        
        # Fallback (pega qualquer planilha se não achar a específica)
        if not arquivo_alvo and os.path.exists(data_dir):
             for f in os.listdir(data_dir):
                 if f.endswith('.xlsx'):
                     arquivo_alvo = os.path.join(data_dir, f)
                     break

        if not arquivo_alvo:
            return JsonResponse({'encontrado': False, 'msg': 'Base de dados não encontrada.'})

        # 3. Leitura
        try:
            if arquivo_alvo.endswith('.csv'):
                try: df = pd.read_csv(arquivo_alvo, sep=';', encoding='latin1', on_bad_lines='skip')
                except: df = pd.read_csv(arquivo_alvo, sep=',', encoding='utf-8', on_bad_lines='skip')
            else:
                df = pd.read_excel(arquivo_alvo)

            # Normaliza colunas
            df.columns = df.columns.astype(str).str.strip().str.upper()
            
            # --- AGORA A MÁGICA: ACHAR A COLUNA DA PLACA ---
            col_placa = None
            
            # 1. Tenta achar coluna "PLACA" exata
            if 'PLACA' in df.columns: 
                col_placa = 'PLACA'
            # 2. Tenta achar coluna "PLACA / CHASSI"
            elif 'PLACA / CHASSI' in df.columns:
                col_placa = 'PLACA / CHASSI'
            # 3. Tenta qualquer coluna que tenha "PLACA" no nome
            else:
                for c in df.columns:
                    if 'PLACA' in c: 
                        col_placa = c
                        break
            
            if not col_placa: 
                return JsonResponse({'encontrado': False, 'msg': 'Coluna de PLACA não identificada na planilha.'})

            # Busca (usando 'contains' porque a coluna pode ter placa E chassi juntos)
            # Isso permite achar a placa mesmo se estiver misturada no texto
            row = df[df[col_placa].astype(str).str.strip().str.upper().str.contains(placa_input, na=False)]

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
                modelo = obter(['MODELO', 'VEICULO', 'BEM'])
                
                # Se a placa e chassi estão na mesma coluna, tenta buscar CHASSI separado, senão limpa da string
                chassi = obter(['CHASSI', 'VIN']) 
                if not chassi and col_placa == 'PLACA / CHASSI':
                     chassi = str(data[col_placa]).replace(placa_input, '').strip() # Tenta limpar a placa para sobrar o chassi

                contrato = obter(['CONTRATO'])
                
                # Segmento
                cc = obter(['CENTRO', 'CUSTO'])
                seg = obter(['SEGMENTO'])
                
                segmento_final = 'OUTROS'
                if 'AGRO' in seg: segmento_final = 'AGRO'
                elif 'PESADO' in seg or 'CAMINHAO' in seg: segmento_final = 'PESADOS'
                elif 'INTRA' in seg or 'EMPILHADEIRA' in seg: segmento_final = 'INTRA'
                elif cc.startswith('G'): segmento_final = 'AGRO'
                elif cc.startswith('H1'): segmento_final = 'PESADOS'
                elif cc.startswith('H3') or cc.startswith('H6'): segmento_final = 'INTRA'

                # Proteção
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
            return JsonResponse({'encontrado': False, 'msg': 'Erro na leitura do arquivo.'})

    except Exception as e:
        return JsonResponse({'encontrado': False, 'msg': str(e)})

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

# -- EDITAR SINISTRO --
@login_required(login_url='login')
def editar_sinistro_view(request, pk):
    sinistro = get_object_or_404(Sinistro, pk=pk)
    # entrega o histórico ordenado para o template
    historico_qs = sinistro.historico.order_by('-data_mudanca')

    if request.method == 'POST':
        setor_antigo = sinistro.setor_atual
        form = SinistroForm(request.POST, instance=sinistro)

        if form.is_valid():
            try:
                sinistro = form.save(commit=False)
                # Opcional: atualiza ultima_interacao ao salvar (ajuste conforme regra de negócio)
                sinistro.save()

                # Cria registro de histórico se houve mudança de setor (ou sempre, se desejar)
                if setor_antigo != sinistro.setor_atual:
                    HistoricoSinistro.objects.create(
                        sinistro=sinistro,
                        setor_anterior=setor_antigo or '-',
                        setor_novo=sinistro.setor_atual,
                        alterado_por=request.user,
                        comentario=request.POST.get('observacoes', '') or 'Alteração via edição'
                    )

                messages.success(request, "Atualizado!")
                return redirect('editar_sinistro', pk=pk)
            except Exception as e:
                import logging, traceback
                logging.exception("Erro ao salvar sinistro %s: %s", pk, e)
                traceback.print_exc()
                messages.error(request, f"Erro ao salvar: {str(e)}")
        else:
            # Loga os erros do formulário para debug e mostra mensagem amigável
            import logging
            logging.warning("Form inválido ao salvar sinistro %s: %s", pk, form.errors.as_json())
            messages.error(request, "Formulário inválido. Verifique os campos e mensagens de erro exibidas.")
    else:
        form = SinistroForm(instance=sinistro)

    return render(request, "sinistros/editar_sinistro.html", {
        "form": form,
        "sinistro": sinistro,
        "historico": historico_qs
    })