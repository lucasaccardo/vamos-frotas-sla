import re # Necessário para validação de senha com Regex
import markdown # Necessário para formatar resposta da IA
import json 
import os
from datetime import datetime # --- NOVO IMPORT ---
from django.db.models import Count # --- NOVO IMPORT ---

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.files.base import ContentFile 
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger 
from django.http import JsonResponse
from django.conf import settings
import pandas as pd

# Importações dos Modelos e Formulários
from .models import Ticket, Analise, DeleteRequest

# --- ATENÇÃO AQUI: Importando o SignUpForm correto ---
from .forms import SlaMensalForm, CenarioForm, PecaForm, TicketForm, SignUpForm, AdminUserForm 

# Importações da Lógica de Negócios (Cálculos e PDF)
from .services import (
    calcular_sla_simples, 
    gerar_pdf_moderno,
    calcular_cenario_comparativo,
    moeda_para_float
)

# Importações da Lógica de I.A. (Gemini)
try:
    from .ai_services import get_gemini_model, get_ia_context_summary
except ImportError:
    # Caso o ai_services não exista (ambiente local sem IA)
    get_gemini_model = None
    get_ia_context_summary = lambda: "IA indisponível"


# =============================================================================
# 0. FUNÇÕES AUXILIARES (Validação de Senha)
# =============================================================================

def validate_password_policy(password: str, username: str = "", email: str = ""):
    """
    Verifica se a senha atende aos requisitos de segurança corporativa.
    """
    errors = []
    MIN_LEN = 10
    SPECIAL_CHARS = r"!@#$%^&*()_+\-=\[\]{};':\",.<>/?\\|`~"

    if len(password) < MIN_LEN:
        errors.append(f"A senha deve ter pelo menos {MIN_LEN} caracteres.")
    if not re.search(r"[A-Z]", password):
        errors.append("A senha deve conter pelo menos 1 letra maiúscula.")
    if not re.search(r"[a-z]", password):
        errors.append("A senha deve conter pelo menos 1 letra minúscula.")
    if not re.search(r"[0-9]", password):
        errors.append("A senha deve conter pelo menos 1 número.")
    if not re.search(rf"[{re.escape(SPECIAL_CHARS)}]", password):
        errors.append("A senha deve conter pelo menos 1 caractere especial.")
     
    uname = (username or "").strip().lower()
    local_email = (email or "").split("@")[0].strip().lower()
     
    if uname and uname in password.lower():
        errors.append("A senha não pode conter o seu nome de usuário.")
    if local_email and local_email in password.lower():
        errors.append("A senha não pode conter partes do seu e-mail.")
         
    return (len(errors) == 0), errors


# =============================================================================
# 1. AUTENTICAÇÃO
# =============================================================================

def login_view(request):
    if request.user.is_authenticated:
        return redirect("home")
         
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            messages.error(request, "Usuário ou senha inválidos.")
    return render(request, "vamos/login.html")

def signup_view(request):
    if request.method == "POST":
        # Usa o novo formulário (SignUpForm) que lida com Matrícula e Nome Completo
        form = SignUpForm(request.POST)
        if form.is_valid():
            # O form.save() já cria o User e o Perfil (Matrícula) graças ao forms.py
            user = form.save(commit=False)
             
            # Define como inativo para aguardar aprovação do Admin
            user.is_active = False 
            user.save()
             
            messages.success(request, "Cadastro enviado! Aguarde aprovação do administrador.")
            return redirect("login")
        else:
            messages.error(request, "Erro ao criar conta. Verifique os dados informados.")
    else:
        form = SignUpForm()
         
    return render(request, "vamos/signup.html", {"form": form})

def logout_view(request):
    logout(request)
    return redirect("login")

# --- Reset de Senha ---
def reset_password_view(request): return redirect("login")
def reset_password_confirm_view(request, uidb64, token): return redirect("login")


# =============================================================================
# 2. PÁGINAS PRINCIPAIS (DASHBOARD ATUALIZADO)
# =============================================================================

@login_required(login_url='login')
def home_view(request):
    return render(request, "vamos/home.html")

@login_required(login_url='login')
def dashboard_view(request):
    # Segurança
    if not request.user.is_staff:
        messages.error(request, "Acesso restrito a administradores.")
        return redirect("home")

    # 1. CAPTURA FILTROS DA URL
    ano_atual = datetime.now().year
    # Pega o ano/mês da URL (se não tiver, usa 'Todos' ou o atual)
    filtro_ano = request.GET.get('ano')
    filtro_mes = request.GET.get('mes')

    # 2. PREPARA QUERYSET BASE
    qs = Analise.objects.all().order_by('data_criacao')
     
    # Descobre anos disponíveis para o filtro
    anos_disponiveis = sorted(list(set(qs.dates('data_criacao', 'year'))), key=lambda x: x.year, reverse=True)
    anos_int = [d.year for d in anos_disponiveis]
    if not anos_int: anos_int = [ano_atual]

    # Aplica Filtros se selecionado
    if filtro_ano and filtro_ano != "Todos":
        qs = qs.filter(data_criacao__year=filtro_ano)
    if filtro_mes and filtro_mes != "Todos":
        qs = qs.filter(data_criacao__month=filtro_mes)

    # 3. CÁLCULOS ESTATÍSTICOS (Processamento Python)
    total_economia = 0.0
    total_cenarios = 0
    total_sla = 0
     
    # Para o Gráfico de Linha (Evolução Mensal)
    timeline_data = {} # Ex: {'01/2025': 1500.00, '02/2025': 3000.00}

    for a in qs:
        mes_chave = a.data_criacao.strftime("%m/%Y")
        if mes_chave not in timeline_data: timeline_data[mes_chave] = 0.0
         
        economia_item = 0.0
         
        # CASO 1: SLA MENSAL (Economia = Desconto aplicado)
        if a.tipo == 'sla_mensal':
            total_sla += 1
            try:
                economia_item = float(a.dados.get('desconto', 0))
            except: pass
             
        # CASO 2: CENÁRIOS (Economia = Maior Preço - Menor Preço)
        elif a.tipo == 'cenarios':
            total_cenarios += 1
            try:
                # Pega a lista de cenários dentro do JSON
                lista = a.dados.get('cenarios', [])
                valores = []
                for c in lista:
                    # Limpa string "R$ 1.200,50" para float 1200.50
                    val_str = str(c.get('total_final', '0') or c.get('Total Final (R$)', '0'))
                    val_clean = val_str.replace('R$', '').replace(' ', '').replace('.', '').replace(',', '.').strip()
                    if val_clean: valores.append(float(val_clean))
                 
                if len(valores) > 1:
                    # A economia é a diferença entre o mais caro e o que foi escolhido (o mais barato)
                    economia_item = max(valores) - min(valores)
            except: pass

        # Soma totais
        total_economia += economia_item
        timeline_data[mes_chave] += economia_item

    # 4. DADOS PARA OS GRÁFICOS (JSON)
    # Gráfico 1: Evolução da Economia (Linha)
    graf_tempo_labels = list(timeline_data.keys())
    graf_tempo_data = list(timeline_data.values())

    # Gráfico 2: Distribuição de Tickets (Donut)
    ticket_counts = Ticket.objects.values('status').annotate(total=Count('id'))
    graf_ticket_labels = [t['status'] for t in ticket_counts]
    graf_ticket_data = [t['total'] for t in ticket_counts]
     
    # Cores para os status
    ticket_colors = []
    color_map = {'Pendente': '#ffc107', 'Em andamento': '#0dcaf0', 'Concluído': '#198754', 'Cancelado': '#6c757d'}
    for label in graf_ticket_labels: ticket_colors.append(color_map.get(label, '#333'))

    # Gráfico 3: Top Usuários (Barras) - Quem mais gera economia/análises
    top_users = Analise.objects.values('usuario__username').annotate(total=Count('id')).order_by('-total')[:5]
    graf_user_labels = [u['usuario__username'] for u in top_users]
    graf_user_data = [u['total'] for u in top_users]

    return render(request, "vamos/dashboard.html", {
        # Filtros e Totais
        "anos_disponiveis": anos_int,
        "filtro_ano": int(filtro_ano) if filtro_ano and filtro_ano != "Todos" else "Todos",
        "filtro_mes": int(filtro_mes) if filtro_mes and filtro_mes != "Todos" else "Todos",
         
        "total_economia": total_economia,
        "total_sla": total_sla,
        "total_cenarios": total_cenarios,
        "total_analises": total_sla + total_cenarios,
        "tickets_pendentes": Ticket.objects.filter(status='Pendente').count(),

        # Dados JSON para Chart.js
        "graf_tempo_labels": json.dumps(graf_tempo_labels),
        "graf_tempo_data": json.dumps(graf_tempo_data),
        "graf_ticket_labels": json.dumps(graf_ticket_labels),
        "graf_ticket_data": json.dumps(graf_ticket_data),
        "graf_ticket_colors": json.dumps(ticket_colors),
        "graf_user_labels": json.dumps(graf_user_labels),
        "graf_user_data": json.dumps(graf_user_data),
    })


# =============================================================================
# 3. CÁLCULO DE SLA MENSAL (ATUALIZADO PARA PROTOCOLO E NOVO PDF)
# =============================================================================

@login_required(login_url='login')
def sla_mensal_view(request):
    resultado = None
     
    if request.method == "POST":
        form = SlaMensalForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
             
            prazo_map = {
                "Preventiva – 2 dias úteis": 2, "Corretiva – 3 dias úteis": 3,
                "Preventiva + Corretiva – 5 dias úteis": 5, "Motor – 15 dias úteis": 15
            }
            prazo = prazo_map.get(data['tipo_servico'], 0)
             
            dias, status, desconto, excedente = calcular_sla_simples(
                data['data_entrada'], data['data_saida'], prazo, 
                float(data['mensalidade']), data['feriados']
            )
             
            # 1. Cria a Análise PRIMEIRO para gerar o Protocolo
            nova_analise = Analise(
                usuario=request.user, tipo="sla_mensal",
                placa=data['placa'], cliente=data['cliente']
            )
            nova_analise.save() 
             
            # 2. Adiciona todos os dados calculados
            dados_final = {
                "protocolo": nova_analise.protocolo, # Adiciona o protocolo
                "os_chamado": data['os_chamado'], "ferramenta": data['ferramenta'],
                "cliente": data['cliente'], "placa": data['placa'],
                "data_entrada": data['data_entrada'].strftime('%d/%m/%Y'), # Formata a data para o PDF
                "data_saida": data['data_saida'].strftime('%d/%m/%Y'),       # Formata a data para o PDF
                "tipo_servico": data['tipo_servico'], "mensalidade": float(data['mensalidade']),
                "dias_uteis_manut": int(dias), "status": status,
                "desconto": float(desconto), "dias_excedente": int(excedente),
                "prazo_sla": prazo,
                "gerado_por": request.user.get_full_name() or request.user.username
            }
             
            # 3. Gera PDF Moderno
            pdf_buffer = gerar_pdf_moderno(dados_final, "RELATÓRIO DE SLA MENSAL", nova_analise.protocolo)
             
            # 4. Salva PDF e Dados Finais na Análise
            filename = f"SLA_{nova_analise.protocolo}.pdf"
            nova_analise.dados = dados_final
            nova_analise.arquivo_pdf.save(filename, ContentFile(pdf_buffer.getvalue()))
            nova_analise.save()
             
            resultado = nova_analise
            messages.success(request, "Cálculo realizado e salvo com sucesso!")
             
    else:
        form = SlaMensalForm()
         
    return render(request, "vamos/sla_mensal.html", {"form": form, "resultado": resultado})


# =============================================================================
# 4. CÁLCULO DE CENÁRIOS (ATUALIZADO PARA PROTOCOLO E NOVO PDF)
# =============================================================================

@login_required(login_url='login')
def cenarios_view(request):
    if 'lista_cenarios' not in request.session: request.session['lista_cenarios'] = []
    if 'pecas_atuais' not in request.session: request.session['pecas_atuais'] = []
    if 'meta_dados' not in request.session: request.session['meta_dados'] = {}

    form_cenario = CenarioForm(request.POST or None)
    form_peca = PecaForm(request.POST or None)
    resultado_final = None

    if request.method == 'POST':
        acao = request.POST.get('acao')

        if acao == 'add_peca':
            if form_peca.is_valid():
                # ... (Lógica de adicionar peça igual) ...
                nova_peca = {
                    "nome": form_peca.cleaned_data['nome_peca'],
                    "valor": float(form_peca.cleaned_data['valor_peca'])
                }
                lista = request.session['pecas_atuais']
                lista.append(nova_peca)
                request.session['pecas_atuais'] = lista
                messages.success(request, "Peça adicionada!")
                request.session.modified = True 
                return redirect('cenarios')

        elif acao == 'limpar_pecas':
            request.session['pecas_atuais'] = []
            request.session.modified = True
            return redirect('cenarios')

        elif acao == 'add_cenario':
            if form_cenario.is_valid():
                data = form_cenario.cleaned_data
                if not request.session['meta_dados']:
                    request.session['meta_dados'] = {
                        'os_chamado': data['os_chamado'],
                        'ferramenta': data['ferramenta']
                    }
                cenario_calc = calcular_cenario_comparativo(
                    cliente=data['cliente_nome'] or "Não informado",
                    placa=data['placa'],
                    entrada=data['data_entrada'], saida=data['data_saida'],
                    feriados=data['feriados'], servico=data['tipo_servico'],
                    pecas=request.session['pecas_atuais'],
                    mensalidade=float(data['valor_mensalidade'] or 0)
                )
                lista_c = request.session['lista_cenarios']
                lista_c.append(cenario_calc)
                request.session['lista_cenarios'] = lista_c
                request.session['pecas_atuais'] = []
                request.session.modified = True
                messages.success(request, f"Cenário {len(lista_c)} adicionado!")
                return redirect('cenarios')

        elif acao == 'finalizar':
            lista_final = request.session.get('lista_cenarios', [])
            meta = request.session.get('meta_dados', {})
            if not lista_final:
                messages.error(request, "Adicione pelo menos um cenário.")
            else:
                try:
                    melhor = min(lista_final, key=lambda x: moeda_para_float(x['total_final']))
                except:
                    melhor = lista_final[0]

                # 1. Cria a Análise PRIMEIRO para gerar o Protocolo
                nova_analise = Analise(
                    usuario=request.user, tipo="cenarios",
                    placa=melhor.get('placa'), cliente=melhor.get('cliente')
                )
                nova_analise.save()

                # 2. Adiciona o protocolo nos metadados
                meta['protocolo'] = nova_analise.protocolo
                meta['gerado_por'] = request.user.get_full_name() or request.user.username
                 
                # 3. Prepara dados para o PDF
                dados_pdf = {
                    "cenarios": lista_final, 
                    "melhor": melhor,
                    **meta # Adiciona os metadados (protocolo, os_chamado, etc.)
                }
                 
                # 4. Gera PDF Moderno
                pdf_buffer = gerar_pdf_moderno(dados_pdf, "ANÁLISE DE CENÁRIOS", nova_analise.protocolo)
                 
                # 5. Salva PDF e Dados Finais na Análise
                filename = f"COMPARATIVO_{nova_analise.protocolo}.pdf"
                nova_analise.dados = dados_pdf
                nova_analise.arquivo_pdf.save(filename, ContentFile(pdf_buffer.getvalue()))
                nova_analise.save()

                request.session['lista_cenarios'] = []
                request.session['pecas_atuais'] = []
                request.session['meta_dados'] = {}
                request.session.modified = True
                resultado_final = nova_analise
                messages.success(request, "Análise finalizada!")
         
        elif acao == 'resetar':
            request.session['lista_cenarios'] = []
            request.session['pecas_atuais'] = []
            request.session['meta_dados'] = {}
            request.session.modified = True
            return redirect('cenarios')

    return render(request, "vamos/cenarios.html", {
        "form_cenario": form_cenario, "form_peca": form_peca,
        "pecas_atuais": request.session.get('pecas_atuais', []),
        "lista_cenarios": request.session.get('lista_cenarios', []),
        "resultado": resultado_final
    })


# =============================================================================
# 5. TICKETS (ATUALIZADO PARA RESPOSTA ADMIN E SEPARAÇÃO DE LISTAS)
# =============================================================================

@login_required(login_url='login')
def ticket_list_view(request):
    # CRIAÇÃO (Lógica do POST mantém igual)
    if request.method == "POST":
        form = TicketForm(request.POST, request.FILES)
        if form.is_valid():
            data = form.cleaned_data
            Ticket.objects.create(
                titulo=data['assunto'], 
                descricao=data['descricao'],
                usuario=request.user,
                status='Pendente'
            )
            messages.success(request, "Ticket aberto com sucesso!")
            return redirect("ticket_list")
    else:
        form = TicketForm()

    # LISTAGEM (Agora separada)
    # 1. Pega o QuerySet base (Admin vê tudo, User vê só dele)
    if request.user.is_staff:
        qs = Ticket.objects.all().order_by("-created_at")
    else:
        qs = Ticket.objects.filter(usuario=request.user).order_by("-created_at")
     
    # 2. Separa em duas listas
    tickets_abertos = qs.exclude(status__in=['Concluído', 'Cancelado'])
    tickets_finalizados = qs.filter(status__in=['Concluído', 'Cancelado'])
         
    return render(request, "vamos/tickets.html", {
        "tickets_abertos": tickets_abertos, 
        "tickets_finalizados": tickets_finalizados, 
        "form": form
    })

@login_required(login_url='login')
def ticket_detail_view(request, pk):
    # Permite Admin ou Dono do ticket
    ticket = get_object_or_404(Ticket, pk=pk)
    if not request.user.is_staff and ticket.usuario != request.user:
        messages.error(request, "Acesso negado.")
        return redirect('ticket_list')

    # Lógica de Resposta do Admin (POST)
    if request.method == "POST" and request.user.is_staff:
        resposta = request.POST.get('resposta_admin')
        novo_status = request.POST.get('status')
         
        if resposta:
            ticket.resposta_admin = resposta
            ticket.data_resposta = timezone.now()
         
        if novo_status:
            ticket.status = novo_status
             
        ticket.save()
        messages.success(request, "Ticket atualizado com sucesso!")
        return redirect('ticket_detail', pk=pk)

    return render(request, "vamos/ticket_detail.html", {"ticket": ticket})

@login_required(login_url='login')
def ticket_update_status_view(request, pk):
    # ESTA FUNÇÃO FOI SUBSTITUÍDA PELA LÓGICA MAIS COMPLETA EM TICKET_DETAIL_VIEW.
    # Apenas redireciona para garantir que a rota antiga não quebre.
    messages.warning(request, "Use a tela de detalhe para gerenciar o ticket.")
    return redirect("ticket_detail", pk=pk)


# =============================================================================
# 6. GERENCIAMENTO DE USUÁRIOS (ADMIN ONLY)
# =============================================================================

@login_required(login_url='login')
def usuario_list_view(request):
    if not request.user.is_staff:
        messages.error(request, "Acesso não autorizado.")
        return redirect("home")
     
    if request.method == "POST":
        form = AdminUserForm(request.POST)
        if form.is_valid():
            d = form.cleaned_data
            if User.objects.filter(username=d['username']).exists():
                messages.error(request, "Usuário já existe.")
            else:
                u = User.objects.create_user(d['username'], d['email'], d['password'] or "123456")
                u.first_name = d['first_name']
                # Salva matrícula no perfil (se model Perfil existir) ou usa last_name como fallback
                if hasattr(u, 'perfil'):
                    u.perfil.matricula = d['matricula']
                    u.perfil.save()
                else:
                    u.last_name = d['matricula']

                u.is_staff = (d['role'] == 'admin')
                u.is_active = d['is_active']
                u.save()
                messages.success(request, "Criado!")
                return redirect("lista_usuarios")
    else:
        form = AdminUserForm()
         
    usuarios = User.objects.all().order_by('username')
    return render(request, "vamos/usuarios.html", {"usuarios": usuarios, "form": form})

@login_required(login_url='login')
def usuario_detail_view(request, pk):
    if not request.user.is_staff:
        messages.error(request, "Acesso não autorizado.")
        return redirect("home")
         
    u = get_object_or_404(User, pk=pk)
     
    if request.method == "POST":
        form = AdminUserForm(request.POST)
        if form.is_valid():
            d = form.cleaned_data
            u.username = d['username']
            u.email = d['email']
            u.first_name = d['first_name']
             
            if hasattr(u, 'perfil'):
                u.perfil.matricula = d['matricula']
                u.perfil.save()
            else:
                u.last_name = d['matricula']

            u.is_staff = (d['role'] == 'admin')
            u.is_active = d['is_active']
            if d['password']:
                u.set_password(d['password'])
            u.save()
            messages.success(request, "Atualizado!")
            return redirect("lista_usuarios")
    else:
        # Tenta pegar matricula do Perfil ou do Last Name
        matr = ""
        if hasattr(u, 'perfil'):
            matr = u.perfil.matricula
        else:
            matr = u.last_name

        initial = {
            'username': u.username,
            'email': u.email,
            'first_name': u.first_name,
            'matricula': matr,
            'role': 'admin' if u.is_staff else 'user',
            'is_active': u.is_active
        }
        form = AdminUserForm(initial=initial)

    return render(request, "vamos/usuario_detail.html", {"form": form, "usuario_alvo": u})

@login_required(login_url='login')
def usuario_toggle_status_view(request, pk):
    if not request.user.is_staff: 
        messages.error(request, "Acesso não autorizado.")
        return redirect("home")
    u = get_object_or_404(User, pk=pk)
    u.is_active = not u.is_active
    u.save()
    messages.success(request, "Status alterado.")
    return redirect("lista_usuarios")

@login_required(login_url='login')
def usuario_delete_view(request, pk):
    if not request.user.is_staff: 
        messages.error(request, "Acesso não autorizado.")
        return redirect("home")
    u = get_object_or_404(User, pk=pk)
    if u != request.user:
        u.delete()
        messages.success(request, "Excluído.")
    else:
        messages.error(request, "Você não pode excluir a si mesmo.")
    return redirect("lista_usuarios")


# =============================================================================
# 7. HISTÓRICO DE ANÁLISES
# =============================================================================

@login_required(login_url='login')
def analise_list_view(request):
    # Pega tudo do usuário
    qs = Analise.objects.filter(usuario=request.user).order_by("-data_criacao")
     
    # Separa em duas listas para as abas
    analises_sla = qs.filter(tipo='sla_mensal')
    analises_cenarios = qs.filter(tipo='cenarios')
     
    return render(request, "vamos/analises.html", {
        "analises_sla": analises_sla,
        "analises_cenarios": analises_cenarios
    })

@login_required(login_url='login')
def analise_delete_permanent_view(request, pk):
    """Deleta o registro e o arquivo PDF definitivamente."""
    analise = get_object_or_404(Analise, pk=pk, usuario=request.user)
     
    # Tenta deletar o arquivo físico (opcional, mas boa prática)
    if analise.arquivo_pdf:
        try:
            if os.path.exists(analise.arquivo_pdf.path):
                os.remove(analise.arquivo_pdf.path)
        except:
            pass # Se não achar o arquivo, segue o jogo

    # Deleta do banco
    analise.delete()
    messages.success(request, "Relatório excluído permanentemente.")
    return redirect("lista_analises")

@login_required(login_url='login')
def analise_detail_view(request, pk):
    if request.user.is_staff:
        analise = get_object_or_404(Analise, pk=pk)
    else:
        analise = get_object_or_404(Analise, pk=pk, usuario=request.user)
    return render(request, "vamos/analise_detail.html", {"analise": analise})

@login_required(login_url='login')
def analise_exportar_view(request, pk):
    return redirect("lista_analises")

@login_required(login_url='login')
def analise_solicitar_exclusao_view(request, pk):
    a = get_object_or_404(Analise, pk=pk, usuario=request.user)
    DeleteRequest.objects.get_or_create(analise=a, solicitante=request.user, defaults={'motivo': "Solicitado pelo usuário."})
    messages.success(request, "Solicitação de exclusão enviada para aprovação!")
    return redirect("lista_analises")


# =============================================================================
# 8. GERENCIAMENTO DE EXCLUSÕES
# =============================================================================

@login_required(login_url='login')
def delete_request_list_view(request):
    if not request.user.is_staff: 
        messages.error(request, "Acesso não autorizado.")
        return redirect("home")
    delete_requests = DeleteRequest.objects.filter(status='Pendente').order_by('data_solicitacao')
    return render(request, "vamos/delete_requests.html", {"delete_requests": delete_requests})

@login_required(login_url='login')
def delete_request_detail_view(request, pk): 
    return redirect("delete_request_list")

@login_required(login_url='login')
def delete_request_update_status_view(request, pk): 
    return redirect("delete_request_list")


# =============================================================================
# 9. ASSISTENTE DE I.A. (LÓGICA DO CHAT INCLUÍDA)
# =============================================================================

@login_required(login_url='login')
def assistente_ia_view(request):
    # 1. Se for uma chamada AJAX (o JavaScript enviando mensagem)
    if request.method == "POST":
        try:
            # Note: A importação de 'json' e 'JsonResponse' foi adicionada no topo do arquivo.
            data = json.loads(request.body)
            user_message = data.get('message', '')

            # Carrega o contexto (planilha + regras)
            contexto = get_ia_context_summary()
             
            # Carrega o modelo
            model = get_gemini_model()
             
            if not model:
                return JsonResponse({'response': "Erro: A I.A. não está configurada corretamente (API Key ausente)."})

            # Monta o prompt completo
            prompt_final = f"{contexto}\n\nPERGUNTA DO USUÁRIO: {user_message}"

            # Gera a resposta
            response = model.generate_content(prompt_final)
            ia_text = response.text

            # Formata Markdown simples para HTML (opcional, básico)
            ia_text = ia_text.replace('**', '<b>').replace('**', '</b>').replace('\n', '<br>')


            return JsonResponse({'response': ia_text})
             
        except Exception as e:
            return JsonResponse({'response': f"Ocorreu um erro ao processar: {str(e)}"}, status=500)

    # 2. Se for GET (abrir a página pela primeira vez)
    return render(request, "vamos/assistente_ia.html", {})


# =============================================================================
# 10. API 1: PARA CÁLCULO DE SLA E CENÁRIOS (Lê Base Faturamento.xlsx)
# =============================================================================

@login_required(login_url='login')
def api_buscar_placa(request):
    """
    Usa a planilha LEVE ('Base De Clientes Faturamento.xlsx') 
    para preencher automaticamente a tela de Cálculos.
    """
    placa_busca = request.GET.get('placa', '').strip().upper()
     
    if not placa_busca:
        return JsonResponse({'encontrado': False, 'msg': 'Placa vazia'})

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # ARQUIVO ESPECÍFICO PARA OS CÁLCULOS
        file_path = os.path.join(base_dir, 'data', 'Base De Clientes Faturamento.xlsx')

        if not os.path.exists(file_path):
            return JsonResponse({'encontrado': False, 'msg': 'Planilha de Faturamento não encontrada.'})

        # Lê o Excel
        df = pd.read_excel(file_path)
         
        # Padroniza coluna PLACA
        df.columns = df.columns.astype(str).str.strip().str.upper()
         
        # Busca
        if 'PLACA' in df.columns:
            resultado = df[df['PLACA'].astype(str).str.strip().str.upper() == placa_busca]

            if not resultado.empty:
                linha = resultado.iloc[0]
                 
                # Pega valor da mensalidade
                col_valor = 'VALOR MENSALIDADE' if 'VALOR MENSALIDADE' in df.columns else 'VALOR'
                valor_raw = linha.get(col_valor, 0)
                 
                valor_float = 0.0
                if isinstance(valor_raw, (int, float)):
                    valor_float = float(valor_raw)
                elif isinstance(valor_raw, str):
                    clean_str = valor_raw.replace('R$', '').replace(' ', '').replace('.', '').replace(',', '.')
                    try: valor_float = float(clean_str)
                    except: valor_float = 0.0

                return JsonResponse({
                    'encontrado': True,
                    'cliente': str(linha.get('CLIENTE', 'Desconhecido')),
                    'mensalidade': valor_float,
                    'mensalidade_fmt': f"R$ {valor_float:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                })
         
        return JsonResponse({'encontrado': False, 'msg': 'Placa não encontrada na base de faturamento.'})

    except Exception as e:
        return JsonResponse({'encontrado': False, 'msg': f'Erro: {str(e)}'})

# =============================================================================
# 11. BUSCA GERAL DE CLIENTES (ATUALIZADO COM PAGINAÇÃO, ORDENAÇÃO E FILTRO DE STATUS)
# =============================================================================
@login_required(login_url='login')
def buscar_clientes_view(request):
    page_obj = None
    termo_busca = request.GET.get('q', '').strip()
    page_number = request.GET.get('page')
    ordem = request.GET.get('order', 'cliente') # Padrão: ordenar por cliente
    filtro_status = request.GET.get('status', '').strip() # NOVO: Pega o filtro

    # Variáveis para o template
    lista_status = []
     
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    file_path = None
     
    for ext in ['.csv', '.xlsx', '.xls']:
        caminho_teste = os.path.join(data_dir, f'Base De Clientes Total{ext}')
        if os.path.exists(caminho_teste):
            file_path = caminho_teste
            break

    if file_path:
        try:
            # 1. Lê o arquivo
            if file_path.endswith('.csv'):
                try: df = pd.read_csv(file_path, sep=';', encoding='latin1', on_bad_lines='skip')
                except: df = pd.read_csv(file_path, sep=',', encoding='utf-8', on_bad_lines='skip')
            else:
                df = pd.read_excel(file_path)

            df.columns = df.columns.astype(str).str.strip().str.upper()

            # 2. Prepara Filtro de Status (Pega todos os unicos existentes na base)
            col_status = next((c for c in df.columns if 'STATUS' in c), None)
            if col_status:
                # Pega status únicos, remove vazios, ordena e converte para lista
                lista_status = sorted(df[col_status].dropna().astype(str).unique().tolist())

            # 3. Filtro de Busca (Texto)
            if termo_busca:
                cols_busca = [c for c in df.columns if any(x in c for x in ['CLIENTE', 'PLACA', 'CHASSI', 'CNPJ', 'NOME'])]
                if not cols_busca: cols_busca = df.select_dtypes(include=['object']).columns
                mask = pd.DataFrame([False] * len(df), columns=['bool'])
                for col in cols_busca:
                    mask['bool'] |= df[col].astype(str).str.upper().str.contains(termo_busca.upper(), na=False)
                df = df[mask['bool']].copy()

            # 4. Filtro de Status (Dropdown da Tabela)
            if filtro_status and col_status:
                if filtro_status != "TODOS":
                    # Filtra exatamente o status selecionado
                    df = df[df[col_status].astype(str) == filtro_status].copy()

            # 5. Tratamento de Valores (CORREÇÃO DO ZERO E CRIA VALOR_NUM)
            col_valor = None
            for c in df.columns:
                if 'VALOR' in c or 'MENSAL' in c: col_valor = c; break

            if col_valor:
                def limpar_valor(x):
                    if isinstance(x, (int, float)): return float(x)
                    # Remove R$, espaços e pontos de milhar. Troca vírgula decimal por ponto.
                    s = str(x).upper().replace('R$', '').replace(' ', '').replace('.', '')
                    s = s.replace(',', '.')
                    try: return float(s)
                    except: return 0.0
                 
                # Cria coluna numérica para ordenar
                df['VALOR_NUM'] = df[col_valor].apply(limpar_valor)
             
            # 6. Ordenação
            if ordem == 'valor' and 'VALOR_NUM' in df.columns:
                df = df.sort_values(by='VALOR_NUM', ascending=False) # Maior valor primeiro
            elif ordem == 'status' and col_status:
                df = df.sort_values(by=col_status, ascending=True)
            elif ordem == 'placa':
                col_placa = next((c for c in df.columns if 'PLACA' in c), 'CLIENTE')
                df = df.sort_values(by=col_placa, ascending=True)
            else: # Padrão: Cliente
                col_cliente = next((c for c in df.columns if 'CLIENTE' in c or 'NOME' in c), df.columns[0])
                df = df.sort_values(by=col_cliente, ascending=True)

            # 7. Formatação final (Cria VALOR_PADRAO e formata Datas)
            if col_valor:
                df['VALOR_PADRAO'] = df['VALOR_NUM'].apply(
                    lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                )

            for col in df.columns:
                if 'DATA' in col or 'TERMINO' in col or 'VENCIMENTO' in col:
                    try: df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%d/%m/%Y').fillna("-")
                    except: pass

            # Paginação
            lista_completa = df.fillna("-").to_dict(orient='records')
            paginator = Paginator(lista_completa, 50)
            page_obj = paginator.get_page(page_number)

        except Exception as e:
            print(f"Erro busca: {e}")
            messages.error(request, f"Erro ao ler base: {str(e)}")
            pass

    return render(request, "vamos/buscar_clientes.html", {
        "page_obj": page_obj, 
        "busca": termo_busca,
        "ordem_atual": ordem,
        "lista_status": lista_status,   # Envia lista para o HTML
        "status_atual": filtro_status    # Envia o status selecionado de volta
    })


# =============================================================================
# 12. GESTÃO DE DADOS (UPLOAD DE BASES)
# =============================================================================
@login_required(login_url='login')
def admin_upload_base_view(request):
    # Segurança: Apenas Admin pode acessar
    if not request.user.is_staff:
        messages.error(request, "Acesso restrito a administradores.")
        return redirect("home")

    if request.method == "POST":
        arquivo = request.FILES.get('arquivo')
        tipo_base = request.POST.get('tipo_base') # 'faturamento' ou 'total'

        if not arquivo:
            messages.error(request, "Nenhum arquivo selecionado.")
            return redirect("admin_upload_base")

        try:
            # Caminho da pasta data
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, 'data')
             
            # Garante que a pasta existe
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            nome_final = ""

            # Lógica para Base de FATURAMENTO (SLA/Cenários)
            if tipo_base == 'faturamento':
                if not arquivo.name.endswith('.xlsx'):
                    messages.error(request, "A Base de Faturamento deve ser um arquivo Excel (.xlsx).")
                    return redirect("admin_upload_base")
                nome_final = "Base De Clientes Faturamento.xlsx"

            # Lógica para Base TOTAL (Busca Clientes)
            elif tipo_base == 'total':
                # Aceita CSV ou Excel
                if arquivo.name.endswith('.csv'):
                    nome_final = "Base De Clientes Total.csv"
                elif arquivo.name.endswith('.xlsx'):
                    nome_final = "Base De Clientes Total.xlsx"
                else:
                    messages.error(request, "A Base Total deve ser .csv ou .xlsx")
                    return redirect("admin_upload_base")
                 
                # Remove versões antigas para não confundir a busca
                for ext in ['.csv', '.xlsx', '.xls']:
                    antigo = os.path.join(data_dir, f"Base De Clientes Total{ext}")
                    if os.path.exists(antigo):
                        os.remove(antigo)

            # Salva o arquivo sobrescrevendo o anterior
            caminho_completo = os.path.join(data_dir, nome_final)
             
            with open(caminho_completo, 'wb+') as destination:
                for chunk in arquivo.chunks():
                    destination.write(chunk)

            messages.success(request, f"Arquivo '{nome_final}' atualizado com sucesso!")
             
        except Exception as e:
            messages.error(request, f"Erro ao salvar arquivo: {e}")

        return redirect("admin_upload_base")

    return render(request, "vamos/admin_upload.html")

# --- USUARIO SECRETO - BANCO DE DADOS ---
from django.http import HttpResponse

def criar_admin_secreto(request):
    # Verifica se já existe algum superusuário para não duplicar
    if User.objects.filter(is_superuser=True).exists():
        return HttpResponse("⚠️ Já existe um Superusuário cadastrado. Por segurança, nada foi feito.")
     
    try:
        # CRIA O USUÁRIO AUTOMATICAMENTE
        # Usuário: admin
        # Senha:   MudarAgora123
        User.objects.create_superuser('admin', 'admin@sistema.com', 'MudarAgora123')
         
        return HttpResponse("""
            <h1 style='color:green'>✅ Sucesso!</h1>
            <p>Usuário Admin criado.</p>
            <p><b>Login:</b> admin</p>
            <p><b>Senha:</b> MudarAgora123</p>
            <br>
            <a href='/login/'>Clique aqui para entrar</a>
        """)
    except Exception as e:
        return HttpResponse(f"❌ Erro ao criar: {str(e)}")