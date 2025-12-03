from django import forms
from django.utils import timezone
# Importações necessárias para o User, Perfil e herdar de UserCreationForm
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Ticket, Perfil # Importamos Ticket e Perfil

# === 1. Formulário de SLA Mensal ===
class SlaMensalForm(forms.Form):
    # === 1) Informações Obrigatórias ===
    os_chamado = forms.CharField(
        label="Chamado O.S",
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Digite o chamado'})
    )
    
    ferramenta = forms.ChoiceField(
        label="Ferramenta",
        choices=[("", "Selecione..."), ("Vetor", "Vetor"), ("Geo", "Geo")],
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    # === 2) Identificação ===
    placa = forms.CharField(
        label="Placa do Veículo",
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'XXX-0000'})
    )
    
    cliente = forms.CharField(
        label="Cliente",
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    mensalidade = forms.DecimalField(
        label="Mensalidade (R$)",
        max_digits=10, 
        decimal_places=2,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )

    # === 3) Período e Serviço ===
    data_entrada = forms.DateField(
        label="Data de Entrada",
        initial=timezone.now().date(),
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    
    data_saida = forms.DateField(
        label="Data de Saída",
        initial=timezone.now().date() + timezone.timedelta(days=3),
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    
    feriados = forms.IntegerField(
        label="Feriados no período",
        initial=0,
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    TIPO_SERVICO_CHOICES = [
        ("Preventiva – 2 dias úteis", "Preventiva – 2 dias úteis"),
        ("Corretiva – 3 dias úteis", "Corretiva – 3 dias úteis"),
        ("Preventiva + Corretiva – 5 dias úteis", "Preventiva + Corretiva – 5 dias úteis"),
        ("Motor – 15 dias úteis", "Motor – 15 dias úteis"),
    ]
    
    tipo_servico = forms.ChoiceField(
        label="Tipo de Serviço (SLA)",
        choices=TIPO_SERVICO_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )

# ---

# === 2. Formulários de Cenários ===
class CenarioForm(forms.Form):
    # Campos globais
    os_chamado = forms.CharField(
        label="Chamado O.S", 
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Ex: 12345'})
    )
    ferramenta = forms.ChoiceField(
        label="Ferramenta",
        required=False,
        choices=[("", "Selecione..."), ("Vetor", "Vetor"), ("Geo", "Geo")],
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    # Campos do Cenário Específico
    placa = forms.CharField(
        label="Placa",
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'XXX-0000'})
    )
    
    cliente_nome = forms.CharField(
        label="Cliente", 
        required=False, 
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    valor_mensalidade = forms.DecimalField(
        label="Mensalidade", 
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )

    data_entrada = forms.DateField(
        label="Entrada",
        initial=timezone.now().date(),
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    data_saida = forms.DateField(
        label="Saída",
        initial=timezone.now().date() + timezone.timedelta(days=5),
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    feriados = forms.IntegerField(
        label="Feriados", 
        initial=0, 
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    tipo_servico = forms.ChoiceField(
        label="Serviço",
        choices=[
            ("Preventiva – 2 dias úteis", "Preventiva – 2 dias úteis"),
            ("Corretiva – 3 dias úteis", "Corretiva – 3 dias úteis"),
            ("Preventiva + Corretiva – 5 dias úteis", "Preventiva + Corretiva – 5 dias úteis"),
            ("Motor – 15 dias úteis", "Motor – 15 dias úteis"),
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )


class PecaForm(forms.Form):
    nome_peca = forms.CharField(
        label="Nome da Peça",
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Ex: Filtro de Óleo'})
    )
    valor_peca = forms.DecimalField(
        label="Valor (R$)",
        min_value=0.01,
        decimal_places=2,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '0,00', 'step': '0.01'})
    )

# ---

# === 3. Formulário de Ticket ===
class TicketForm(forms.Form):
    assunto = forms.CharField(
        label="Assunto",
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Resumo do problema'})
    )
    descricao = forms.CharField(
        label="Descrição Detalhada",
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Descreva o erro ou sugestão...'})
    )
    anexo = forms.FileField(
        label="Anexo (Print/Imagem)",
        required=False,
        widget=forms.ClearableFileInput(attrs={'class': 'form-control'})
    )

# ---

# === 4. Formulário de Cadastro Público (SignUpForm AGORA MELHORADO) ===
class SignUpForm(UserCreationForm):
    full_name = forms.CharField(
        label="Nome completo", 
        max_length=100, 
        widget=forms.TextInput(attrs={'class': 'form-control bg-dark text-light border-secondary', 'placeholder': 'Nome completo'})
    )
    email = forms.EmailField(
        label="E-mail corporativo", 
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control bg-dark text-light border-secondary', 'placeholder': 'seu.email@grupovamos.com.br'})
    )
    matricula = forms.CharField(
        label="Matrícula", 
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control bg-dark text-light border-secondary', 'placeholder': '00000'})
    )

    class Meta:
        model = User
        fields = ('username', 'full_name', 'matricula', 'email')

    def __init__(self, *args, **kwargs):
        super(SignUpForm, self).__init__(*args, **kwargs)
        # Estilizando os campos padrões do Django (username, senha)
        self.fields['username'].widget.attrs.update({'class': 'form-control bg-dark text-light border-secondary', 'placeholder': 'Login'})
        
        # Ajuste para os campos de senha aparecerem estilizados
        for field_name in self.fields:
            if 'password' in field_name:
                self.fields[field_name].widget.attrs.update({'class': 'form-control bg-dark text-light border-secondary'})

    def save(self, commit=True):
        user = super(SignUpForm, self).save(commit=False)
        
        # Salva o Nome e Sobrenome separando o nome completo
        full_name = self.cleaned_data['full_name']
        names = full_name.split()
        user.first_name = names[0]
        # Salva o restante como last_name, ou vazio se for apenas um nome
        user.last_name = " ".join(names[1:]) if len(names) > 1 else "" 
        user.email = self.cleaned_data['email']
        
        if commit:
            user.save()
            # Salva a matrícula no Perfil, que foi criado pelo signal
            if hasattr(user, 'perfil'):
                user.perfil.matricula = self.cleaned_data['matricula']
                user.perfil.save()
        return user

# ---

# === 5. ADMIN - GERENCIAR USUÁRIOS (Tela Escura) ===
class AdminUserForm(forms.Form):
    # Note as classes 'bg-dark text-light' para ficar escuro igual ao seu print
    username = forms.CharField(
        label="Usuário (Login)",
        widget=forms.TextInput(attrs={'class': 'form-control bg-dark text-light border-secondary'})
    )
    first_name = forms.CharField(
        label="Nome Completo",
        widget=forms.TextInput(attrs={'class': 'form-control bg-dark text-light border-secondary'})
    )
    # Adicionando o campo Matrícula conforme solicitado
    matricula = forms.CharField(
        label="Matrícula",
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control bg-dark text-light border-secondary'})
    )
    email = forms.EmailField(
        label="E-mail",
        widget=forms.EmailInput(attrs={'class': 'form-control bg-dark text-light border-secondary'})
    )
    
    TIPO_CHOICES = [('user', 'Usuário Comum'), ('admin', 'Admin')]
    role = forms.ChoiceField(
        label="Tipo de Acesso",
        choices=TIPO_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select bg-dark text-light border-secondary'})
    )
    
    password = forms.CharField(
        label="Senha (Opcional - Preencha apenas se quiser alterar)",
        required=False,
        widget=forms.PasswordInput(attrs={'class': 'form-control bg-dark text-light border-secondary', 'placeholder': 'Nova senha...'})
    )
    
    is_active = forms.BooleanField(
        label="Aprovar / Ativar Usuário",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

# --- FORMULÁRIO DE PERFIL ---
class PerfilForm(forms.ModelForm):
    class Meta:
        model = Perfil
        fields = ['foto'] # Só queremos que ele mude a foto por enquanto
        widgets = {
            'foto': forms.FileInput(attrs={'class': 'form-control'})
        }