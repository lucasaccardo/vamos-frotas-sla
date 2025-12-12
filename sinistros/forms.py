from django import forms
from .models import Sinistro

# --- FORMULÁRIO DE CRIAÇÃO (Novo Sinistro) ---
class SinistroForm(forms.ModelForm):
    class Meta:
        model = Sinistro
        fields = '__all__'
        exclude = ['criado_por', 'criado_em', 'ultima_interacao', 'total_a_pagar', 'total_pago']
        widgets = {
            'data_ocorrencia': forms.DateInput(attrs={'type': 'date'}),
            'retornar_ate': forms.DateInput(attrs={'type': 'date'}),
            'observacoes': forms.Textarea(attrs={'rows': 3}),
        }

# --- FORMULÁRIO DE EDIÇÃO (apenas campos que o usuário pode alterar no fluxo) ---
class EditarSinistroForm(forms.ModelForm):
    # Sobrescrevemos status_os para ser required=False no HTML (validação será no clean)
    # e definimos o widget dele.
    status_os = forms.CharField(
        required=False, 
        label='Status da O.S.', 
        widget=forms.Textarea(attrs={'rows': 2, 'class': 'form-control'})
    )
    
    # Checkbox que faltava na lista anterior
    check_laudo = forms.BooleanField(required=False, label='Laudo Pericial')

    class Meta:
        model = Sinistro
        fields = [
            'setor_atual',
            'responsavel_setor',
            'status_os',  # Importante estar aqui para salvar no banco
            'retornar_ate',
            'observacoes',
            # campos financeiros
            'valor_fipe',
            'valor_implemento',
            'valor_franquia',
            'valor_seguradora',
            'valor_cliente',
            # checklist
            'check_bo',
            'check_ficha',
            'check_cnh',
            'check_fotos',
            'check_laudo', # Adicionado aqui também
        ]
        widgets = {
            'retornar_ate': forms.DateInput(attrs={'type': 'date'}),
            'observacoes': forms.Textarea(attrs={'rows': 3}),
        }
        labels = {
            'retornar_ate': 'Prazo Limite',
        }

    # --- VALIDAÇÃO CONDICIONAL ---
    def clean(self):
        cleaned = super().clean()
        
        # Pega o setor do formulário, ou se não foi enviado, pega o da instância atual
        setor = cleaned.get('setor_atual') or self.instance.setor_atual
        status = cleaned.get('status_os')

        # Normaliza para maiúsculo para garantir a comparação
        setor_str = str(setor).upper()

        # Regra: Se setor for MANUTENCAO, status é obrigatório
        if setor_str in ['MANUTENCAO', 'MANUTENÇÃO'] and not status:
            self.add_error('status_os', 'O Status da O.S. é obrigatório quando o setor for Manutenção.')
        
        return cleaned

# --- FORMULÁRIO DE UPLOAD ---
class UploadBaseForm(forms.Form):
    arquivo = forms.FileField(label="Selecione a Planilha (Excel ou CSV)")