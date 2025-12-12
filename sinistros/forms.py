from django import forms
from .models import Sinistro

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
    # Campos adicionais (não pertencem ao modelo) usados pelo template
    status_os = forms.CharField(required=False, label='Status da O.S.', widget=forms.Textarea(attrs={'rows': 3}))
    check_laudo = forms.BooleanField(required=False, label='Laudo Pericial')

    class Meta:
        model = Sinistro
        fields = [
            'setor_atual',
            'responsavel_setor',
            # 'status_os' NÃO aqui — é declarado acima como campo do form
            'retornar_ate',
            'observacoes',
            # campos financeiros — inclua conforme necessidade
            'valor_fipe',
            'valor_implemento',
            'valor_franquia',
            'valor_seguradora',
            'valor_cliente',
            # checklist existentes no modelo
            'check_bo',
            'check_ficha',
            'check_cnh',
            'check_fotos',
        ]
        widgets = {
            'retornar_ate': forms.DateInput(attrs={'type': 'date'}),
            'observacoes': forms.Textarea(attrs={'rows': 3}),
        }
        labels = {
            'retornar_ate': 'Prazo Limite',
        }

# --- FORMULÁRIO DE UPLOAD ---
class UploadBaseForm(forms.Form):
    arquivo = forms.FileField(label="Selecione a Planilha (Excel ou CSV)")