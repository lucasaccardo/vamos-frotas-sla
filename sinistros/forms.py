from django import forms
from .models import Sinistro

# --- FORMULÁRIO DE CRIAÇÃO (Novo Sinistro) ---
class SinistroForm(forms.ModelForm):
    class Meta:
        model = Sinistro
        fields = '__all__'
        # Excluímos campos automáticos ou calculados para não poluir o formulário de edição manual
        exclude = ['criado_por', 'criado_em', 'ultima_interacao', 'total_a_pagar', 'total_pago']
        widgets = {
            'data_ocorrencia': forms.DateInput(attrs={'type': 'date'}),
            'retornar_ate': forms.DateInput(attrs={'type': 'date'}),
            'observacoes': forms.Textarea(attrs={'rows': 3}),
        }

# --- FORMULÁRIO DE EDIÇÃO (Usado na visualização/edição do processo) ---
class EditarSinistroForm(forms.ModelForm):
    class Meta:
        model = Sinistro
        fields = [
            # Workflow
            'setor_atual',
            'responsavel_setor',
            'status_os',
            'retornar_ate',
            'observacoes',
            
            # Financeiro
            'valor_fipe',
            'valor_implemento',
            'valor_franquia',
            'valor_seguradora',
            'valor_cliente',
            
            # Documentos (Necessário para a aba Documentos funcionar)
            'check_bo',
            'check_cnh',
            'check_fotos',
            'check_laudo',
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