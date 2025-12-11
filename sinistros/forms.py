from django import forms
from .models import Sinistro

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

# --- FORMULÁRIO DE UPLOAD ---
class UploadBaseForm(forms.Form):
    arquivo = forms.FileField(label="Selecione a Planilha (Excel ou CSV)")