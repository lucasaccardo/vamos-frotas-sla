from django import forms
from .models import Sinistro

class SinistroForm(forms.ModelForm):
    class Meta:
        model = Sinistro
        fields = '__all__'
        exclude = ['criado_por', 'data_inicio_tratativa', 'ultima_interacao', 'total_a_pagar', 'total_pago']
        widgets = {
            'data_ocorrencia': forms.DateInput(attrs={'type': 'date'}),
            'retornar_ate': forms.DateInput(attrs={'type': 'date'}),
            'data_envio_simpar': forms.DateInput(attrs={'type': 'date'}),
            'data_retorno_simpar': forms.DateInput(attrs={'type': 'date'}),
            'observacoes': forms.Textarea(attrs={'rows': 3}),
        }