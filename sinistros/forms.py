from django import forms
from .models import Sinistro

# --- FORMULÁRIO DE CRIAÇÃO (Novo Sinistro) ---
class SinistroForm(forms.ModelForm):
    class Meta:
        model = Sinistro
        fields = '__all__'
        exclude = ['criado_por', 'criado_em', 'ultima_interacao', 'total_a_pagar', 'total_pago', 'aguarda_aprovacao_os', 'aprovador_os']
        widgets = {
            'data_ocorrencia': forms.DateInput(attrs={'type': 'date'}),
            'retornar_ate': forms.DateInput(attrs={'type': 'date'}),
            'observacoes': forms.Textarea(attrs={'rows': 3}),
        }

# --- FORMULÁRIO DE EDIÇÃO (apenas campos que o usuário pode alterar no fluxo) ---
class EditarSinistroForm(forms.ModelForm):
    # Campo status_os mantido como textarea livre
    status_os = forms.CharField(
        required=False, 
        label='Status da O.S.', 
        widget=forms.Textarea(attrs={'rows': 2, 'class': 'form-control'})
    )
    
    check_laudo = forms.BooleanField(required=False, label='Laudo Pericial')

    class Meta:
        model = Sinistro
        fields = [
            'setor_atual',
            'responsavel_setor',
            'status_os',
            'retornar_ate',
            'observacoes',
            
            # Novos campos de Aprovação
            'aguarda_aprovacao_os',
            'aprovador_os',

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
            'check_laudo',
        ]
        widgets = {
            'retornar_ate': forms.DateInput(attrs={'type': 'date'}),
            'observacoes': forms.Textarea(attrs={'rows': 3}),
            'aprovador_os': forms.TextInput(attrs={'placeholder': 'Nome do aprovador'}),
        }
        labels = {
            'retornar_ate': 'Prazo Limite',
            'aguarda_aprovacao_os': 'Aguardando Aprovação de O.S.',
            'aprovador_os': 'Nome do Aprovador',
        }

    # --- VALIDAÇÃO CONDICIONAL ---
    def clean(self):
        cleaned = super().clean()
        
        # Pega o setor do formulário, ou se não foi enviado, pega o da instância atual
        setor = cleaned.get('setor_atual') or (self.instance.setor_atual if self.instance else None)
        status = cleaned.get('status_os')
        
        # Novos campos
        aguarda = cleaned.get('aguarda_aprovacao_os')
        aprovador = cleaned.get('aprovador_os')

        # Normaliza para maiúsculo para garantir a comparação
        setor_str = str(setor).upper() if setor else ''
        is_manutencao = 'MANUT' in setor_str  # Pega MANUTENCAO e MANUTENÇÃO

        # Regra 1: Se setor for MANUTENCAO, status_os é obrigatório (lógica antiga mantida)
        if is_manutencao and not status:
            self.add_error('status_os', 'O Status da O.S. é obrigatório quando o setor for Manutenção.')

        # Regra 2: Se marcou "Aguardando Aprovação", deve informar o Aprovador
        if is_manutencao and aguarda and not aprovador:
            self.add_error('aprovador_os', 'Informe o nome do aprovador quando marcar "Aguardando Aprovação de O.S.".')
        
        return cleaned

# --- FORMULÁRIO DE UPLOAD ---
class UploadBaseForm(forms.Form):
    arquivo = forms.FileField(label="Selecione a Planilha (Excel ou CSV)")