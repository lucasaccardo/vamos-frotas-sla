from .models import Ticket

def notificacoes_globais(request):
    """
    Calcula contadores globais para exibir na sidebar em todas as telas.
    """
    if not request.user.is_authenticated:
        return {}

    ticket_count = 0

    # Lógica para ADMIN: Conta tickets que estão "Pendente" (precisam de resposta)
    if request.user.is_staff:
        ticket_count = Ticket.objects.filter(status='Pendente').count()
    
    # Lógica para USUÁRIO COMUM: Conta tickets "Em andamento" (que o admin respondeu)
    # ou "Concluído" (que foi fechado recentemente)
    else:
        ticket_count = Ticket.objects.filter(
            usuario=request.user, 
            status__in=['Em andamento', 'Concluído']
        ).count()

    return {
        'sidebar_ticket_count': ticket_count
    }