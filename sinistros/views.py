from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required(login_url='login')
def sinistros_home_view(request):
    """Dashboard Principal do Módulo de Sinistros."""
    # Define na sessão que estamos neste módulo
    request.session['modulo_ativo'] = 'sinistros'
    
    return render(request, "sinistros/home.html")