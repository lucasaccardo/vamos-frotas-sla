# sinistros/middleware.py
from django.conf import settings
from django.contrib import auth
from django.utils import timezone
from django.shortcuts import redirect

class SessionIdleTimeout:
    """
    Middleware que finaliza sessão se inatividade > idle timeout.
    Configurar IDLE_TIMEOUT_SECONDS no settings.py (ex.: 1800).
    """
    def __init__(self, get_response):
        self.get_response = get_response
        self.timeout = getattr(settings, 'IDLE_TIMEOUT_SECONDS', None)

    def __call__(self, request):
        if not request.user.is_authenticated or self.timeout is None:
            # atualiza last activity para requests anônimos? opcional
            return self.get_response(request)

        now = timezone.now()
        last_activity = request.session.get('last_activity')
        if last_activity:
            try:
                last = timezone.datetime.fromisoformat(last_activity)
                # timezone aware? converter para timezone.utc/local conforme necessário
            except Exception:
                last = None
        else:
            last = None

        if last:
            elapsed = (now - last).total_seconds()
            if elapsed > self.timeout:
                # logout and optionally redirect to login com ?next=
                auth.logout(request)
                # opcional: mensagem via messages framework
                return redirect('login')  # ajuste nome da view de login
        # atualizar last_activity no final (ou aqui)
        request.session['last_activity'] = now.isoformat()
        return self.get_response(request)
