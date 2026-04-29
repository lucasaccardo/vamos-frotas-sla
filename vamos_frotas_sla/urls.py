from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from two_factor.urls import urlpatterns as tf_urls

# --- ADICIONADO: Importação do sistema 2FA ---
from two_factor.urls import urlpatterns as tf_urls

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # --- ADICIONADO: Rotas de Autenticação do 2FA ---
    path('', include(tf_urls)),
    
    path('', include('vamos.urls')),        # Manda para o app principal
    path('sinistros/', include('sinistros.urls')), # Manda para o NOVO app
    path('api/procedures/', include('procedures.urls')), # API para procedures
    path('procedures/', include('procedures.web_urls')), # Web UI para procedures
]

# Isso permite que o navegador acesse os PDFs na sua pasta local
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
