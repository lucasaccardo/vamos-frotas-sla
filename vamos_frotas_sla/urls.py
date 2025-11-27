from django.contrib import admin
from django.urls import path, include

# --- ADICIONE ESTAS DUAS IMPORTAÇÕES ---
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Redireciona tudo para o arquivo de urls do app 'vamos'
    path('', include('vamos.urls')), 
]

# --- ADICIONE ESTE BLOCO NO FINAL ---
# Isso permite que o navegador acesse os PDFs na sua pasta local
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)