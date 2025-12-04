"""
Django settings for vamos_frotas_sla project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv
import dj_database_url  # Importante para o banco na nuvem

# Carrega as variáveis do arquivo .env (apenas localmente)
load_dotenv()

# Caminho base do projeto (onde fica o manage.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# === SEGURANÇA ===
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-insecure-secret-key")
# Se DJANGO_DEBUG não estiver definido (na nuvem), assume False por segurança
DEBUG = os.getenv("DJANGO_DEBUG", "True") == "True"

# Permite qualquer host (necessário para o Render funcionar sem configurar domínio)
ALLOWED_HOSTS = ["*"]


# === APPS INSTALADOS ===
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    
    # Biblioteca para AWS S3 (Necessária para a configuração nova)
    "storages", 

    # Seus Apps
    "vamos",      # Core/Manutenção
    "accounts",   # Login
    "sinistros",  # Novo App de Sinistros (Correto!)
]

# === MIDDLEWARE (ORDEM IMPORTA!) ===
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware", # Essencial para CSS na nuvem
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "vamos_frotas_sla.urls"

# === TEMPLATES (HTML) ===
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                
                # --- NOTIFICAÇÕES GLOBAIS ---
                # Certifique-se que o arquivo vamos/context_processors.py existe!
                "vamos.context_processors.notificacoes_globais",
            ],
        },
    },
]

WSGI_APPLICATION = "vamos_frotas_sla.wsgi.application"


# === BANCO DE DADOS (LÓGICA INTELIGENTE) ===
# Se tiver DATABASE_URL (Nuvem), usa ele. Senão, usa SQLite (Local).
DATABASES = {
    'default': dj_database_url.config(
        default=os.getenv('DATABASE_URL', 'sqlite:///db.sqlite3'),
        conn_max_age=600
    )
}


# === VALIDAÇÃO DE SENHA ===
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 8}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# === INTERNACIONALIZAÇÃO ===
LANGUAGE_CODE = "pt-br"
TIME_ZONE = "America/Sao_Paulo"
USE_I18N = True
USE_TZ = True


# === ARQUIVOS ESTÁTICOS (CSS, JS, Imagens do Site) ===
# Continuam no Render usando Whitenoise (mais rápido e barato)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [BASE_DIR / "static"]
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# === ARQUIVOS DE MÍDIA (PDFs, Uploads) -> VÃO PARA AMAZON S3 ===
# Configurações da AWS (Lê das variáveis de ambiente)
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
AWS_S3_REGION_NAME = 'us-east-1' # Região padrão (Norte da Virgínia)
AWS_S3_SIGNATURE_VERSION = 's3v4'

# Configurações de Upload
AWS_DEFAULT_ACL = None
AWS_S3_FILE_OVERWRITE = False # Não substitui arquivos com mesmo nome (cria cópia)

# Dita as regras: Static no Render, Media na Amazon
STORAGES = {
    "default": {
        "BACKEND": "storages.backends.s3.S3Storage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

# URL base para acessar os arquivos na Amazon
# (Isso faz o link do PDF funcionar no site)
MEDIA_URL = f'https://{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/'


# === LOGIN / LOGOUT ===
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "portal"  # <--- ALTERADO: Redireciona para o Portal após login
LOGOUT_REDIRECT_URL = "login"


# === CONFIGURAÇÃO PADRÃO ===
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# === CONFIGURAÇÕES I.A. (GEMINI) ===
# Sua chave real
GOOGLE_API_KEY = "AIzaSyA821yX6bOVatN5bf2BNikhAhngRSlo6p4" 
GEMINI_MODEL = "gemini-2.0-flash"

# === CONFIGURAÇÃO DE E-MAIL (GMAIL) ===
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD')
DEFAULT_FROM_EMAIL = os.getenv('DEFAULT_FROM_EMAIL')

# ==========================================
# CONFIGURAÇÃO DE SESSÃO (TIMEOUT)
# ==========================================
# Tempo em segundos que a sessão dura (30 minutos x 60 segundos = 1800)
SESSION_COOKIE_AGE = 1800

# Se True, o tempo reseta a cada clique/ação do usuário.
# Se ficar False, o usuário cai após 30min mesmo se estiver trabalhando.
SESSION_SAVE_EVERY_REQUEST = True

# Segurança extra: Fecha a sessão se fechar o navegador
SESSION_EXPIRE_AT_BROWSER_CLOSE = True