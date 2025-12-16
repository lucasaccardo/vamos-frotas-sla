"""
Django settings for vamos_frotas_sla project.
Versão Final - Segura para Produção (+ Middleware de Timeout)
"""

from pathlib import Path
import os
from dotenv import load_dotenv
import dj_database_url

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# === SEGURANÇA BÁSICA ===
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-insecure-secret-key")

# Por padrão é FALSE (Seguro). Para debugar, altere nas variáveis de ambiente.
DEBUG = os.getenv("DJANGO_DEBUG", "False") == "True"

ALLOWED_HOSTS = ["*"]

# === APPS ===
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    "storages", 
    "vamos",
    "accounts",
    "sinistros",
]

# === MIDDLEWARE ===
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    
    # ⚠️ MIDDLEWARE DE TIMEOUT POR INATIVIDADE (Inserido aqui)
    "sinistros.middleware.SessionIdleTimeout",
    
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "vamos_frotas_sla.urls"

# === TEMPLATES ===
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
                "vamos.context_processors.notificacoes_globais",
            ],
        },
    },
]

WSGI_APPLICATION = "vamos_frotas_sla.wsgi.application"

# === BANCO DE DADOS ===
DATABASES = {
    'default': dj_database_url.config(
        default=os.getenv('DATABASE_URL', 'sqlite:///db.sqlite3'),
        conn_max_age=600
    )
}

# === SENHAS ===
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 8}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# === GERAL ===
LANGUAGE_CODE = "pt-br"
TIME_ZONE = "America/Sao_Paulo"
USE_I18N = True
USE_TZ = True

# === ESTÁTICOS E MÍDIA ===
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [BASE_DIR / "static"]
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Configuração AWS S3 (se as variáveis existirem, usa S3, senão usa local/whitenoise)
if os.getenv('AWS_ACCESS_KEY_ID'):
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
    AWS_S3_REGION_NAME = 'us-east-1'
    AWS_S3_SIGNATURE_VERSION = 's3v4'
    AWS_DEFAULT_ACL = None
    AWS_S3_FILE_OVERWRITE = False

    STORAGES = {
        "default": {"BACKEND": "storages.backends.s3.S3Storage"},
        "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
    }
    MEDIA_URL = f'https://{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/'
else:
    # Fallback para desenvolvimento local
    MEDIA_URL = '/media/'
    MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# === LOGIN ===
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "portal" 
LOGOUT_REDIRECT_URL = "login"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# === I.A. E EMAIL ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA821yX6bOVatN5bf2BNikhAhngRSlo6p4") 
GEMINI_MODEL = "gemini-2.0-flash"

EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD')
DEFAULT_FROM_EMAIL = os.getenv('DEFAULT_FROM_EMAIL')

# ==============================================================================
# === CONFIGURAÇÕES DE SESSÃO E SEGURANÇA AVANÇADA ===
# ==============================================================================

# 1. Configuração de Timeout/Inatividade
SESSION_COOKIE_AGE = 1800           # 30 minutos em segundos (Sessão do Django)
SESSION_SAVE_EVERY_REQUEST = True   # True = Renova o tempo a cada clique (Inatividade)
SESSION_EXPIRE_AT_BROWSER_CLOSE = True # Fecha sessão ao fechar navegador

# Configuração específica para o middleware customizado (Sinistros)
IDLE_TIMEOUT_SECONDS = 1800  # 30 minutos de inatividade para logout forçado

# 2. Configurações de Segurança HTTPS/Cookies
# Aplicar regras estritas APENAS se não estiver em modo DEBUG (Produção)
if not DEBUG:
    # Força cookies apenas via HTTPS
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    
    # Proteção contra XSS e Sniffing
    SESSION_COOKIE_HTTPONLY = True
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    
    # HSTS (HTTP Strict Transport Security) - Força navegadores a usarem HTTPS
    SECURE_HSTS_SECONDS = 31536000  # 1 ano
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    
    # Redirecionamento SSL
    SECURE_SSL_REDIRECT = True
    
    # Essencial para Render/Heroku/AWS (Identifica HTTPS através do proxy)
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
else:
    # Em desenvolvimento (localhost), relaxamos essas regras
    SESSION_COOKIE_SECURE = False
    CSRF_COOKIE_SECURE = False
    SECURE_SSL_REDIRECT = False
