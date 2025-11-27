"""
Django settings for vamos_frotas_sla project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv # <--- Adicione isso

# Carrega as variáveis do arquivo .env
load_dotenv()

# Caminho base do projeto (onde fica o manage.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# === SEGURANÇA ===
# Em produção, use variáveis de ambiente. Para dev, isso funciona.
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-insecure-secret-key")
DEBUG = os.getenv("DJANGO_DEBUG", "True") == "True"

ALLOWED_HOSTS = os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")


# === APPS INSTALADOS ===
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    
    # Seu app principal
    "vamos",
    "accounts",
]

# === MIDDLEWARE ===
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
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
        "DIRS": [BASE_DIR / "templates"],  # Procura templates na pasta global 'templates' se existir
        "APP_DIRS": True,                  # Procura templates dentro de cada app (vamos/templates/vamos)
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "vamos_frotas_sla.wsgi.application"


# === BANCO DE DADOS ===
# CUIDADO: Se o PostgreSQL não estiver instalado/rodando, vai dar erro "Connection Refused".
# Se der erro, comente este bloco e descomente o bloco do SQLite abaixo.

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("POSTGRES_DB", "vamos_frotas_sla"),
        "USER": os.getenv("POSTGRES_USER", "postgres"),
        "PASSWORD": os.getenv("POSTGRES_PASSWORD", "Holanda2609"),
        "HOST": os.getenv("POSTGRES_HOST", "localhost"),
        "PORT": os.getenv("POSTGRES_PORT", "5432"),
    }
}

# --- OPÇÃO SQLITE (Use se o Postgres der erro) ---
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': BASE_DIR / 'db.sqlite3',
#     }
# }
# -------------------------------------------------


# === VALIDAÇÃO DE SENHA ===
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 8}}, # Ajustado para 8 (padrão web)
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# === INTERNACIONALIZAÇÃO ===
LANGUAGE_CODE = "pt-br"
TIME_ZONE = "America/Sao_Paulo"
USE_I18N = True
USE_TZ = True


# === ARQUIVOS ESTÁTICOS (CSS, JS, Imagens) ===
STATIC_URL = "/static/"

# Importante: Diz ao Django para buscar na pasta 'static' na raiz do projeto
STATICFILES_DIRS = [
    BASE_DIR / "static",
]

# Em produção, descomente:
# STATIC_ROOT = BASE_DIR / "staticfiles"


# === ARQUIVOS DE MÍDIA (Uploads, PDFs gerados) ===
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"


# === LOGIN / LOGOUT (Essenciais para @login_required) ===
# Se tentar acessar página restrita, vai para:
LOGIN_URL = "login"
# Depois de logar, vai para:
LOGIN_REDIRECT_URL = "home"
# Depois de deslogar, vai para:
LOGOUT_REDIRECT_URL = "login"


# === CONFIGURAÇÃO PADRÃO ===
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# === CONFIGURAÇÕES I.A. (GEMINI) ===
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

