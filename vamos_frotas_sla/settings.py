"""
Django settings for vamos_frotas_sla project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv
import dj_database_url  # <--- IMPORTANTE PARA O BANCO DA NUVEM

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
    
    # Seu app principal
    "vamos",
    "accounts",
]

# === MIDDLEWARE (ORDEM IMPORTA!) ===
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware", # <--- ADICIONADO (Essencial para CSS na nuvem)
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


# === ARQUIVOS ESTÁTICOS (CSS, JS, Imagens) ===
STATIC_URL = "/static/"

# Onde o Django procura arquivos estáticos durante o desenvolvimento
STATICFILES_DIRS = [
    BASE_DIR / "static",
]

# Onde o Django "junta" todos os arquivos estáticos para a nuvem (CORREÇÃO DO ERRO)
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Motor para servir arquivos estáticos de forma otimizada na nuvem
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'


# === ARQUIVOS DE MÍDIA (Uploads, PDFs gerados) ===
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"


# === LOGIN / LOGOUT ===
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "home"
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