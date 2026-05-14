"""
Django settings for vamos_frotas_sla project.
Versão Final - Segura para Produção e Auditada (Projeto Integrador)
"""

from pathlib import Path
import os
import sys
from urllib.parse import urlparse
from dotenv import load_dotenv
import dj_database_url
from django.core.exceptions import ImproperlyConfigured

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RUNNING_TESTS = "test" in sys.argv
SECURITY_AUDIT_LOG_PATH = os.getenv(
    "SECURITY_AUDIT_LOG_PATH",
    str(BASE_DIR / "logs" / "security_audit.log"),
)

# === SEGURANÇA BÁSICA ===
_is_production_environment = bool(os.getenv("RENDER_EXTERNAL_URL")) or os.getenv("RENDER", "").lower() == "true"
_default_debug = "False" if _is_production_environment else "True"
DEBUG = os.getenv("DJANGO_DEBUG", _default_debug).lower() == "true"

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = "dev-only-insecure-secret-key-change-me"
    else:
        raise ImproperlyConfigured(
            "DJANGO_SECRET_KEY é obrigatório quando DJANGO_DEBUG=False."
        )

_allowed_hosts_env = os.getenv("DJANGO_ALLOWED_HOSTS", "")
ALLOWED_HOSTS = [host.strip() for host in _allowed_hosts_env.split(",") if host.strip()]
if not ALLOWED_HOSTS:
    if DEBUG:
        ALLOWED_HOSTS = ["localhost", "127.0.0.1", "[::1]"]
    else:
        raise ImproperlyConfigured(
            "DJANGO_ALLOWED_HOSTS é obrigatório quando DJANGO_DEBUG=False."
        )

CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if origin.strip()
]
_render_external_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()
if _render_external_url:
    parsed = urlparse(_render_external_url)
    if parsed.scheme and parsed.netloc:
        render_origin = f"{parsed.scheme}://{parsed.netloc}"
        if render_origin not in CSRF_TRUSTED_ORIGINS:
            CSRF_TRUSTED_ORIGINS.append(render_origin)

# === APPS INSTALADOS ===
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    "storages", 
    "rest_framework",
    "vamos",
    "accounts",
    "sinistros",
    "procedures",
    
    # --- ADICIONADO: Bibliotecas do Tópico 2 (2FA) ---
    'django_otp',
    'django_otp.plugins.otp_static',
    'django_otp.plugins.otp_totp',
    'two_factor',
    'two_factor.plugins.phonenumber',
    'axes',
]

# === MIDDLEWARE ===
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    
    # --- ADICIONADO: Middleware que obriga a checagem do Token 2FA ---
    "django_otp.middleware.OTPMiddleware",
    
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    'axes.middleware.AxesMiddleware',
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
            ],
        },
    },
]

WSGI_APPLICATION = "vamos_frotas_sla.wsgi.application"

# === BANCO DE DADOS ===
# Em producao, use PostgreSQL via DATABASE_URL. O fallback para SQLite existe
# apenas para desenvolvimento local sem banco configurado.
_database_url = os.getenv("DATABASE_URL", "").strip()
_database_ssl_required = os.getenv("DATABASE_SSL_REQUIRE", "False").lower() == "true"

if _database_url:
    DATABASES = {
        "default": dj_database_url.config(
            default=_database_url,
            conn_max_age=600,
            ssl_require=_database_ssl_required,
        )
    }
elif os.getenv("POSTGRES_DB"):
    _database_options = {}
    if _database_ssl_required:
        _database_options["sslmode"] = "require"
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("POSTGRES_DB"),
            "USER": os.getenv("POSTGRES_USER", "postgres"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
            "HOST": os.getenv("POSTGRES_HOST", "localhost"),
            "PORT": os.getenv("POSTGRES_PORT", "5432"),
            "CONN_MAX_AGE": 600,
            "OPTIONS": _database_options,
        }
    }
elif DEBUG:
    DATABASES = {
        "default": dj_database_url.config(
            default="sqlite:///db.sqlite3",
            conn_max_age=600,
        )
    }
else:
    raise ImproperlyConfigured(
        "Configure DATABASE_URL ou POSTGRES_DB para usar PostgreSQL em producao."
    )

# === SENHAS E VALIDAÇÃO ===
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 8}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# === TÓPICO 1: HASH E SALT (GESTÃO DE CREDENCIAIS) ===
# Argon2 é o hasher primário em produção por resistência superior a brute force.
# Os demais hashers permanecem para compatibilidade com hashes legados.
PASSWORD_HASHERS = [
    "vamos.hashers.ConfigurableArgon2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
    "django.contrib.auth.hashers.BCryptSHA256PasswordHasher",
]

# === INTERNACIONALIZAÇÃO ===
LANGUAGE_CODE = "pt-br"
TIME_ZONE = "America/Sao_Paulo"
USE_I18N = True
USE_TZ = True

# === ESTÁTICOS E MÍDIA ===
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [BASE_DIR / "static"]

if os.getenv('AWS_ACCESS_KEY_ID'):
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
    AWS_S3_REGION_NAME = 'us-east-1'
    AWS_S3_SIGNATURE_VERSION = 's3v4'
    AWS_DEFAULT_ACL = None
    AWS_S3_FILE_OVERWRITE = False
    AWS_S3_OBJECT_PARAMETERS = {
        'ServerSideEncryption': 'AES256',
    }

    STORAGES = {
        "default": {"BACKEND": "storages.backends.s3.S3Storage"},
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
            if DEBUG
            else "whitenoise.storage.CompressedManifestStaticFilesStorage"
        },
    }
    MEDIA_URL = f'https://{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/'
else:
    STORAGES = {
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
            if DEBUG
            else "whitenoise.storage.CompressedManifestStaticFilesStorage"
        },
    }
    MEDIA_URL = '/media/'
    MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# === AUTENTICAÇÃO E LOGIN ===
LOGIN_URL = 'two_factor:login'
LOGIN_REDIRECT_URL = "portal" 
LOGOUT_REDIRECT_URL = "login"
TERMOS_VERSAO = os.getenv("TERMOS_VERSAO", "2026-05")

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# === REST FRAMEWORK ===
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 100,
}

# === INTEGRAÇÕES E EMAIL ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
GEMINI_MODEL = "gemini-2.0-flash"

EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD')
DEFAULT_FROM_EMAIL = os.getenv('DEFAULT_FROM_EMAIL')

# ==============================================================================
# === TÓPICO 3: CONFIGURAÇÕES DE SESSÃO E SEGURANÇA AVANÇADA ===
# ==============================================================================
SESSION_COOKIE_AGE = 1800            
SESSION_SAVE_EVERY_REQUEST = True    
SESSION_EXPIRE_AT_BROWSER_CLOSE = True 

if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_SSL_REDIRECT = True
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    USE_X_FORWARDED_HOST = True
else:
    SESSION_COOKIE_SECURE = False
    CSRF_COOKIE_SECURE = False
    SECURE_SSL_REDIRECT = False

# --- PROTEÇÃO CONTRA FORÇA BRUTA (django-axes) ---
AUTHENTICATION_BACKENDS = [
    'axes.backends.AxesStandaloneBackend',
    'django.contrib.auth.backends.ModelBackend',
]
AXES_FAILURE_LIMIT = 5 # Bloqueia após 5 tentativas erradas
AXES_COOLOFF_TIME = 1  # Bloqueia por 1 hora
AXES_LOCKOUT_TEMPLATE = 'axes/lockout.html' # Opcional: página de erro
AXES_ENABLED = not RUNNING_TESTS

# --- RECUPERAÇÃO DE SENHA (Tópico 3 da Entrega 3) ---
# O token de redefinição de senha expira em 1 hora (3600 segundos)
PASSWORD_RESET_TIMEOUT = 3600

# --- LOGS DE SEGURANÇA E AUDITORIA ---
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "security_chain": {
            "class": "vamos.logging_handlers.HashChainAuditHandler",
            "formatter": "standard",
            "filename": SECURITY_AUDIT_LOG_PATH,
        },
    },
    "loggers": {
        "vamos.security": {
            "handlers": ["console", "security_chain"],
            "level": "INFO",
            "propagate": False,
        },
        "django.contrib.auth": {
            "handlers": ["console", "security_chain"],
            "level": "INFO",
            "propagate": False,
        },
        "axes.watch_login": {
            "handlers": ["console", "security_chain"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}
