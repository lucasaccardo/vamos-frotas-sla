from django.apps import AppConfig


class VamosConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'vamos'

    def ready(self):
        import vamos.security_logging  # noqa: F401
