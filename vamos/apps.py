from django.apps import AppConfig
from django.conf import settings
from pathlib import Path
import os


class VamosConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'vamos'

    def ready(self):
        import vamos.security_logging  # noqa: F401
        try:
            security_log_file = Path(getattr(settings, "SECURITY_LOG_FILE", ""))
            if security_log_file:
                security_log_file.parent.mkdir(parents=True, exist_ok=True)
                if not security_log_file.exists():
                    security_log_file.touch()
                os.chmod(security_log_file.parent, 0o700)
                os.chmod(security_log_file, 0o600)
        except OSError:
            pass
