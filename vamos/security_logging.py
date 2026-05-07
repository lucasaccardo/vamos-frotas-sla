import logging

from django.contrib.auth.signals import user_logged_in, user_logged_out, user_login_failed
from django.dispatch import receiver

logger = logging.getLogger("vamos.security")


def _client_ip(request):
    if not request:
        return "desconhecido"
    forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "desconhecido")


@receiver(user_logged_in, dispatch_uid="vamos_user_logged_in")
def log_user_logged_in(sender, request, user, **kwargs):
    logger.info("auth_login_success user_id=%s ip=%s", user.id, _client_ip(request))


@receiver(user_logged_out, dispatch_uid="vamos_user_logged_out")
def log_user_logged_out(sender, request, user, **kwargs):
    user_id = getattr(user, "id", "anon")
    logger.info("auth_logout user_id=%s ip=%s", user_id, _client_ip(request))


@receiver(user_login_failed, dispatch_uid="vamos_user_login_failed")
def log_user_login_failed(sender, credentials, request, **kwargs):
    username = credentials.get("username") or credentials.get("email") or "desconhecido"
    logger.warning("auth_login_failed username=%s ip=%s", username, _client_ip(request))


try:
    from two_factor.signals import user_verified

    @receiver(user_verified, dispatch_uid="vamos_user_2fa_verified")
    def log_user_2fa_verified(sender, request, user, device, **kwargs):
        logger.info(
            "auth_2fa_verified user_id=%s device=%s ip=%s",
            user.id,
            device.__class__.__name__,
            _client_ip(request),
        )
except ImportError:
    logger.info("two_factor signal user_verified indisponível para logging de 2FA.")
