import logging

from django.contrib.auth import get_user_model
from django.contrib.auth import views as auth_views

logger = logging.getLogger("vamos.security")


def _mask_email(email):
    email = (email or "").strip()
    if "@" not in email:
        return "nao_informado"
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = "*" * len(local)
    else:
        masked_local = f"{local[:2]}***"
    return f"{masked_local}@{domain}"


class LoggedPasswordResetView(auth_views.PasswordResetView):
    def form_valid(self, form):
        email = form.cleaned_data.get("email")
        account_exists = get_user_model().objects.filter(email__iexact=email).exists()
        logger.info(
            "password_reset_requested email=%s account_exists=%s",
            _mask_email(email),
            account_exists,
        )
        return super().form_valid(form)


class LoggedPasswordResetConfirmView(auth_views.PasswordResetConfirmView):
    def dispatch(self, *args, **kwargs):
        response = super().dispatch(*args, **kwargs)
        if hasattr(self, "validlink") and not self.validlink:
            logger.warning("password_reset_invalid_or_expired uid=%s", kwargs.get("uidb64"))
        return response

    def form_valid(self, form):
        logger.info("password_reset_success uid=%s", self.kwargs.get("uidb64"))
        return super().form_valid(form)
