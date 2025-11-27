from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    matricula = models.CharField(max_length=50, blank=True, null=True)
    role = models.CharField(max_length=50, blank=True, null=True)
    status = models.CharField(max_length=20, blank=True, null=True)
    accepted_terms_on = models.DateTimeField(blank=True, null=True)
    force_password_reset = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} ({self.role})"
