import os

from django.contrib.auth.hashers import Argon2PasswordHasher


class ConfigurableArgon2PasswordHasher(Argon2PasswordHasher):
    time_cost = int(os.getenv("DJANGO_ARGON2_TIME_COST", "3"))
    memory_cost = int(os.getenv("DJANGO_ARGON2_MEMORY_COST", "102400"))
    parallelism = int(os.getenv("DJANGO_ARGON2_PARALLELISM", "8"))
