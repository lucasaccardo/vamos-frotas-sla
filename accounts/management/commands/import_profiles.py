import csv
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from accounts.models import Profile

class Command(BaseCommand):
    help = "Importa usuários do Supabase para User + Profile"

    def add_arguments(self, parser):
        parser.add_argument("file", type=str, help="Caminho do arquivo CSV de usuários")

    def handle(self, *args, **options):
        file_path = options["file"]

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normaliza o role para minúsculas
                role = (row.get("role") or "").lower()

                # Cria ou atualiza o User
                user, _ = User.objects.update_or_create(
                    username=row["username"],
                    defaults={
                        "email": row.get("email") or "",
                        "is_staff": True if role in ["admin", "superadmin"] else False,
                        "is_superuser": True if role == "superadmin" else False,
                    },
                )

                # Cria ou atualiza o Profile
                Profile.objects.update_or_create(
                    user=user,
                    defaults={
                        "matricula": row.get("matricula") or None,
                        "role": row.get("role") or None,
                        "status": row.get("status") or None,
                        "accepted_terms_on": row.get("accepted_terms_on") or None,
                        "force_password_reset": str(row.get("force_password_reset") or "").lower() in ["true", "1", "yes"],
                    },
                )

        self.stdout.write(self.style.SUCCESS("Importação de usuários concluída!"))
