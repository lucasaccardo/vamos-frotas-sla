import csv
from django.core.management.base import BaseCommand
from vamos.models import Usuario

class Command(BaseCommand):
    help = "Importa usuários do Supabase (CSV) para PostgreSQL"

    def add_arguments(self, parser):
        parser.add_argument("file", type=str, help="Caminho do arquivo CSV de usuários")

    def handle(self, *args, **options):
        file_path = options["file"]

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                Usuario.objects.update_or_create(
                    username=row["username"],
                    defaults={
                        "password": row["password"],  # já vem hash
                        "role": row["role"],
                        "full_name": row["full_name"],
                        "matricula": row["matricula"],
                        "email": row["email"],
                        "status": row["status"],
                        "accepted_terms_on": row.get("accepted_terms_on") or None,
                        "reset_token": row.get("reset_token") or None,
                        "reset_expires_at": row.get("reset_expires_at") or None,
                        "last_password_change": row.get("last_password_change") or None,
                        "force_password_reset": row.get("force_password_reset") or None,
                    },
                )
        self.stdout.write(self.style.SUCCESS("Importação de usuários concluída!"))
