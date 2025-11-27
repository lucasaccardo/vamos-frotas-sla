import csv
from django.core.management.base import BaseCommand
from vamos.models import UserNotification

class Command(BaseCommand):
    help = "Importa User Notifications do Supabase (CSV) para PostgreSQL"

    def add_arguments(self, parser):
        parser.add_argument("file", type=str, help="Caminho do arquivo CSV de notificações")

    def handle(self, *args, **options):
        file_path = options["file"]

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                UserNotification.objects.update_or_create(
                    id=row["id"],
                    defaults={
                        "username": row.get("username") or None,
                        "message": row.get("message") or None,
                        "created_at": row.get("created_at") or None,
                        "is_read": row.get("is_read") in ["true", "True", "1"],
                    },
                )
        self.stdout.write(self.style.SUCCESS("Importação de User Notifications concluída!"))

