import csv
from django.core.management.base import BaseCommand
from tickets.models import Ticket

class Command(BaseCommand):
    help = "Importa tickets do Supabase (CSV) para PostgreSQL"

    def add_arguments(self, parser):
        parser.add_argument("file", type=str, help="Caminho do arquivo CSV de tickets")

    def handle(self, *args, **options):
        file_path = options["file"]

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                Ticket.objects.update_or_create(
                    id=row["id"],
                    defaults={
                        "username": row["username"],
                        "assunto": row.get("assunto") or "",
                        "descricao": row.get("descricao") or "",
                        "status": row.get("status") or "aberto",
                    },
                )
        self.stdout.write(self.style.SUCCESS("Importação de tickets concluída!"))
