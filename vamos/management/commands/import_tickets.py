import csv
from django.core.management.base import BaseCommand
from vamos.models import Ticket

class Command(BaseCommand):
    help = "Importa Tickets de suporte do Supabase (CSV) para PostgreSQL"

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
                        "username": row.get("username") or None,
                        "full_name": row.get("full_name") or None,
                        "email": row.get("email") or None,
                        "assunto": row.get("assunto") or None,
                        "descricao": row.get("descricao") or None,
                        "status": row.get("status") or None,
                        "resposta": row.get("resposta") or None,
                        "data_criacao": row.get("data_criacao") or None,
                        "data_resposta": row.get("data_resposta") or None,
                        "anexo_path": row.get("anexo_path") or None,
                    },
                )
        self.stdout.write(self.style.SUCCESS("Importação de Tickets concluída!"))
