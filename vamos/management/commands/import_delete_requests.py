import csv
from django.core.management.base import BaseCommand
from vamos.models import DeleteRequest

class Command(BaseCommand):
    help = "Importa Delete Requests do Supabase (CSV) para PostgreSQL"

    def add_arguments(self, parser):
        parser.add_argument("file", type=str, help="Caminho do arquivo CSV de delete requests")

    def handle(self, *args, **options):
        file_path = options["file"]

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                DeleteRequest.objects.update_or_create(
                    id=row["id"],
                    defaults={
                        "analise_id": row.get("analise_id") or None,
                        "pdf_path": row.get("pdf_path") or None,
                        "protocolo": row.get("protocolo") or None,
                        "requested_by_username": row.get("requested_by_username") or None,
                        "request_date": row.get("request_date") or None,
                        "status": row.get("status") or None,
                        "reviewed_by_username": row.get("reviewed_by_username") or None,
                        "review_date": row.get("review_date") or None,
                        "rejection_reason": row.get("rejection_reason") or None,
                    },
                )
        self.stdout.write(self.style.SUCCESS("Importação de Delete Requests concluída!"))
