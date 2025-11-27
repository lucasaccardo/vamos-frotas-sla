import csv, json
from django.core.management.base import BaseCommand
from vamos.models import Analise

class Command(BaseCommand):
    help = "Importa análises do Supabase (CSV) para PostgreSQL"

    def add_arguments(self, parser):
        parser.add_argument("file", type=str, help="Caminho do arquivo CSV de análises")

    def handle(self, *args, **options):
        file_path = options["file"]

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dados = json.loads(row["dados_json"])
                except Exception:
                    dados = {}

                Analise.objects.update_or_create(
                    id=row["id"],
                    defaults={
                        "username": row["username"],
                        "tipo": row["tipo"],
                        "data_hora": row["data_hora"],
                        "dados_json": dados,
                        "pdf_path": row.get("pdf_path") or None,
                    },
                )
        self.stdout.write(self.style.SUCCESS("Importação de análises concluída!"))
