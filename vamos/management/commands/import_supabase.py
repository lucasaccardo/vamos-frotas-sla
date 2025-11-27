import csv
import json
from django.core.management.base import BaseCommand
from vamos.models import Analise, Usuario

class Command(BaseCommand):
    help = "Importa dados exportados do Supabase (CSV ou JSON) para PostgreSQL"

    def add_arguments(self, parser):
        parser.add_argument("file", type=str, help="Caminho do arquivo CSV ou JSON")

    def handle(self, *args, **options):
        file_path = options["file"]

        if file_path.endswith(".csv"):
            self.import_csv(file_path)
        elif file_path.endswith(".json"):
            self.import_json(file_path)
        else:
            self.stdout.write(self.style.ERROR("Formato não suportado. Use CSV ou JSON."))

    def import_csv(self, file_path):
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                Analise.objects.update_or_create(
                    id=row["id"],
                    defaults={
                        "username": row["username"],
                        "tipo": row["tipo"],
                        "data_hora": row["data_hora"],
                        "dados_json": json.loads(row["dados_json"]) if row.get("dados_json") else {},
                        "pdf_path": row["pdf_path"],
                    },
                )
        self.stdout.write(self.style.SUCCESS("Importação CSV concluída!"))

    def import_json(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            for row in data:
                Analise.objects.update_or_create(
                    id=row["id"],
                    defaults={
                        "username": row["username"],
                        "tipo": row["tipo"],
                        "data_hora": row["data_hora"],
                        "dados_json": row.get("dados_json", {}),
                        "pdf_path": row.get("pdf_path", ""),
                    },
                )
        self.stdout.write(self.style.SUCCESS("Importação JSON concluída!"))
