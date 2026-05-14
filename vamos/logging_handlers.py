import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import logging


class HashChainAuditHandler(logging.Handler):
    """
    Handler append-only com encadeamento de hash para evidência de integridade.
    Cada linha armazena hash anterior + hash atual.
    """

    def __init__(self, filename):
        super().__init__()
        self.filepath = Path(filename)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self):
        if not self.filepath.exists():
            return "GENESIS"
        with self.filepath.open("rb") as file_obj:
            lines = file_obj.readlines()
        if not lines:
            return "GENESIS"
        try:
            payload = json.loads(lines[-1].decode("utf-8"))
            return payload.get("hash", "GENESIS")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "CORRUPTED"

    def emit(self, record):
        try:
            previous_hash = self._last_hash()
            message = self.format(record)
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
                "previous_hash": previous_hash,
            }
            serialized_event = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            event_hash = hashlib.sha256(serialized_event.encode("utf-8")).hexdigest()
            event["hash"] = event_hash
            with self.filepath.open("a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            self.handleError(record)
