from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"orbit_causal.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        handler = logging.FileHandler(self.path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._handler: logging.FileHandler | None = handler

    def event(self, name: str, **fields: Any) -> None:
        reserved = {"timestamp_utc", "event"} & set(fields)
        if reserved:
            raise ValueError(f"event fields use reserved names: {sorted(reserved)}")
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "event": name,
            **fields,
        }
        self._logger.info(json.dumps(record, sort_keys=True, allow_nan=False))

    def flush(self) -> None:
        if self._handler is not None:
            self._handler.flush()
            stream = self._handler.stream
            if stream is not None:
                os.fsync(stream.fileno())

    def close(self) -> None:
        if self._handler is not None:
            self.flush()
            self._handler.close()
            self._logger.removeHandler(self._handler)
            self._handler = None
