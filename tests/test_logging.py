from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from orbit_causal.logging_utils import JsonlLogger


def test_jsonl_logger_protects_reserved_fields_and_fsyncs() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "events.jsonl"
        logger = JsonlLogger(path)
        with pytest.raises(ValueError, match="reserved"):
            logger.event("real_event", event="forged")
        with pytest.raises(ValueError, match="Out of range float"):
            logger.event("bad_event", value=float("nan"))
        logger.event("real_event", run_id="run")
        with patch("orbit_causal.logging_utils.os.fsync") as sync:
            logger.flush()
            sync.assert_called_once()
            logger.close()
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["event"] == "real_event"
        assert record["run_id"] == "run"
