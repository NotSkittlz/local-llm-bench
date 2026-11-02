"""JSON and CSV writers for ``RunResult``. JSON is the canonical document."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llmbench.exceptions import ReportError
from llmbench.schema import SCHEMA_VERSION, RunResult


def write_json(result: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> RunResult:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read results {path}: {exc}") from exc
    try:
        result = RunResult.model_validate(raw)
    except Exception as exc:
        raise ReportError(f"invalid result schema in {path}: {exc}") from exc
    if result.schema_version.split(".", 1)[0] != SCHEMA_VERSION.split(".", 1)[0]:
        raise ReportError(
            f"{path} has schema {result.schema_version}, this tool speaks {SCHEMA_VERSION}"
        )
    return result


def write_csv(result: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "created_at",
        "data_lineage",
        "synthetic",
        "label",
        "suite",
        "prompt_id",
        "concurrency",
        "warmup",
        "ok",
        "error_kind",
        "ttft_ms",
        "e2e_ms",
        "decode_tps",
        "prefill_tps",
        "output_tokens",
        "prompt_tokens",
        "token_source",
        "model",
        "backend_type",
        "quantization",
        "temperature",
        "seed",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for trial in result.trials:
            writer.writerow(
                {
                    "run_id": result.run_id,
                    "created_at": result.created_at,
                    "data_lineage": result.data_lineage,
                    "synthetic": result.synthetic,
                    "label": result.label or "",
                    "suite": trial.suite,
                    "prompt_id": trial.prompt_id,
                    "concurrency": trial.concurrency,
                    "warmup": trial.warmup,
                    "ok": trial.ok,
                    "error_kind": trial.error_kind or "",
                    "ttft_ms": trial.ttft_ms,
                    "e2e_ms": trial.e2e_ms,
                    "decode_tps": trial.decode_tps,
                    "prefill_tps": trial.prefill_tps,
                    "output_tokens": trial.output_tokens,
                    "prompt_tokens": trial.prompt_tokens,
                    "token_source": trial.token_source or "",
                    "model": result.model.name,
                    "backend_type": result.backend.type,
                    "quantization": result.model.quantization or "",
                    "temperature": result.sampling.temperature,
                    "seed": result.config.seed,
                }
            )
