from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
import re
import shlex

from intrep.mixed_corpus import MixedDocument, write_mixed_documents_jsonl


ToolLogRecord = Mapping[str, object]


def load_tool_log_jsonl(
    path: str | Path,
    *,
    source_name: str = "tool_log",
    modality: str = "tool_log",
    limit: int | None = None,
) -> list[MixedDocument]:
    records: list[ToolLogRecord] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if limit is not None and len(records) >= limit:
            break
        if not line.strip():
            continue
        records.append(_load_jsonl_object(line, line_number))
    return adapt_tool_log_records(records, source_name=source_name, modality=modality)


def write_converted_tool_log_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_name: str = "tool_log",
    modality: str = "tool_log",
    limit: int | None = None,
) -> list[MixedDocument]:
    documents = load_tool_log_jsonl(
        input_path,
        source_name=source_name,
        modality=modality,
        limit=limit,
    )
    write_mixed_documents_jsonl(output_path, documents)
    return documents


def adapt_tool_log_records(
    records: Iterable[ToolLogRecord],
    *,
    source_name: str = "tool_log",
    modality: str = "tool_log",
) -> list[MixedDocument]:
    source_component = _safe_identifier_component(source_name)
    documents: list[MixedDocument] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"tool log record {index} must be an object")
        documents.append(
            MixedDocument(
                id=f"{source_component}_{_record_id(record, index)}",
                modality=modality,
                content=_render_tool_log_content(record, index),
            )
        )
    return documents


def adapt_tool_log_record(
    record: ToolLogRecord,
    *,
    source_name: str = "tool_log",
    modality: str = "tool_log",
    fallback_id: str,
) -> MixedDocument:
    documents = adapt_tool_log_records(
        [{**record, "_fallback_id": fallback_id}],
        source_name=source_name,
        modality=modality,
    )
    return documents[0]


def _render_tool_log_content(record: ToolLogRecord, index: int) -> str:
    observation = _render_observation(record)
    action = _render_action(record, index)
    terminal_text = _render_terminal_text(record)
    explicit_next_observation = _optional_text(
        record,
        (
            "next_observation",
            "next_obs",
            "after_observation",
            "after_state",
            "post_state",
            "state_after",
        ),
    )

    lines = [f"<obs> {observation}", f"<action> {action}"]
    if explicit_next_observation is not None and terminal_text is not None:
        lines.append(terminal_text)
        lines.append(f"<next_obs> {explicit_next_observation}")
    elif explicit_next_observation is not None:
        lines.append(f"<next_obs> {explicit_next_observation}")
    elif terminal_text is not None:
        lines.append(f"<next_obs> {terminal_text}")
    return " ".join(lines)


def _render_observation(record: ToolLogRecord) -> str:
    explicit = _optional_text(
        record,
        (
            "observation",
            "obs",
            "before_observation",
            "before_state",
            "pre_state",
            "state_before",
            "context",
            "prompt",
            "input",
        ),
    )
    parts: list[str] = []
    if explicit is not None:
        parts.append(explicit)
    for field_name in ("cwd", "workdir", "repo", "repository", "branch", "session_id"):
        if field_name in record and record[field_name] is not None:
            parts.append(f"{field_name}={_render_value(record[field_name])}")
    tool = _tool_name(record)
    if tool is not None:
        parts.append(f"tool={tool}")
    if not parts:
        return "tool execution context"
    return " ; ".join(parts)


def _render_action(record: ToolLogRecord, index: int) -> str:
    action = record.get("action")
    if isinstance(action, Mapping):
        text = _optional_text(
            action,
            ("text", "repr", "action", "command", "cmd", "name", "type"),
        )
        if text is not None:
            return text
    if isinstance(action, str) and action.strip():
        return action.strip()

    command = _optional_text(
        record,
        (
            "command",
            "cmd",
            "command_line",
            "shell_command",
            "operation",
            "operation_text",
            "action_text",
        ),
    )
    if command is not None:
        return command

    argv = record.get("argv")
    if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)):
        return _render_argv(argv)

    tool = _tool_name(record)
    if tool is not None:
        arguments = _first_present(
            record,
            (
                "arguments",
                "args",
                "parameters",
                "params",
                "input",
                "tool_input",
                "request",
            ),
        )
        if arguments is None:
            return tool
        return f"{tool} {_render_value(arguments)}"

    raise ValueError(f"tool log record {index} is missing an action or command field")


def _render_terminal_text(record: ToolLogRecord) -> str | None:
    if _is_error_record(record):
        error = _render_error(record)
        return f"<error> {error}" if error is not None else None
    result = _render_result(record)
    return f"<result> {result}" if result is not None else None


def _render_result(record: ToolLogRecord) -> str | None:
    parts: list[str] = []
    for field_name in (
        "result",
        "output",
        "stdout",
        "stderr",
        "return_value",
        "response",
        "summary",
    ):
        if field_name in record and record[field_name] is not None:
            parts.append(f"{field_name}={_render_value(record[field_name])}")
    for field_name in ("exit_code", "returncode", "status"):
        if field_name in record and record[field_name] is not None:
            parts.insert(0, f"{field_name}={_render_value(record[field_name])}")
    if not parts:
        return None
    return " ; ".join(parts)


def _render_error(record: ToolLogRecord) -> str | None:
    parts: list[str] = []
    for field_name in ("exit_code", "returncode", "status"):
        if field_name in record and record[field_name] is not None:
            parts.append(f"{field_name}={_render_value(record[field_name])}")
    for field_name in ("error", "exception", "message", "stderr", "output", "stdout"):
        if field_name in record and record[field_name] is not None:
            parts.append(f"{field_name}={_render_value(record[field_name])}")
    if not parts:
        return None
    return " ; ".join(parts)


def _is_error_record(record: ToolLogRecord) -> bool:
    for field_name in ("error", "exception"):
        value = record.get(field_name)
        if value is not None and _render_value(value):
            return True
    exit_code = record.get("exit_code", record.get("returncode"))
    if isinstance(exit_code, int):
        return exit_code != 0
    if isinstance(exit_code, str):
        try:
            return int(exit_code.strip()) != 0
        except ValueError:
            pass
    status = record.get("status")
    if isinstance(status, str):
        return status.strip().lower() in {
            "error",
            "failed",
            "failure",
            "timeout",
            "cancelled",
            "canceled",
        }
    return False


def _record_id(record: ToolLogRecord, index: int) -> str:
    value = _optional_text(
        record,
        (
            "id",
            "record_id",
            "event_id",
            "trace_id",
            "call_id",
            "invocation_id",
            "tool_call_id",
            "timestamp",
            "_fallback_id",
        ),
    )
    if value is None:
        return f"{index:06d}"
    return _safe_identifier_component(value)


def _tool_name(record: ToolLogRecord) -> str | None:
    return _optional_text(record, ("tool", "tool_name", "name", "function", "executor"))


def _optional_text(record: Mapping[str, object], field_names: tuple[str, ...]) -> str | None:
    for field_name in field_names:
        if field_name in record and record[field_name] is not None:
            text = _render_value(record[field_name])
            if text:
                return text
    return None


def _first_present(record: Mapping[str, object], field_names: tuple[str, ...]) -> object | None:
    for field_name in field_names:
        if field_name in record and record[field_name] is not None:
            return record[field_name]
    return None


def _render_argv(argv: Sequence[object]) -> str:
    if all(isinstance(part, str) for part in argv):
        return shlex.join(list(argv))
    return _render_value(list(argv))


def _render_value(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _load_jsonl_object(line: str, line_number: int) -> dict[str, object]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSONL record at line {line_number}: {error.msg}") from error
    if not isinstance(record, dict):
        raise ValueError(f"Invalid JSONL record at line {line_number}: expected object")
    return record


def _safe_identifier_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    if not component:
        raise ValueError("identifier components must contain at least one safe character")
    return component


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert local command/tool execution JSONL records into MixedDocument JSONL."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser(
        "convert-jsonl",
        help="Convert a local tool log JSONL file to MixedDocument JSONL.",
    )
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--source-name", default="tool_log")
    convert.add_argument("--modality", default="tool_log")
    convert.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "convert-jsonl":
        try:
            documents = write_converted_tool_log_jsonl(
                args.input,
                args.output,
                source_name=args.source_name,
                modality=args.modality,
                limit=args.limit,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(f"wrote {len(documents)} mixed documents to {args.output}")
        return

    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
