from __future__ import annotations

# Compatibility facade. New code should import from signal_io,
# signal_rendering, or legacy_mixed_bridge directly.

from pathlib import Path

from intrep.legacy_mixed_bridge import (
    MODALITY_TO_CHANNEL,
    infer_channel_from_modality,
    load_corpus_jsonl_as_mixed_documents,
    load_signals_as_mixed_documents,
    mixed_document_to_signal,
    mixed_documents_to_signals,
    signal_to_mixed_document,
    signals_to_mixed_documents,
)
from intrep.mixed_corpus import MixedDocument
from intrep.signal_io import (
    first_json_record as _first_json_record,
    load_signals_jsonl,
    load_signals_jsonl_v2,
    payload_ref_from_record as _payload_ref_from_record,
    payload_ref_to_record as _payload_ref_to_record,
    reject_payload_refs,
    signal_from_record as _signal_from_record,
    signal_to_record as _signal_to_record,
    write_signals_jsonl_v2,
)
from intrep.signal_rendering import render_signal_corpus, render_signals_for_training
from intrep.signal_stream import render_signal, render_signal_stream
from intrep.signals import PayloadRef, Signal, render_payload_text
