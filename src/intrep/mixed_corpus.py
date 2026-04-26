from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MixedDocument:
    id: str
    modality: str
    content: str


def render_document(document: MixedDocument) -> str:
    return f"<doc type={document.modality} id={document.id}>\n{document.content}\n</doc>\n"


def render_corpus(documents: list[MixedDocument]) -> str:
    return "\n".join(render_document(document) for document in documents)


def default_mixed_documents() -> list[MixedDocument]:
    return [
        MixedDocument(
            id="ja_explain_001",
            modality="text",
            content="箱を開けると、中にある物体を観測できる。観測は行動の結果で変化する。",
        ),
        MixedDocument(
            id="en_explain_001",
            modality="text",
            content="A world model uses observations and actions to predict what will be observed next.",
        ),
        MixedDocument(
            id="env_symbolic_001",
            modality="environment_symbolic",
            content="<obs> key in box ; box closed <action> open box <next_obs> key visible",
        ),
        MixedDocument(
            id="env_natural_001",
            modality="environment_natural",
            content="鍵は箱の中にある。箱を開けると、鍵が見える。",
        ),
        MixedDocument(
            id="env_symbolic_002",
            modality="environment_symbolic",
            content="<obs> agent at desk ; cup on desk <action> move cup shelf <next_obs> cup on shelf",
        ),
        MixedDocument(
            id="env_natural_002",
            modality="environment_natural",
            content="机の上にカップがある。カップを棚へ移動すると、次の観測ではカップは棚にある。",
        ),
        MixedDocument(
            id="code_001",
            modality="code",
            content="def move(obj, target):\n    return f'{obj} is at {target}'",
        ),
        MixedDocument(
            id="log_001",
            modality="log",
            content="[tool] action=open_box status=ok observation=key_visible",
        ),
    ]
