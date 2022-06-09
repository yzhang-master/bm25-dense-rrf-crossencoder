from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class RankedDocument:
    doc_id: str
    score: float


def validate_documents(documents: list[Document]) -> None:
    if not documents:
        raise ValueError("at least one document is required")
    ids = [document.doc_id for document in documents]
    if any(not doc_id for doc_id in ids):
        raise ValueError("document ids must be non-empty")
    if len(ids) != len(set(ids)):
        raise ValueError("document ids must be unique")
