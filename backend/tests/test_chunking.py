from app.services.chunking import chunk_text, normalize_text


def test_normalize_text_collapses_spacing() -> None:
    assert normalize_text("Hello   world\r\n\r\n\r\nNext") == "Hello world\n\nNext"


def test_chunk_text_uses_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(30))
    chunks = chunk_text(text, max_words=10, overlap=2)

    assert len(chunks) == 4
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]

