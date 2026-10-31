import pytest

from cynbit.parser import ParsingError, clean_text, parse_bytes, parse_file


def test_parse_pdf(samples_dir):
    doc = parse_file(samples_dir / "resumes" / "ananya_rao_ml_student.pdf")
    assert doc.file_type == "pdf"
    assert doc.pages >= 1
    assert "Ananya Rao" in doc.text
    assert "PyTorch" in doc.text
    assert doc.word_count > 200


def test_parse_docx(samples_dir):
    doc = parse_file(samples_dir / "resumes" / "karthik_menon_fullstack.docx")
    assert doc.file_type == "docx"
    assert "React" in doc.text and "Node.js" in doc.text


def test_parse_txt(samples_dir):
    doc = parse_file(samples_dir / "resumes" / "rohan_das_needs_work.txt")
    assert "Rohan Das" in doc.text


def test_unsupported_extension():
    with pytest.raises(ParsingError):
        parse_bytes(b"hello", "resume.exe")


def test_empty_file():
    with pytest.raises(ParsingError):
        parse_bytes(b"", "resume.pdf")


def test_corrupted_pdf():
    with pytest.raises(ParsingError):
        parse_bytes(b"%PDF-1.4 this is not really a pdf" * 3, "broken.pdf")


def test_too_little_text():
    with pytest.raises(ParsingError):
        parse_bytes(b"just a few words", "short.txt")


def test_clean_text_normalises_bullets_and_ligatures():
    out = clean_text("•  Built   a model\n\n\n\n▪ Improved eﬃciency")
    assert out.split("\n")[0] == "- Built a model"
    assert "efficiency" in out
    assert "\n\n\n" not in out
