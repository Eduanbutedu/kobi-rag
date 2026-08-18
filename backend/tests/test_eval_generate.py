import pytest

from eval.generate_questions import (
    build_prompt,
    clean_question,
    generate_case,
    quality_flags,
    select_chunks,
)


def _chunks(count, length=300):
    return [
        {"id": i, "source": "a.pdf", "text": f"chunk {i} " + "x" * length}
        for i in range(1, count + 1)
    ]


def test_clean_question_accepts_a_plain_question():
    assert clean_question("Yıllık izin kaç gündür?") == "Yıllık izin kaç gündür?"


@pytest.mark.parametrize(
    "raw",
    [
        '"Yıllık izin kaç gündür?"',
        "1. Yıllık izin kaç gündür?",
        "- Yıllık izin kaç gündür?",
        "Soru: Yıllık izin kaç gündür?",
        "  Yıllık izin kaç gündür?  ",
    ],
)
def test_clean_question_strips_model_decoration(raw):
    assert clean_question(raw) == "Yıllık izin kaç gündür?"


def test_clean_question_takes_the_question_line_not_the_preamble():
    raw = "Elbette, işte sorunuz\nYıllık izin kaç gündür?"
    assert clean_question(raw) == "Yıllık izin kaç gündür?"


def test_clean_question_ignores_leading_blank_lines_from_no_think():
    raw = "\n\n\n\nYıllık izin kaç gündür?"
    assert clean_question(raw) == "Yıllık izin kaç gündür?"


def test_clean_question_extracts_from_a_narrated_reply():
    raw = 'Bu metnin cevapladığı soru şudur: "Yıllık izin kaç gündür?"'
    assert clean_question(raw) == "Yıllık izin kaç gündür?"


@pytest.mark.parametrize(
    "raw",
    [
        "Bu metinde, yıllık izin kaç gündür?",
        "Bu parçada yıllık izin kaç gündür?",
        "Yukarıdaki metne göre, yıllık izin kaç gündür?",
    ],
)
def test_clean_question_strips_meta_prefixes(raw):
    # Model kuralı yok sayıp "Bu metinde, ..." yazdığında soru kurtarılabiliyor
    assert clean_question(raw) == "Yıllık izin kaç gündür?"


def test_quality_flags_are_empty_for_a_good_question():
    assert quality_flags("FD002 alt kümesinde eğitim için kaç motor bulunmaktadır?") == []


def test_quality_flags_catch_meta_reference():
    assert "meta-reference" in quality_flags("Bir soru, bu metin neyi açıklamaktadır?")


def test_quality_flags_catch_yes_no_questions():
    assert "yes-no" in quality_flags("FD003 genelleme zorluklarını artırır mı?")
    assert "yes-no" in quality_flags("Bu yöntem diğerlerinden daha başarılı mıdır?")


def test_quality_flags_allow_wh_questions_ending_in_dir():
    assert "yes-no" not in quality_flags("C-MAPSS veri setinin amacı nedir?")


def test_quality_flags_catch_short_and_ascii_only_questions():
    assert "very-short" in quality_flags("Kaç motor var?")
    assert "maybe-not-turkish" in quality_flags("How many engines are in the FD002 subset?")


@pytest.mark.parametrize("raw", ["", "   ", "Bu bir cevaptır.", "Kısa?", "Soru yok"])
def test_clean_question_rejects_unusable_replies(raw):
    assert clean_question(raw) == ""


def test_select_chunks_is_reproducible_for_a_seed():
    chunks = _chunks(20)
    assert select_chunks(chunks, 5, seed=42, min_chars=200) == select_chunks(
        chunks, 5, seed=42, min_chars=200
    )


def test_select_chunks_varies_with_seed():
    chunks = _chunks(50)
    first = [c["id"] for c in select_chunks(chunks, 5, seed=1, min_chars=200)]
    second = [c["id"] for c in select_chunks(chunks, 5, seed=2, min_chars=200)]
    assert first != second


def test_select_chunks_skips_short_chunks():
    chunks = _chunks(3) + [{"id": 99, "source": "a.pdf", "text": "kısa"}]
    assert 99 not in [c["id"] for c in select_chunks(chunks, 10, seed=1, min_chars=200)]


def test_select_chunks_caps_at_available_count():
    assert len(select_chunks(_chunks(3), 10, seed=1, min_chars=200)) == 3


def test_select_chunks_without_long_enough_chunks_exits():
    with pytest.raises(SystemExit, match="min-chars"):
        select_chunks(_chunks(3, length=10), 2, seed=1, min_chars=200)


def test_build_prompt_includes_the_chunk_text():
    assert "izin metni" in build_prompt("izin metni")


def test_generate_case_marks_the_source_chunk_as_relevant(monkeypatch):
    monkeypatch.setattr(
        "eval.generate_questions.complete", lambda *a, **kw: "Yıllık izin kaç gündür?"
    )
    case = generate_case({"id": 7, "source": "izin.pdf", "text": "metin"}, index=3)
    assert case.id == "gen003"
    assert case.question == "Yıllık izin kaç gündür?"
    assert case.relevant_chunk_ids == [7]
    assert "review" in case.note.lower()
    assert "izin.pdf" in case.note


def test_generate_case_returns_none_for_unusable_reply(monkeypatch):
    monkeypatch.setattr("eval.generate_questions.complete", lambda *a, **kw: "Bir cevap.")
    assert generate_case({"id": 7, "source": "a.pdf", "text": "metin"}, index=1) is None


def test_generate_case_records_quality_flags_in_the_note(monkeypatch):
    monkeypatch.setattr(
        "eval.generate_questions.complete",
        lambda *a, **kw: "Bir soru, bu metin başarılı mıdır?",
    )
    case = generate_case({"id": 7, "source": "a.pdf", "text": "metin"}, index=1)
    assert "CHECK:" in case.note
    assert "meta-reference" in case.note
    assert "yes-no" in case.note
