import codecs
import json

import pytest

from eval.annotate import (
    NONE,
    QUIT,
    SELECT,
    SKIP,
    Selection,
    append_case,
    build_case,
    format_hits,
    load_questions,
    parse_selection,
    previous_work,
)
from eval.textio import read_text_utf8

HITS = [
    {"id": 41, "source": "is-kanunu.pdf", "score": 0.81, "text": "İşçi  \n dava açabilir."},
    {"id": 87, "source": "is-kanunu.pdf", "score": 0.77, "text": "Bir ay içinde başvurulur."},
    {"id": 12, "source": "kvkk-kanunu.pdf", "score": 0.55, "text": "Veri sorumlusu bildirir."},
]


# --- Soru dosyasını okuma ---------------------------------------------------


def test_comments_and_blank_lines_are_dropped(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text(
        "# başlık\n\nişten çıkardığım işçi dava açabilir mi\n   \n"
        "  # girintili yorum\nyıllık izin kaç gün\n",
        encoding="utf-8",
    )
    assert load_questions(path) == [
        "işten çıkardığım işçi dava açabilir mi",
        "yıllık izin kaç gün",
    ]


def test_a_byte_order_mark_does_not_corrupt_the_first_question(tmp_path):
    # Windows editörleri UTF-8 dosyanın başına BOM koyabiliyor
    path = tmp_path / "q.txt"
    path.write_bytes(codecs.BOM_UTF8 + "yıllık izin kaç gün\n".encode())

    assert load_questions(path) == ["yıllık izin kaç gün"]
    assert not read_text_utf8(path).startswith("﻿")


def test_a_bom_before_a_comment_line_is_also_handled(tmp_path):
    path = tmp_path / "q.txt"
    path.write_bytes(codecs.BOM_UTF8 + b"# yorum\nsoru burada\n")
    assert load_questions(path) == ["soru burada"]


def test_file_without_bom_still_reads(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("şirket unvanı nasıl seçilir\n", encoding="utf-8")
    assert load_questions(path) == ["şirket unvanı nasıl seçilir"]


def test_missing_or_empty_question_file_exits(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        load_questions(tmp_path / "yok.txt")

    empty = tmp_path / "q.txt"
    empty.write_text("# yalnızca yorum\n\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no questions"):
        load_questions(empty)


# --- Girdi yorumlama --------------------------------------------------------


def test_single_and_multiple_numbers():
    assert parse_selection("1", 3) == Selection(SELECT, (1,))
    assert parse_selection("1,3", 3) == Selection(SELECT, (1, 3))


@pytest.mark.parametrize("raw", ["1, 3", "1 3", " 1,3 ", "1,,3", "3,1"])
def test_number_lists_tolerate_spacing_and_order(raw):
    assert set(parse_selection(raw, 3).indices) == {1, 3}


def test_duplicate_numbers_are_collapsed():
    assert parse_selection("2,2,2", 3).indices == (2,)


def test_blank_means_no_answer():
    assert parse_selection("", 3).action == NONE
    assert parse_selection("   ", 3).action == NONE


@pytest.mark.parametrize("raw", ["s", "S", " s ", "skip"])
def test_s_skips(raw):
    assert parse_selection(raw, 3).action == SKIP


@pytest.mark.parametrize("raw", ["q", "Q", " q ", "quit"])
def test_q_quits(raw):
    assert parse_selection(raw, 3).action == QUIT


@pytest.mark.parametrize("raw", ["﻿1,2", "1,2﻿", "​1,2"])
def test_a_zero_width_character_on_the_typed_line_is_ignored(raw):
    # Windows'ta ilk satıra BOM karışabiliyor; yoksa "1,2" tanınmaz ve
    # soru sessizce "cevapsız" olarak kaydedilirdi
    assert parse_selection(raw, 3) == Selection(SELECT, (1, 2))


def test_a_lone_zero_width_character_still_means_no_answer():
    assert parse_selection("﻿", 3).action == NONE


def test_zero_width_does_not_break_skip_and_quit():
    assert parse_selection("﻿s", 3).action == SKIP
    assert parse_selection("﻿q", 3).action == QUIT


@pytest.mark.parametrize("raw", ["0", "4", "99"])
def test_out_of_range_numbers_are_rejected(raw):
    with pytest.raises(ValueError, match="out of range"):
        parse_selection(raw, 3)


@pytest.mark.parametrize("raw", ["x", "1,x", "bir", "-1", "1.5"])
def test_non_numeric_input_is_rejected(raw):
    with pytest.raises(ValueError, match="not a number"):
        parse_selection(raw, 3)


# --- Satır oluşturma --------------------------------------------------------


def test_case_records_selected_chunk_ids_and_sources():
    case = build_case(1, "işçi dava açabilir mi", [HITS[0], HITS[2]])

    assert case.id == "man001"
    assert case.relevant_chunk_ids == [41, 12]
    assert "MANUAL" in case.note
    assert "is-kanunu.pdf" in case.note
    assert "kvkk-kanunu.pdf" in case.note


def test_case_without_any_choice_is_marked_unanswered():
    case = build_case(7, "cevabı olmayan soru", [])

    assert case.id == "man007"
    assert case.relevant_chunk_ids == []
    assert "NO-HIT-IN-TOP10" in case.note


def test_case_ids_are_zero_padded():
    assert build_case(42, "s", [HITS[0]]).id == "man042"
    assert build_case(100, "s", [HITS[0]]).id == "man100"


# --- Kaldığı yerden devam ---------------------------------------------------


def test_previous_work_reports_questions_and_highest_id(tmp_path):
    answered = tmp_path / "a.jsonl"
    unanswered = tmp_path / "u.jsonl"
    append_case(answered, build_case(1, "birinci soru", [HITS[0]]))
    append_case(answered, build_case(2, "ikinci soru", [HITS[1]]))
    append_case(unanswered, build_case(3, "üçüncü soru", []))

    done, highest = previous_work([answered, unanswered])

    assert done == {"birinci soru", "ikinci soru", "üçüncü soru"}
    assert highest == 3


def test_previous_work_is_empty_when_nothing_was_written(tmp_path):
    assert previous_work([tmp_path / "yok.jsonl"]) == (set(), 0)


def test_append_keeps_earlier_rows(tmp_path):
    path = tmp_path / "a.jsonl"
    append_case(path, build_case(1, "birinci", [HITS[0]]))
    append_case(path, build_case(2, "ikinci", [HITS[1]]))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["man001", "man002"]
    assert rows[0]["question"] == "birinci"


def test_written_rows_survive_a_round_trip_with_turkish_text(tmp_path):
    path = tmp_path / "a.jsonl"
    append_case(path, build_case(1, "işçi çıkışında ne yapmalıyım", [HITS[0]]))
    [row] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert row["question"] == "işçi çıkışında ne yapmalıyım"


# --- Ekran çıktısı ----------------------------------------------------------


def test_listing_numbers_every_hit_and_shows_its_chunk_id():
    text = format_hits("işçi dava açabilir mi", HITS, 2, 40)

    assert "[2/40]" in text
    assert "işçi dava açabilir mi" in text
    for position, hit in enumerate(HITS, start=1):
        assert f"\n{position:>2}. id={hit['id']}" in text
    assert "is-kanunu.pdf" in text


def test_listing_collapses_whitespace_in_the_preview():
    assert "İşçi dava açabilir." in format_hits("s", HITS[:1], 1, 1)


def test_listing_handles_no_results():
    assert "(no results)" in format_hits("s", [], 1, 1)
