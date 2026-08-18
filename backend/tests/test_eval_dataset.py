import pytest

from eval.dataset import DatasetError, EvalCase, load_dataset, write_dataset

VALID_LINE = (
    '{"id": "q1", "question": "Yıllık izin kaç gün?", "relevant_chunk_ids": [4], "note": ""}'
)


def _write(tmp_path, *lines):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_loads_valid_rows(tmp_path):
    path = _write(tmp_path, VALID_LINE)
    [case] = load_dataset(path)
    assert case.id == "q1"
    assert case.question == "Yıllık izin kaç gün?"
    assert case.relevant_chunk_ids == [4]


def test_blank_lines_are_ignored(tmp_path):
    path = _write(tmp_path, VALID_LINE, "", "   ", VALID_LINE.replace('"q1"', '"q2"'))
    assert len(load_dataset(path)) == 2


def test_note_defaults_to_empty(tmp_path):
    path = _write(tmp_path, '{"id": "q1", "question": "S?", "relevant_chunk_ids": [1]}')
    assert load_dataset(path)[0].note == ""


def test_duplicate_chunk_ids_are_collapsed(tmp_path):
    path = _write(tmp_path, '{"id": "q1", "question": "S?", "relevant_chunk_ids": [3, 3, 5]}')
    assert load_dataset(path)[0].relevant_chunk_ids == [3, 5]


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "yok.jsonl")


def test_empty_dataset_raises(tmp_path):
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(_write(tmp_path, "", "  "))


def test_invalid_json_reports_line_number(tmp_path):
    path = _write(tmp_path, VALID_LINE, "{bozuk")
    with pytest.raises(DatasetError, match="line 2"):
        load_dataset(path)


def test_duplicate_ids_raise(tmp_path):
    path = _write(tmp_path, VALID_LINE, VALID_LINE)
    with pytest.raises(DatasetError, match="duplicate id"):
        load_dataset(path)


@pytest.mark.parametrize(
    "row",
    [
        '{"question": "S?", "relevant_chunk_ids": [1]}',
        '{"id": "", "question": "S?", "relevant_chunk_ids": [1]}',
        '{"id": "q1", "relevant_chunk_ids": [1]}',
        '{"id": "q1", "question": "   ", "relevant_chunk_ids": [1]}',
        '{"id": "q1", "question": "S?", "relevant_chunk_ids": []}',
        '{"id": "q1", "question": "S?", "relevant_chunk_ids": "4"}',
        '{"id": "q1", "question": "S?", "relevant_chunk_ids": ["4"]}',
        '{"id": "q1", "question": "S?", "relevant_chunk_ids": [true]}',
        '{"id": "q1", "question": "S?", "relevant_chunk_ids": [1], "note": 5}',
        '["not", "an", "object"]',
    ],
)
def test_malformed_rows_raise(tmp_path, row):
    with pytest.raises(DatasetError):
        load_dataset(_write(tmp_path, row))


def test_write_then_load_round_trips(tmp_path):
    path = tmp_path / "out.jsonl"
    cases = [
        EvalCase("q1", "Yıllık izin kaç gün?", [4], "not"),
        EvalCase("q2", "Fazla mesai ücreti nedir?", [7, 8]),
    ]
    assert write_dataset(path, cases) == 2
    assert load_dataset(path) == cases
