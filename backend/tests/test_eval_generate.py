from collections import Counter

import pytest

from eval.generate_questions import (
    build_prompt,
    clean_question,
    format_filter_report,
    format_selection,
    generate_case,
    partition_chunks,
    preview_of,
    quality_flags,
    select_chunks,
)


def _prose(marker, length=300):
    """Sentence-like filler that survives the boilerplate filter."""
    sentence = (
        f"İşveren {marker} numaralı kayıt uyarınca işçinin yıllık ücretli izin "
        "hakkını sözleşme süresince kullandırmakla yükümlüdür. "
    )
    return (sentence * (length // len(sentence) + 1))[:length]


def _chunks(count, length=300, source="a.pdf", start=1):
    return [
        {"id": i, "source": source, "text": _prose(i, length)}
        for i in range(start, start + count)
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


# --- Soru mu, yoksa soru işareti iliştirilmiş düz cümle mi? -----------------

# 80 chunk'lık gerçek çalışmadan çıkan gerileme örnekleri
STATEMENTS_WITH_QUESTION_MARK = [
    "Kişisel verilerin nasıl yönetileceği açıkça belirlenmelidir.?",
    "İşyeri hekiminden alınan sağlık raporları işe başlatılamaz?",
    "Ormanların korunması, planlanması, yetiştirilmesi, işletilmesi, "
    "sınırlandırılması bu esaslara göre yapılır?",
]


@pytest.mark.parametrize("line", STATEMENTS_WITH_QUESTION_MARK)
def test_statements_with_a_question_mark_are_rejected(line):
    assert clean_question(line) == ""


@pytest.mark.parametrize(
    "line",
    [
        "Hangi belgeler başvuruda zorunludur?",
        "İşveren kaç gün içinde bildirim yapmak zorundadır?",
        "Kıdem tazminatı nasıl hesaplanır?",
        "Veri sorumlusu kimdir?",
        "Bu yükümlülük neden getirilmiştir?",
        "Başvuru nereye yapılır?",
        "Ödeme ne zaman yapılır?",
        "Tazminat tutarı ne kadardır?",
        "İtiraz süresi kaç gündür?",
        "Kurul hangisine karar verir?",
    ],
)
def test_real_questions_are_accepted(line):
    assert clean_question(line) == line


@pytest.mark.parametrize(
    "line",
    [
        "Bu karar kesin midir?",
        "Sözleşme feshedilebilir mi?",
        "Bu tutar yeterli mudur?",
        "Başvuru süresi uzatılmış mıdır?",
    ],
)
def test_questions_marked_only_by_the_particle_are_accepted(line):
    assert clean_question(line) == line


@pytest.mark.parametrize(
    "line",
    [
        # "ne" -> nedeniyle / "kim" -> kimyasal / "kaç" -> kaçınılmaz
        "Bu düzenleme mücbir sebep nedeniyle uygulanmamıştır?",
        "Tehlikeli kimyasal maddelerin listesi güncellenmiştir?",
        "Bu sonuç kaçınılmaz olarak ortaya çıkmıştır?",
        "İşyerinde kimlik doğrulaması yapılır?",
        "Kimse bu haktan vazgeçemez?",
        # "mi" -> mimari / miktar / milyon
        "Binanın mimari projesi onaylanır?",
        "Ödenecek miktar bankaya yatırılır?",
        "Bilanço tutarı 100 milyon lirayı aşamaz?",
        # cümle ortasındaki unvan "müdür" soru eki değildir
        "Genel müdür bu yetkiyi devredemez?",
    ],
)
def test_lookalike_words_do_not_make_a_statement_a_question(line):
    assert clean_question(line) == ""


@pytest.mark.parametrize(
    "line",
    [
        "Hangisinde bu şart aranır?",
        "Kurul hangisine karar verir?",
        "Bu oran kaçıncı maddede belirlenir?",
        "Sorumluluk kimlerdedir?",
        "Başvuru nerelerden yapılabilir?",
        "Bildirim nasıldır?",
    ],
)
def test_inflected_question_words_are_matched(line):
    assert clean_question(line) == line


def test_niyet_does_not_match_the_question_word_niye():
    assert clean_question("Tarafların niyeti sözleşmede belirtilir?") == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("İşveren kaç gün içinde bildirir ?", "İşveren kaç gün içinde bildirir?"),
        ("İşveren kaç gün içinde bildirir??", "İşveren kaç gün içinde bildirir?"),
    ],
)
def test_harmless_punctuation_is_normalised(raw, expected):
    assert clean_question(raw) == expected


@pytest.mark.parametrize(
    "line",
    [
        "Kişisel veriler nasıl korunur.?",
        "Kişisel veriler nasıl korunur .?",
        "Kişisel veriler nasıl korunur!?",
        "Kişisel veriler nasıl korunur;?",
    ],
)
def test_sentence_punctuation_before_the_question_mark_is_rejected(line):
    # Soru kelimesi olsa bile ".?" düz cümle imzasıdır
    assert clean_question(line) == ""


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


def test_select_chunks_skips_boilerplate():
    toc = {
        "id": 99,
        "source": "a.pdf",
        "text": "İÇİNDEKİLER\n1. GİRİŞ ...................................... 7\n"
        "2. KAPSAM ..................................... 9\n" * 3,
    }
    picked = select_chunks(_chunks(3) + [toc], 10, seed=1, min_chars=200)
    assert 99 not in [c["id"] for c in picked]


# --- Katmanlı örnekleme -----------------------------------------------------


def _corpus():
    """A skewed corpus: one document dominates, as in the real one."""
    return (
        _chunks(100, source="buyuk.pdf", start=1)
        + _chunks(10, source="orta.pdf", start=200)
        + _chunks(4, source="kucuk.pdf", start=300)
    )


def test_sampling_is_balanced_across_documents_not_proportional():
    # Rastgele seçimde buyuk.pdf payı ~%88 olurdu; katmanlı seçimde eşitlenmeli
    picked = select_chunks(_corpus(), 12, seed=42, min_chars=200)
    per_source = Counter(c["source"] for c in picked)

    assert len(picked) == 12
    assert per_source["buyuk.pdf"] == 4
    assert per_source["orta.pdf"] == 4
    assert per_source["kucuk.pdf"] == 4


def test_exhausted_document_gives_its_share_to_the_others():
    # kucuk.pdf yalnızca 4 chunk sunabiliyor; kalan kota diğerlerine dağılmalı
    picked = select_chunks(_corpus(), 24, seed=42, min_chars=200)
    per_source = Counter(c["source"] for c in picked)

    assert len(picked) == 24
    assert per_source["kucuk.pdf"] == 4
    assert per_source["buyuk.pdf"] + per_source["orta.pdf"] == 20


def test_selection_never_exceeds_what_the_corpus_holds():
    picked = select_chunks(_corpus(), 500, seed=42, min_chars=200)
    assert len(picked) == 114


def test_per_document_sets_a_hard_quota():
    picked = select_chunks(_corpus(), 999, seed=42, min_chars=200, per_document=3)
    per_source = Counter(c["source"] for c in picked)

    assert len(picked) == 9
    assert set(per_source.values()) == {3}


def test_per_document_does_not_redistribute_a_shortfall():
    # kucuk.pdf 4 chunk sunuyor; kota 6 olsa bile diğerleri 6'yı aşmamalı
    picked = select_chunks(_corpus(), 999, seed=42, min_chars=200, per_document=6)
    per_source = Counter(c["source"] for c in picked)

    assert per_source["kucuk.pdf"] == 4
    assert per_source["buyuk.pdf"] == 6
    assert per_source["orta.pdf"] == 6


def test_stratified_selection_is_deterministic_for_a_seed():
    first = select_chunks(_corpus(), 12, seed=7, min_chars=200)
    second = select_chunks(_corpus(), 12, seed=7, min_chars=200)
    assert [c["id"] for c in first] == [c["id"] for c in second]


def test_stratified_selection_varies_with_seed():
    first = [c["id"] for c in select_chunks(_corpus(), 12, seed=1, min_chars=200)]
    second = [c["id"] for c in select_chunks(_corpus(), 12, seed=2, min_chars=200)]
    assert first != second


# --- Eleme raporu -----------------------------------------------------------


def test_partition_separates_usable_from_short_and_boilerplate():
    chunks = _chunks(3) + [
        {"id": 98, "source": "a.pdf", "text": "kısa"},
        {"id": 99, "source": "a.pdf", "text": "1. GİRİŞ ....... 7\n2. KAPSAM ....... 9\n" * 12},
    ]
    usable, dropped = partition_chunks(chunks, min_chars=200)

    assert [c["id"] for c in usable] == [1, 2, 3]
    assert dropped["too-short"] == 1
    assert dropped["_total"] == 1
    assert dropped["filler-runs"] == 1


def test_filter_report_counts_each_category():
    usable, dropped = partition_chunks(
        _chunks(2) + [{"id": 9, "source": "a.pdf", "text": "kısa"}], min_chars=200
    )
    report = format_filter_report(3, usable, dropped)

    assert "1 too short" in report
    assert "2 usable" in report


def test_selection_report_lists_documents():
    report = format_selection(select_chunks(_corpus(), 6, seed=42, min_chars=200))
    assert "buyuk.pdf" in report
    assert "kucuk.pdf" in report


# --- İnceleme için ek alanlar ------------------------------------------------


def test_preview_collapses_whitespace_and_truncates():
    assert preview_of("bir  \n  iki\n\tüç") == "bir iki üç"
    assert len(preview_of("kelime " * 200)) == 200


def test_draft_carries_source_and_preview(monkeypatch):
    monkeypatch.setattr(
        "eval.generate_questions.complete", lambda *a, **kw: "Yıllık izin kaç gündür?"
    )
    chunk = {"id": 7, "source": "is-kanunu.pdf", "text": "İşveren  \n  izin vermekle yükümlüdür."}

    case = generate_case(chunk, index=1)
    row = case.to_dict()

    assert row["source"] == "is-kanunu.pdf"
    assert row["chunk_preview"] == "İşveren izin vermekle yükümlüdür."
    assert "candidate_chunk_ids" not in row  # store verilmediğinde retrieval yapılmaz


def test_draft_records_retrieval_candidates(monkeypatch):
    monkeypatch.setattr(
        "eval.generate_questions.complete", lambda *a, **kw: "Yıllık izin kaç gündür?"
    )
    hits = [
        {"id": 7, "source": "is-kanunu.pdf", "text": "yedi", "score": 0.9},
        {"id": 12, "source": "is-kanunu.pdf", "text": "oniki", "score": 0.8},
    ]
    monkeypatch.setattr("eval.generate_questions.retrieve", lambda store, q, k: hits)

    case = generate_case({"id": 7, "source": "is-kanunu.pdf", "text": "metin"}, 1, store=object())
    row = case.to_dict()

    assert row["candidate_chunk_ids"] == [7, 12]
    assert row["candidates"][1] == {"id": 12, "source": "is-kanunu.pdf", "preview": "oniki"}
    # Adaylar yalnızca öneri; ilgili kabul edilen hâlâ sadece kaynak chunk
    assert row["relevant_chunk_ids"] == [7]


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
