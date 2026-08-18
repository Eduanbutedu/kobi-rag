"""Boilerplate detection, checked against text taken verbatim from the corpus.

Each sample below is a real chunk from data/corpus, with its chunk id noted
so the judgement can be re-checked against the store.
"""

import pytest

from eval.chunk_filter import boilerplate_reasons, is_informative

# --- Gerçek korpustan: kullanılabilir metinler -------------------------------

# id=7412, turk-ticaret-kanunu.pdf
PROSE_TTK = (
    "VIII- Bagaj ve araçların zıyaı veya hasarından doğan sorumluluğun sınırı \n\n"
    "MADDE 1263- (1) Kabin bagajının uğradığı zıya veya hasardan dolayı taşıyanın \n"
    "sorumluluğu, hiçbir hâlde, her taşıma için yolcu başına 2.250 Özel Çekme Hakkını "
    "aşamaz. (2) Araçlar ve içlerinde veya üzerlerinde taşınan her çeşit eşya için "
    "taşıyanın sorumluluğu sınırlıdır."
)

# id=632, borclar-kanunu.pdf -- kısa başlık satırları ortalamayı düşürüyor ama metin gerçek
PROSE_SHORT_HEADINGS = (
    "B. Ücret \nI. Hak etme zamanı \n"
    "MADDE 521- Simsar, ancak yaptığı faaliyet sonucunda sözleşme kurulursa ücrete \n"
    "hak kazanır. Simsarın faaliyeti sonucunda kurulan sözleşme geciktirici koşula "
    "bağlanmışsa ücret, koşulun gerçekleşmesi hâlinde ödenir."
)

# id=1046, is-kanunu.pdf -- değişiklik notu içeriyor ama gövdesi gerçek madde metni
PROSE_WITH_AMENDMENT_NOTE = (
    "(Ek cümle: 4/4/2015-6645/37 md.) \n"
    "Ancak, turizm, özel güvenlik, sağlık hizmeti ve 30/5/2013 tarihli ve 6491 sayılı \n"
    "Türk Petrol Kanunu uyarınca petrol araştırma, arama ve sondaj faaliyetleri \n"
    "kapsamında yürütülen işlerde \n"
    "işçinin yazılı onayının alınması şartıyla denkleştirme süresi dört aya kadar artırılabilir."
)

# --- Gerçek korpustan: elenmesi gereken metinler -----------------------------

# id=1521, kvkk-orneklerle-rehber.pdf
TOC_DOTTED = (
    "AYDINLATMA YÜKÜMLÜLÜĞÜ .................................................................. 27\n"
    "7. İLGİLİ KİŞİNİN HAKLARI.....................................................28\n"
    "8. VERİ SORUMLUSUNA BAŞVURU..........................................................30\n"
)

# id=2187, kvkk-veri-guvenligi-rehberi.pdf -- nokta yerine boşluk "leader" kullanıyor
TOC_WHITESPACE = (
    "Kapsam   \n \n \n \n \n \n \n \n \n \n3\n1.3. Tanımlar  \n \n \n \n \n \n \n \n \n \n4\n"
    "2. KİŞİSEL VERİ GÜVENLİĞİNE İLİŞKİN İDARİ TEDBİRLER \n \n \n \n7\n"
    "2.1. Mevcut Risk ve Tehditlerin Belirlenmesi  \n \n \n \n \n \n8\n"
)

# id=2946, planli-alanlar-imar-yonetmeligi.pdf
AMENDMENT_TABLE = (
    "11/3/2020 \n31065 \n9. 23/1/2021 \n31373 \n10. 11/7/2021 \n31538 \n11. 1/8/2021 \n"
    "31555 \n12. 9/10/2021 \n31623 \n13. 25/2/2022 \n31761 \n14. 18/8/2022 \n31927 \n"
)

# id=1077, is-kanunu.pdf
REPEALED_ARTICLES = (
    "82 – (Mülga: 15/5/2008-5763/37 md.) \n \n"
    "İşçilerin hakları \nMadde 83 - (Mülga: 20/6/2012-6331/37 md.) \n \n"
    "İçki veya uyuşturucu madde kullanma yasağı \nMadde 84 - (Mülga: 20/6/2012-6331/37 md.) \n \n"
    "Ağır ve tehlikeli işler \nMadde 85 - (Mülga: 20/6/2012-6331/37 md.) \n"
)

# id=25, turbofan_paper_v10.pdf -- sayısal sonuç tablosu
NUMERIC_TABLE = (
    ".18 \n0.8254 \nSVR \n0.8991 \n18.70 \n13.42 \n0.7988 \n0.9054 \n19.17 \n14.50 \n"
    "0.7885 \nMLP \n0.9025 \n17.96 \n12.17 \n0.8149 \n0.9098 \n17.23 \n12.17 \n0.8291 \n"
)

# id=1510, kvkk-kanunu.pdf
AMENDMENT_LIST = (
    "esi \nKararının Numarası \n6698 Sayılı Kanunun Değişen \nveya İptal Edilen Maddeleri \n"
    "Yürürlüğe Giriş Tarihi \n7061 \n27, Geçici Madde 2 \n5/12/2017 \nKHK/703 \n19, 20, 21, 25 \n"
)


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("prose_ttk", PROSE_TTK),
        ("prose_short_headings", PROSE_SHORT_HEADINGS),
        ("prose_with_amendment_note", PROSE_WITH_AMENDMENT_NOTE),
    ],
)
def test_real_prose_is_kept(name, text):
    assert boilerplate_reasons(text) == [], f"{name} wrongly rejected"
    assert is_informative(text) is True


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("toc_dotted", TOC_DOTTED),
        ("toc_whitespace", TOC_WHITESPACE),
        ("amendment_table", AMENDMENT_TABLE),
        ("repealed_articles", REPEALED_ARTICLES),
        ("numeric_table", NUMERIC_TABLE),
        ("amendment_list", AMENDMENT_LIST),
    ],
)
def test_real_boilerplate_is_rejected(name, text):
    assert boilerplate_reasons(text), f"{name} wrongly kept"
    assert is_informative(text) is False


def test_each_signal_names_its_own_reason():
    assert "filler-runs" in boilerplate_reasons(TOC_DOTTED)
    assert "blank-line-leaders" in boilerplate_reasons(TOC_WHITESPACE)
    assert "digit-heavy" in boilerplate_reasons(AMENDMENT_TABLE)
    assert "repealed-articles" in boilerplate_reasons(REPEALED_ARTICLES)
    assert "no-sentence-like-line" in boilerplate_reasons(AMENDMENT_LIST)


def test_amendment_note_alone_does_not_reject_an_article():
    # Tek bir "(Ek cümle: ...)" notu gerçek madde metnini elemez;
    # eleme yalnızca blok baştan sona kayıtlardan oluşunca olmalı
    assert "repealed-articles" not in boilerplate_reasons(PROSE_WITH_AMENDMENT_NOTE)
    assert "repealed-articles" in boilerplate_reasons(REPEALED_ARTICLES)


def test_empty_text_is_not_informative():
    assert boilerplate_reasons("") == ["empty"]
    assert boilerplate_reasons("   \n  \n ") == ["empty"]
    assert is_informative("") is False


def test_a_single_long_sentence_is_informative():
    text = (
        "İşveren, işçinin yıllık ücretli izin hakkını iş sözleşmesinin devamı "
        "süresince kullandırmak zorundadır ve bu hakkından vazgeçilemez."
    )
    assert is_informative(text) is True


def test_line_wrapped_prose_survives_low_average_words():
    # PDF çıkarımı cümleyi tek tek kelimelere bölebiliyor; en uzun satır kuralı
    # bunu kurtarır, ortalama kelime sayısı tek başına elerdi
    text = (
        "Ödül sözü veren, giderlerinin ödenmesini isteyenlerin beklenen sonucu "
        "gerçekleştiremeyeceklerini ispat ederek bu yükümlülükten kurtulabilir.\n"
        "Ancak, \nbir \nya \nda \nbirden \nçok \nkişiye \nödenecek \ngiderlerin \ntoplamı"
    )
    assert is_informative(text) is True


def test_reasons_accumulate_for_thoroughly_bad_text():
    assert len(boilerplate_reasons(TOC_DOTTED)) >= 2
